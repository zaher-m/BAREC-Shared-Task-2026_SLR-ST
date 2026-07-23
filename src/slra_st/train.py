"""Fine-tune one Arabic encoder with one ordinal objective on BAREC sentences.

One trainer for all 63 runs. Each run is one (backbone, objective, text variant, data
regime); the scripts in experiments/ sweep it.

Objectives, all giving a continuous level on the 1..19 scale so members can be averaged:
  reg  : MSE on the scalar output
  corn : CORN ordinal regression, 18 conditional binary tasks
  soft : SORD-style soft labels (T=2.0) + KL
  wkl  : cross-entropy + 0.5 * soft-QWK penalty
  emd  : squared Earth-Mover distance between the predicted and one-hot CDFs
  oll  : ordinal log-loss (implemented but never used in a run)
Five objectives on ONE good backbone turned out to decorrelate better than five different
backbones, which is why the ensemble is mostly AraBERTv2.

Data regimes: train-only, --alldata (train+dev, test becomes the holdout), --fold N
(k-fold by document over all labeled data), --pseudo (append pseudo-labeled rows).

Saves per run: best checkpoint, cached dev/test scores, and a meta json with the model's
own thresholds.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from . import paths
from .calibration import QWKThresholdOptimizer
from .metrics import N_CLASSES, print_report, qwk


class SentDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.ids = df["ID"].astype(str).tolist()
        self.labels = df["label19"].astype(int).tolist()
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        # No padding here, the collator pads per batch. Sequences are much shorter than
        # max_len=160 (d3tok train p99 = 78 tokens), so fixed padding wasted most of the
        # compute. Safe to do because masked mean pooling ignores pad positions: same
        # logits, 1.3-1.4x faster epochs.
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        item = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        item["label"] = self.labels[i] - 1
        return item


class PadCollator:
    """Pad each batch to its own longest sequence.

    A class and not a closure so DataLoader workers can pickle it.
    """

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, features):
        labels = torch.tensor([f.pop("label") for f in features], dtype=torch.long)
        batch = self.tok.pad(features, padding=True, return_tensors="pt")
        batch["label"] = labels
        return batch


class Encoder(nn.Module):
    """Pretrained encoder, masked mean pooling, one linear head."""

    def __init__(self, name, objective, dropout=0.1, lora=False):
        super().__init__()
        if lora:
            # 7B backbone: bf16 weights + LoRA on the attention projections only.
            from peft import LoraConfig, get_peft_model
            bb = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16)
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            self.backbone = get_peft_model(bb, cfg)
        else:
            self.backbone = AutoModel.from_pretrained(name, trust_remote_code=True)
        h = self.backbone.config.hidden_size
        self.objective = objective
        self.drop = nn.Dropout(dropout)
        out = 1 if objective == "reg" else (N_CLASSES - 1 if objective == "corn" else N_CLASSES)
        self.head = nn.Linear(h, out)

    def forward(self, **batch):
        model_in = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = self.backbone(**model_in)
        last = out.last_hidden_state
        mask = model_in["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.head(self.drop(pooled))


def corn_loss(logits, y, num_classes):
    """CORN loss (Shi et al. / coral_pytorch), written out here to skip the dependency."""
    losses, n_terms = 0.0, 0
    for k in range(num_classes - 1):
        mask = y > (k - 1) if k > 0 else torch.ones_like(y, dtype=torch.bool)  # keep y >= k
        if mask.sum() == 0:
            continue
        lg = logits[mask, k]
        target = (y[mask] > k).float()
        losses = losses + nn.functional.binary_cross_entropy_with_logits(lg, target, reduction="sum")
        n_terms += mask.sum().item()
    return losses / max(n_terms, 1)


def corn_scores(logits):
    """Expected level (1..19) from CORN logits, via cumulative sigmoids."""
    probs = torch.sigmoid(logits)
    cum = torch.cumprod(probs, dim=1)  # P(y > k)
    return 1.0 + cum.sum(dim=1)


_LEVELS = torch.arange(N_CLASSES).float()
_WMAT = (torch.arange(N_CLASSES).float()[:, None] - torch.arange(N_CLASSES).float()[None, :]) ** 2 / (N_CLASSES - 1) ** 2


def soft_targets(y, T=2.0):
    lv = _LEVELS.to(y.device)
    d2 = (lv[None, :] - y[:, None].float()) ** 2
    return torch.softmax(-d2 / T, dim=1)  # unimodal, peaked at y


def soft_ce_loss(logits, y, T=2.0):
    logp = torch.log_softmax(logits, dim=1)
    return nn.functional.kl_div(logp, soft_targets(y, T), reduction="batchmean")


def wkl_loss(logits, y, lam=0.5):
    """Cross-entropy + lam * soft-QWK (de la Torre). num/den is roughly 1 - kappa."""
    ce = nn.functional.cross_entropy(logits, y)
    p = torch.softmax(logits, dim=1)
    W = _WMAT.to(logits.device)
    num = (W[y] * p).sum()
    hist = torch.bincount(y, minlength=N_CLASSES).float()
    mean_p = p.sum(0)
    den = (W * torch.outer(hist, mean_p)).sum() / len(y) + 1e-6
    return ce + lam * (num / den)


def emd_loss(logits, y):
    p = torch.softmax(logits, dim=1)
    oh = nn.functional.one_hot(y, N_CLASSES).float()
    return ((p.cumsum(1) - oh.cumsum(1)) ** 2).sum(1).mean()


def oll_loss(logits, y, alpha=1.5):
    p = torch.softmax(logits, dim=1).clamp(1e-6, 1 - 1e-6)
    lv = _LEVELS.to(y.device)
    d = (lv[None, :] - y[:, None].float()).abs() ** alpha
    return (-torch.log(1 - p) * d).sum(1).mean()


def expected_scores(logits):
    """Expected level 1..19 from a 19-way head."""
    p = torch.softmax(logits, dim=1)
    lv = (torch.arange(N_CLASSES).float() + 1.0).to(logits.device)
    return (p * lv[None, :]).sum(1)


def compute_loss(objective, logits, y):
    if objective == "reg":  return nn.functional.mse_loss(logits.squeeze(-1), y.float() + 1.0)
    if objective == "corn": return corn_loss(logits, y, N_CLASSES)
    if objective == "soft": return soft_ce_loss(logits, y)
    if objective == "wkl":  return wkl_loss(logits, y)
    if objective == "emd":  return emd_loss(logits, y)
    if objective == "oll":  return oll_loss(logits, y)
    raise ValueError(objective)


def predict_scores(model, loader, objective, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "label"}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            if objective == "reg":
                s = logits.squeeze(-1).float()
            elif objective == "corn":
                s = corn_scores(logits.float())
            else:
                s = expected_scores(logits.float())
            scores.append(s.cpu().numpy())
    return np.concatenate(scores)


def build_splits(args):
    """Load the three splits and apply whichever data regime was asked for."""
    tr = pd.read_parquet(paths.split_file(args.variant, "train"))
    dv = pd.read_parquet(paths.split_file(args.variant, "validation"))
    te = pd.read_parquet(paths.split_file(args.variant, "test"))

    if args.fold >= 0:
        # k-fold over ALL labeled data, split BY DOCUMENT so the held-out fold is really
        # unseen (sentences from one document never end up on both sides). md5 folding is
        # deterministic, so there is no fold map to store.
        import hashlib
        pool = pd.concat([tr, dv, te], ignore_index=True)
        docmap = {}
        for f_ in ("barec_sent_train", "barec_sent_validation", "barec_sent_test"):
            m = pd.read_parquet(paths.DATA / f"{f_}.parquet")[["ID", "Document"]]
            docmap.update(dict(zip(m["ID"].astype(str), m["Document"].astype(str))))
        docs = pool["ID"].astype(str).map(lambda i: docmap.get(i, i))
        folds = docs.map(lambda d: int(hashlib.md5(str(d).encode()).hexdigest(), 16) % args.nfolds)
        tr = pool[folds != args.fold].reset_index(drop=True)
        dv = pool[folds == args.fold].reset_index(drop=True)
        te = dv
        print(f"[fold {args.fold}/{args.nfolds}] train={len(tr)} holdout(OOF)={len(dv)} "
              f"docs_held={docs[folds==args.fold].nunique()}", flush=True)
    elif args.alldata:
        # +13% data: dev goes into training and TEST becomes the early-stop/calibration
        # holdout. Blind is never used for either.
        tr = pd.concat([tr, dv], ignore_index=True)
        dv = te
        print(f"[alldata] train={len(tr)} (train+dev), holdout=test={len(te)}", flush=True)

    if args.pseudo:
        # Pseudo-labeled blind sentences go into TRAINING only, never the holdout.
        ps = pd.read_parquet(args.pseudo)
        tr = pd.concat([tr, ps[["ID", "text", "label19"]]], ignore_index=True)
        print(f"[pseudo] +{len(ps)} pseudo-labeled -> train={len(tr)}", flush=True)

    return tr, dv, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short id used in output filenames")
    ap.add_argument("--variant", default="d3tok", choices=["raw", "d3tok"])
    ap.add_argument("--objective", default="reg", choices=["reg", "corn", "soft", "wkl", "emd", "oll"])
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--upsample", action="store_true",
                    help="sqrt inverse-frequency sampling. Only used in the first campaign, "
                         "then dropped because the A/B was mixed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alldata", action="store_true",
                    help="train on train+dev; hold out TEST for early-stop and calibration")
    ap.add_argument("--fold", type=int, default=-1,
                    help="k-fold mode: pool all labeled data, hold out DOCUMENT-fold f")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--lora", action="store_true", help="bf16 backbone + attention LoRA (7B scale)")
    ap.add_argument("--pseudo", default="",
                    help="parquet of pseudo-labeled rows (ID/text/label19) appended to TRAINING")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    tr, dv, te = build_splits(args)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # LLM tokenizers usually have no pad token
    tr_ds, dv_ds, te_ds = (SentDS(tr, tok, args.max_len), SentDS(dv, tok, args.max_len),
                           SentDS(te, tok, args.max_len))
    collate = PadCollator(tok)

    if args.upsample:
        labels = np.array(tr_ds.labels)
        freq = np.bincount(labels, minlength=N_CLASSES).astype(float)
        w_per_class = 1.0 / np.sqrt(np.clip(freq, 1, None))
        sampler = WeightedRandomSampler(torch.as_tensor(w_per_class[labels], dtype=torch.double),
                                        len(labels), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, num_workers=4,
                               pin_memory=True, collate_fn=collate)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4,
                               pin_memory=True, collate_fn=collate)
    dv_loader = DataLoader(dv_ds, batch_size=64, num_workers=4, pin_memory=True, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=64, num_workers=4, pin_memory=True, collate_fn=collate)

    model = Encoder(args.model, args.objective, lora=args.lora).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    best_qwk, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for batch in tr_loader:
            y = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
                loss = compute_loss(args.objective, logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        dv_scores = predict_scores(model, dv_loader, args.objective, device)
        dv_pred = np.clip(np.rint(dv_scores), 1, 19).astype(int)
        q = qwk(dv_ds.labels, dv_pred)
        print(f"ep{ep} loss={run/len(tr_loader):.4f} dev_QWK(naive)={q*100:.3f} {time.time()-t0:.0f}s",
              flush=True)
        if q > best_qwk:
            best_qwk, bad = q, 0
            # With LoRA save only the trainable tensors, a full 7B copy would be ~14GB.
            keep = {k for k, p in model.named_parameters() if p.requires_grad} if args.lora else None
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if keep is None or k in keep}
        else:
            bad += 1
            if bad > args.patience:
                print("early stop", flush=True)
                break

    model.load_state_dict(best_state, strict=False)
    dv_scores = predict_scores(model, dv_loader, args.objective, device)
    te_scores = predict_scores(model, te_loader, args.objective, device)

    # These thresholds are just for the report. Ensembles refit thresholds on the combined
    # score, so nothing downstream reads them.
    opt_th = QWKThresholdOptimizer().fit(dv_scores, dv_ds.labels)
    dv_pred, te_pred = opt_th.predict(dv_scores), opt_th.predict(te_scores)
    print_report(f"{args.tag} DEV  cal", dv_ds.labels, dv_pred)
    print_report(f"{args.tag} TEST cal", te_ds.labels, te_pred)

    paths.MEMBER_SCORES.mkdir(parents=True, exist_ok=True)
    paths.METADATA.mkdir(parents=True, exist_ok=True)
    np.savez(paths.member_scores(args.tag),
             dev_ids=np.array(dv_ds.ids), dev_scores=dv_scores, dev_labels=np.array(dv_ds.labels),
             test_ids=np.array(te_ds.ids), test_scores=te_scores, test_labels=np.array(te_ds.labels))

    ckpt_dir = paths.checkpoint(args.tag)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / "model.pt")
    tok.save_pretrained(ckpt_dir)
    meta = {"model": args.model, "variant": args.variant, "objective": args.objective,
            "lora": bool(args.lora), "max_len": args.max_len, "tag": args.tag,
            "dev_qwk_cal": qwk(dv_ds.labels, dv_pred) * 100,
            "thresholds": opt_th.thresholds_.tolist()}
    # Written twice on purpose: next to the checkpoint, and into artifacts/metadata/ which
    # is tracked, so the record survives even if the weights are deleted.
    json.dump(meta, open(paths.meta(args.tag), "w"), indent=2)
    json.dump(meta, open(ckpt_dir / "meta.json", "w"), indent=2)
    print(f"saved {paths.member_scores(args.tag).name} and checkpoint {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
