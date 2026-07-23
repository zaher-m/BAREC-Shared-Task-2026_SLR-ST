"""Run an ensemble config over unlabeled sentences and write a submission.

Loads each member in turn, preprocesses the text with whatever variant that member was
trained on, and caches the per-member scores next to the ensemble score. The cache is the
reason later submissions needed no GPU: they are just new thresholds on the same matrix.

Example:
    python -m slra_st.infer --input data/blind_sent.parquet --ensemble final_v8 \
                            --out submissions/submitted/prediction_final
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from . import paths, submission
from .calibration import labels_from_thresholds
from .train import Encoder, PadCollator, SentDS, predict_scores


def read_input(path, id_col, text_col):
    path = str(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    return df[id_col].astype(str).tolist(), df[text_col].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", default="Sentence ID")
    ap.add_argument("--text-col", default="Sentence")
    ap.add_argument("--ensemble", default="final_v8", help="config name under configs/ensembles/ or a path")
    ap.add_argument("--out", default=str(paths.PREDICTIONS / "blind_prediction"))
    ap.add_argument("--pre-d3tok", action="store_true",
                    help="input text is already preprocessed; skip CAMeL (also used for raw)")
    args = ap.parse_args()

    ens = json.load(open(paths.ensemble(args.ensemble)))
    weights = np.asarray(ens["weights"], float)
    weights /= weights.sum()
    thresholds = np.asarray(ens["thresholds"], float)

    ids, texts = read_input(args.input, args.id_col, args.text_col)

    metas = {tag: json.load(open(paths.checkpoint(tag) / "meta.json")) for tag in ens["members"]}
    variants = {m["variant"] for m in metas.values()}

    if args.pre_d3tok:
        proc = {v: texts for v in variants}
    else:
        from .preprocess import light_norm
        proc = {}
        if "raw" in variants:
            proc["raw"] = [light_norm(t) for t in texts]
        if "d3tok" in variants:
            from .preprocess import build_d3tok
            tfn = build_d3tok()
            proc["d3tok"] = [tfn(t) for t in texts]

    device = "cuda"
    member_scores = []
    for tag in ens["members"]:
        meta = metas[tag]
        ckpt = paths.checkpoint(tag)
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        model = Encoder(meta["model"], meta["objective"], lora=meta.get("lora", False)).to(device)
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device),
                              strict=not meta.get("lora", False))
        base = pd.DataFrame({"ID": ids, "text": proc[meta["variant"]], "label19": [1] * len(ids)})
        ds = SentDS(base, tok, meta["max_len"])
        # bs=256: sequences are short and dynamic padding keeps the batches small, so a big
        # batch is much faster here. Mattered a lot when the GPU was shared.
        loader = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=PadCollator(tok))
        member_scores.append(predict_scores(model, loader, meta["objective"], device))
        del model
        torch.cuda.empty_cache()
        print(f"scored {tag}")

    X = np.vstack(member_scores).T
    ens_score = X @ weights
    pred = labels_from_thresholds(ens_score, thresholds).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths.BLIND_SCORES.mkdir(parents=True, exist_ok=True)
    cache = paths.BLIND_SCORES / f"{out_path.name}_scores.npz"
    np.savez(cache, ids=np.array(ids), members=np.array(ens["members"]),
             weights=np.array(ens["weights"]), member_scores=X, ens_score=ens_score)
    print(f"cached member scores -> {cache}")

    submission.write(out_path, ids, pred)
    print(f"wrote {out_path} and {out_path}.zip  ({len(ids)} rows)")


if __name__ == "__main__":
    main()
