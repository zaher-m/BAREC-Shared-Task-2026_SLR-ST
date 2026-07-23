"""Build the two text variants from the official BAREC parquet splits.

raw   : light normalization only. Diacritics are KEPT on purpose: the easy BAREC levels are
        vocalized children's text, so diacritics are a readability signal. Only ~2% of
        sentences change here at all.
d3tok : CAMeL Tools morphological segmentation (base + clitics), same scheme the highest-scoring 2025 system
        used. Worth ~+6-7 QWK over raw on bert-large-arabertv2 (see e04), but AraModernBERT
        does better on raw, so pick per backbone.

Writes data/proc_{variant}_{split}.parquet with columns [ID, text, label19].
"""
import argparse
import re
import time

import pandas as pd

from . import paths

KASHIDA = "ـ"


def light_norm(s: str) -> str:
    s = s.replace(KASHIDA, "")
    return re.sub(r"\s+", " ", s).strip()


def build_d3tok():
    """Return a text -> segmented text function. Loads the CAMeL MLE disambiguator (slow)."""
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer
    from camel_tools.tokenizers.word import simple_word_tokenize

    mle = MLEDisambiguator.pretrained("calima-msa-r13")
    tok = MorphologicalTokenizer(disambiguator=mle, scheme="d3tok", split=True, diac=False)

    def fn(s: str) -> str:
        s = light_norm(s)
        if not s:
            return s
        return " ".join(tok.tokenize(simple_word_tokenize(s)))

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["raw", "d3tok"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = ap.parse_args()

    tfn = build_d3tok() if args.variant == "d3tok" else light_norm

    for split in args.splits:
        df = pd.read_parquet(paths.DATA / f"barec_sent_{split}.parquet")
        n = len(df)
        out_text, t0 = [], time.time()
        for i, s in enumerate(df["Sentence"].astype(str).tolist()):
            out_text.append(tfn(s))
            if args.variant == "d3tok" and (i + 1) % 5000 == 0:
                el = time.time() - t0
                print(f"  [{split}] {i+1}/{n}  {el:.0f}s  ({(i+1)/el:.0f} sent/s)", flush=True)
        out = pd.DataFrame({
            "ID": df["ID"].astype(str).values,
            "text": out_text,
            "label19": df["Readability_Level_19"].astype(int).values,
        })
        dst = paths.split_file(args.variant, split)
        out.to_parquet(dst)
        print(f"[{split}] {n} rows -> {dst}", flush=True)


if __name__ == "__main__":
    main()
