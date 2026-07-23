"""Write and check CodaBench submission files.

Required format: a CSV named exactly `prediction` (no extension), header
`Sentence ID,Prediction`, zipped with the single entry `prediction`.

Every candidate was checked before upload: column names, 8,077 unique ids in the blind
file's order, integers in 1..19, no NaN, zip contents. The `changed` number (share of rows
that differ from the previous submission) catches a re-thresholding that did nothing, or
one that changed everything by mistake.
"""
import zipfile
from pathlib import Path

import pandas as pd

from . import paths

N_BLIND = 8077


def write(path, ids, pred):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Sentence ID": ids, "Prediction": [int(p) for p in pred]}).to_csv(path, index=False)
    with zipfile.ZipFile(str(path) + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname="prediction")
    return path


def blind_reference_ids():
    return pd.read_parquet(paths.DATA / "blind_sent.parquet")["Sentence ID"].astype(str).tolist()


def validate(path, reference_ids=None, previous=None, n_rows=N_BLIND):
    """Return {check: bool} for one submission file, plus a `changed` share if `previous`."""
    path = Path(path)
    p = pd.read_csv(path)
    ref = blind_reference_ids() if reference_ids is None else [str(i) for i in reference_ids]
    checks = {
        "columns": list(p.columns) == ["Sentence ID", "Prediction"],
        "rows": len(p) == n_rows == p["Sentence ID"].nunique(),
        "levels_1_19": p["Prediction"].dtype.kind == "i"
                       and bool(((p["Prediction"] >= 1) & (p["Prediction"] <= 19)).all()),
        "no_nan": not bool(p.isna().any().any()),
        "ids_aligned": list(p["Sentence ID"].astype(str)) == ref,
    }
    zip_path = Path(str(path) + ".zip")
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            checks["zip_entry"] = z.namelist() == ["prediction"]
    if previous is not None:
        prev = pd.read_csv(previous)["Prediction"].values
        checks["changed"] = float((p["Prediction"].values != prev).mean())
    return checks


def report(path, reference_ids=None, previous=None):
    checks = validate(path, reference_ids, previous)
    for k, v in checks.items():
        if k == "changed":
            print(f"  {k:12s} {v*100:.1f}% of rows differ from the previous submission")
        else:
            print(f"  {k:12s} {'OK' if v else 'FAIL'}")
    hard = [v for k, v in checks.items() if k != "changed"]
    print("  ALL PASS:", all(hard))
    return all(hard)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validate a submission file")
    ap.add_argument("path")
    ap.add_argument("--previous", default=None)
    args = ap.parse_args()
    raise SystemExit(0 if report(args.path, previous=args.previous) else 1)
