"""Repository layout in one place, so scripts never hard-code relative paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
ENSEMBLES = CONFIGS / "ensembles"
THRESHOLDS = CONFIGS / "thresholds"

ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
MEMBER_SCORES = ARTIFACTS / "scores" / "members"
BLIND_SCORES = ARTIFACTS / "scores" / "blind"
METADATA = ARTIFACTS / "metadata"
LOGS = ARTIFACTS / "logs"
PREDICTIONS = ARTIFACTS / "predictions"
DIAGNOSTICS = ARTIFACTS / "diagnostics"

# Scratch output for re-runs. Gitignored, so re-running an experiment can never overwrite a
# committed config, diagnostic or prediction. Anything an experiment writes by default belongs
# here; the committed originals are only replaced deliberately, with an explicit --out.
REGENERATED = ARTIFACTS / "regenerated"

SUBMISSIONS = ROOT / "submissions"
SUBMITTED = SUBMISSIONS / "submitted"
CANDIDATES = SUBMISSIONS / "candidates"


def member_scores(tag: str) -> Path:
    return MEMBER_SCORES / f"scores_{tag}.npz"


def meta(tag: str) -> Path:
    return METADATA / f"meta_{tag}.json"


def checkpoint(tag: str) -> Path:
    return MODELS / tag


def ensemble(name: str) -> Path:
    """Accept a bare config name ('v6', 'research/psfam') or an explicit path."""
    p = Path(name)
    if p.suffix == ".json" and p.exists():
        return p
    return ENSEMBLES / f"{name}.json"


def split_file(variant: str, split: str) -> Path:
    return DATA / f"proc_{variant}_{split}.parquet"
