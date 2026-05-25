from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
