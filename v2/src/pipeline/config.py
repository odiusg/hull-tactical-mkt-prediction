from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]  # v2/src/pipeline/ → repo root
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = Path(__file__).parents[2] / "results"  # v2/results/

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["date_id"]
FINANCE_COLS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
NON_FEATURE_COLS = META_COLS + FINANCE_COLS

TARGET_COL = "market_forward_excess_returns"

PREFIX_GROUPS = {
    "D": [f"D{i}" for i in range(1, 10)],
    "E": [f"E{i}" for i in range(1, 21)],
    "I": [f"I{i}" for i in range(1, 10)],
    "M": [f"M{i}" for i in range(1, 19)],
    "P": [f"P{i}" for i in range(1, 14)],
    "S": [f"S{i}" for i in range(1, 13)],
    "V": [f"V{i}" for i in range(1, 14)],
}

TEST_RATIO = 0.10
HIGH_NA_THRESHOLD = 0.40

INITIAL_TRAIN_DAYS = 1512  # ~6 years of trading days
N_FOLDS_TUNE = 5
N_FOLDS_EVAL = 10

M_ROLL_WINDOW = 5
E_DIFF_WINDOWS = [21, 63]
I_DIFF_WINDOW = 1
I_LAG_WINDOW = 5
P_ROLL_WINDOW = 21
V_ROLL_WINDOW = 21
RATE_REGIME_THRESHOLD = 0.0001  # ~2.5% annualized

MAX_LOOKBACK = max(E_DIFF_WINDOWS)  # 63

IC_T_STAT_MIN = 0.5
CORR_THRESHOLD = 0.85
VAR_QUANTILE = 0.25

K_DEFAULT = 1.0
K_ROLL_WINDOW = 63
K_MIN_PERIODS = 20
