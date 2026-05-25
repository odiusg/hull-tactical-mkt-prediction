from scipy.stats import randint, loguniform, uniform

# Column definitions

META_COLS = ["date_id", "is_scored"]

FINANCE_COLS = [
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
]

NON_FEATURE_COLS = META_COLS + FINANCE_COLS

PREFIX_MAP = {
    "M": "Market Dynamics",
    "E": "Economic",
    "I": "Interest Rate",
    "P": "Price/Valuation",
    "V": "Volatility",
    "S": "Sentiment",
    "MOM": "Momentum",
    "D": "Dummy",
}

TARGET_COL = "forward_returns"


# Feature engineering configuration 

MACRO_COLS = ["E18", "E20"]
MACRO_WINDOWS = [60, 120]

MD_COLS = ["M3", "M4", "M11", "M12", "M18"]
MOM_WINDOWS = [5, 10]
RANK_WINDOW = 20

PRICE_COLS = ["P5", "P6", "P9", "P12"]
MR_WINDOWS = [30, 60]

SENT_COLS = ["S2", "S7"]
SENT_SMOOTH_WINDOW = 20
SENT_SPIKE_WINDOW = 5


# Model / CV configuration

RANDOM_STATE = 42

N_SPLITS = 5

PARAM_DISTRIBUTIONS = { # Hyperparameter search space for LGBMClassifier
    "num_leaves": randint(20, 200),
    "learning_rate": loguniform(0.005, 0.2),
    "max_depth": randint(3, 13),
    "min_child_samples": randint(5, 60),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_alpha": loguniform(1e-4, 10),
    "reg_lambda": loguniform(1e-4, 10),
}


# Backtest / strategy configuration

VOL_TARGETS = [0.10, 0.12, 0.14, 0.16, 0.18]

VOL_WINDOWS = [30, 60, 90]

VOL_SMOOTH_W = 5

