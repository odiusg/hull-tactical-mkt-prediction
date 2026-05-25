import pandas as pd
from .paths import RAW_DIR, DATA_DIR 


# Data loading
def load_train_test(train_name="train.csv", test_name="test.csv", sort_by="date_id"):
    # Prefer data/raw/, fall back to data/ for backwards compatibility
    train_path = RAW_DIR / train_name if (RAW_DIR / train_name).exists() else DATA_DIR / train_name
    test_path  = RAW_DIR / test_name  if (RAW_DIR / test_name).exists()  else DATA_DIR / test_name

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if sort_by is not None and sort_by in train.columns:
        train = train.sort_values(sort_by).reset_index(drop=True)

    return train, test

# Time-based split
def time_based_split(df, train_ratio=0.7, val_ratio=0.2, sort_by="date_id"):
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)

    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_set = df.iloc[:train_end].reset_index(drop=True)
    val_set = df.iloc[train_end:val_end].reset_index(drop=True)
    test_set = df.iloc[val_end:].reset_index(drop=True)

    return train_set, val_set, test_set


