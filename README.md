# Hull Tactical – Market Prediction

A full reproducible pipeline for EDA, feature engineering, model training and
back-testing built for the
[Kaggle Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
competition.

The goal of the competition is to predict the **excess return of the S&P 500**
while honouring a **120 % volatility budget**, with the official evaluation
metric being a volatility-adjusted Sharpe ratio.

---

## 1. Pipeline overview

```
raw csv  ─▶  preprocessing  ─▶  feature_engineering  ─▶  feature_selection
                                                              │
                                                              ▼
                                                       LightGBM training
                                                       (HalvingRandomSearchCV
                                                        + TimeSeriesSplit)
                                                              │
                                                              ▼
                                        strategy back-test (position mapping
                                          × volatility targeting grid search)
```

Each step is a self-contained module under `src/hull_tactical/` and is wrapped
by a corresponding driver in `scripts/`.

| Stage | Module | Driver |
|------|---|---|
| Data loading & time split | `data_loading.py` | – |
| EDA helpers | `eda.py` | `notebooks/eda_overview.ipynb` |
| Cleaning + missing flags | `preprocessing.py` | `scripts/run_preprocessing.py` |
| Feature selection (variance + correlation + LightGBM gain) | `feature_selection.py` | `scripts/run_feature_selection.py` |
| Engineered features (momentum, rank, mean-reversion, sentiment) | `feature_engineering.py` | `scripts/run_feature_engineering.py` |
| LightGBM directional classifier | `modelling.py` | `scripts/run_training.py` |
| Strategy back-test | `backtest.py` | `scripts/run_backtest.py` |
| End-to-end | – | `scripts/run_full_pipeline.py` |

---

## 2. Repository layout

```
hull_tactical_mkt_prediction/
├── README.md
├── LICENSE
├── pyproject.toml          # editable install for `hull_tactical`
├── requirements.txt
├── .gitignore
├── configs/                # (gitignored) JSON configs written at run time
├── artifacts/              # (gitignored) trained models (*.joblib)
├── results/                # (gitignored) plots, xlsx, json reports
├── data/
│   ├── raw/                # put train.csv / test.csv here (gitignored)
│   ├── interim/            # cleaned_train.xlsx etc.
│   └── processed/
├── notebooks/
│   ├── eda_overview.ipynb
│   ├── eda_report.ipynb
│   ├── feature_selection.ipynb
│   └── model_backtest.ipynb
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_feature_engineering.py
│   ├── run_feature_selection.py
│   ├── run_training.py
│   ├── run_backtest.py
│   └── run_full_pipeline.py
└── src/
    └── hull_tactical/
        ├── __init__.py
        ├── paths.py
        ├── config.py
        ├── data_loading.py
        ├── utils.py
        ├── eda.py
        ├── preprocessing.py
        ├── feature_engineering.py
        ├── feature_selection.py
        ├── modelling.py
        └── backtest.py
```

---

## 3. Quick start

```bash
# 1. clone & enter
git clone https://github.com/<your-handle>/hull_tactical_mkt_prediction.git
cd hull_tactical_mkt_prediction

# 2. create a virtual env (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install dependencies + this package in editable mode
pip install -r requirements.txt
pip install -e .

# 4. drop the competition CSVs in data/raw/
#    (download from Kaggle; the files are NOT checked in)
data/raw/train.csv
data/raw/test.csv

# 5. run the full pipeline
python scripts/run_full_pipeline.py
```

You can also run each stage in isolation:

```bash
python scripts/run_preprocessing.py
python scripts/run_feature_selection.py
python scripts/run_feature_engineering.py
python scripts/run_training.py
python scripts/run_backtest.py
```

All artefacts land in `artifacts/`, all reports in `results/`.

---

## 4. Modelling details

- **Target.**  `forward_returns`, binarised as `direction = (target > 0)`.
- **Estimator.**  `lightgbm.LGBMClassifier`, tuned by
  `HalvingRandomSearchCV` on `TimeSeriesSplit(n_splits=5)`
  (no shuffling, no leakage).
- **Selection.**  Variance → correlation → cumulative-gain (≤ 90 %) on a
  separate LightGBM regressor.
- **Strategy.**  Probability `p̂` is mapped to portfolio weight via
  Linear / Sigmoid / Tanh, smoothed and rescaled to a target annual volatility
  (10 – 18 %), then evaluated with the Kaggle-style adjusted Sharpe metric.

---

## 5. License

Released under the [MIT License](LICENSE).
