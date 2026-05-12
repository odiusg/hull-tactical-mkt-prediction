# Hull Tactical – S&P 500 战术配置策略

[English](README.md)

基于 [Kaggle Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) 比赛构建的端到端机器学习pipeline，覆盖 EDA、特征工程、模型训练与策略回测。

比赛任务是预测 **S&P 500 的超额收益（excess return）**，并在 **120% 波动率预算（volatility budget）** 约束下进行仓位配置，官方评估指标为波动率调整后的 Sharpe ratio。

---

## 1. Pipeline 总览

```
原始 CSV  ─▶  数据预处理  ─▶  特征工程  ─▶  特征筛选
                                                  │
                                                  ▼
                                          LightGBM 模型训练
                                          (HalvingRandomSearchCV
                                          + TimeSeriesSplit)
                                                  │
                                                  ▼
                                        策略回测（仓位映射
                                          × 波动率目标 grid search）
```

每一步对应 `src/hull_tactical/` 下一个独立模块，由 `scripts/` 下同名 driver 脚本调用。

| 阶段 | 模块 | 入口脚本 |
|---|---|---|
| 数据加载与时间切分 | `data_loading.py` | – |
| EDA 工具函数 | `eda.py` | `notebooks/eda_overview.ipynb` |
| 数据清洗 + 缺失标记 | `preprocessing.py` | `scripts/run_preprocessing.py` |
| 特征筛选（方差 + 相关性 + LightGBM gain） | `feature_selection.py` | `scripts/run_feature_selection.py` |
| 特征工程（动量、排序、均值回归、情绪类） | `feature_engineering.py` | `scripts/run_feature_engineering.py` |
| LightGBM 方向性分类器 | `modelling.py` | `scripts/run_training.py` |
| 策略回测 | `backtest.py` | `scripts/run_backtest.py` |
| 端到端 | – | `scripts/run_full_pipeline.py` |

---

## 2. 项目结构

```
hull_tactical_mkt_prediction/
├── README.md
├── LICENSE
├── pyproject.toml          # `hull_tactical` 的 editable install 配置
├── requirements.txt
├── .gitignore
├── configs/                # （gitignored）运行时生成的 JSON 配置
├── artifacts/              # （gitignored）训练好的模型 (*.joblib)
├── results/                # （gitignored）图表、xlsx、json 报告
├── data/
│   ├── raw/                # 把 train.csv / test.csv 放这里（gitignored）
│   ├── interim/            # cleaned_train.xlsx 等中间结果
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

## 3. 快速开始

```bash
# 1. clone 项目
git clone https://github.com/<your-handle>/hull_tactical_mkt_prediction.git
cd hull_tactical_mkt_prediction

# 2. 创建虚拟环境（Python >= 3.10）
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. 安装依赖 + 当前包（editable mode）
pip install -r requirements.txt
pip install -e .

# 4. 把比赛 CSV 放到 data/raw/
#    （从 Kaggle 下载，文件不入版本控制）
data/raw/train.csv
data/raw/test.csv

# 5. 运行完整 pipeline
python scripts/run_full_pipeline.py
```

也可以分阶段单独运行：

```bash
python scripts/run_preprocessing.py
python scripts/run_feature_engineering.py
python scripts/run_feature_selection.py
python scripts/run_training.py
python scripts/run_backtest.py
```

模型产物输出到 `artifacts/`，分析报告输出到 `results/`。

---

## 4. 建模细节

- **预测目标（target）**：`forward_returns`，当前实现为二分类，`direction = (target > 0)`。
- **模型（estimator）**：`lightgbm.LGBMClassifier`，使用 `HalvingRandomSearchCV` 在 `TimeSeriesSplit(n_splits=5)` 上调参（不 shuffle，遵守时间顺序）。
- **特征筛选**：variance filter → correlation filter → 在独立 LightGBM regressor 上做 cumulative gain 筛选（保留累计 gain ≤ 90% 的特征）。
- **策略层**：将预测概率 `p̂` 通过 Linear / Sigmoid / Tanh 映射为仓位权重，做平滑后用波动率目标（10%–18% 年化）进行缩放，最后用 Kaggle 官方的 adjusted Sharpe 评估。

## 5. License

本项目采用 [MIT License](LICENSE)。

---

## 附录：关键英文术语对照

| 中文 | 英文 |
|---|---|
| 超额收益 | Excess return |
| 波动率预算 / 波动率目标 | Volatility budget / Volatility targeting |
| 已实现波动率 | Realised volatility |
| 信息系数 | Information Coefficient (IC) |
| 排序 IC | Rank IC / Spearman IC |
| IC 信息比率 | ICIR |
| 仓位映射 | Position mapping / sizing |
| 换手率 | Turnover |
| 基点 | Basis point (bps) |
| 滑点 | Slippage |
| 前视偏差 | Lookahead bias / leakage |
| 净化 K 折 | Purged K-Fold |
| 隔离期 | Embargo |
| 回测过拟合 | Backtest overfitting |
| 最大回撤 | Max Drawdown (MDD) |
| 买方 / 卖方 | Buy-side / Sell-side |
