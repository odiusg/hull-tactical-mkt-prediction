# V2 Architecture


## 该版为第二版，有关比赛内容与第一版请参考

- [English README](../README.md)
- [中文 README](../README_zh.md)

---
## 1. 重要提示
以下是整体架构的关键约定：

\- 预测目标：market_forward_excess_returns（不是 forward_returns），回归问题

\- 最终评估指标：adjusted Sharpe ratio（见score函数）

\- 数据切分：前90%训练+验证，后10%测试

\- 禁止泄露：所有 fit 操作只在当前 fold 训练集上执行，禁止泄露

\- Walk-Forward：Expanding窗口，5 fold调参，10 fold评估

## 2. 整体架构

### 1. EDA（已完成 v2/notebooks/eda.ipynb）& Preprocessing

#### 1. 考察缺失率

缺失率 ≥ 40% 的特征直接删除

缺失率 < 40% 的特征前向填充

#### 2. 极端值

由于金融数据存在异常值较为常见，EDA 环节没有发现数据脏点，不做处理

#### 3. 分布

`market_forward_excess_returns` 分布接近正态，均值与中位数均趋近于零（0.0001 vs 0.0003），轻微左偏（skew = -0.180），反映股市"涨慢跌快"的典型特征。峰度为 2.244，略低于正态分布的 3，这与目标变量经过 MAD winsorizing 处理后极值被压缩有关。整体而言分布较为对称，适合直接作为回归目标。

`risk_free_rate` 分布呈显著双峰形态，左峰集中于 0–0.00005 附近，对应低利率时期；右峰集中于 0.00018–0.00022，对应正常及偏紧的货币政策周期。整体轻微右偏（skew = 0.210），由右峰分布更分散且存在右侧长尾所致。据此，在后续特征工程构造一列高/低利率 regime 哑变量，阈值设定为 0.0001（对应年化约 2.5%）。

### 2. 特征工程

> **重要：** 从此步骤起，将数据集切分为前 90%（训练集）和后 10%（测试集），以下所有操作均在前 90% 内进行。所有需要 fit 的操作——包括特征选择统计量（IC、t 检验、方差）的计算，以及 z-score Scaler 的拟合——均在**当前 fold 的训练集**上执行，再 transform 对应的验证集，以防止数据泄露。

#### 1. 特征选择

原始特征有 94 个，数据 9048 行，做完整特征工程会产生几百乃至上千列，带来计算慢和过拟合风险。因此采用分层处理：

**第一层**（在每个 fold 的训练集内执行）：先做特征选择，再做特征工程。按顺序执行：

1. 删除哑变量、缺失值 > 40% 的列
2. 相关性过滤：IC 检验 + t 检验 + 簇内相关性剔除（同簇内保留方差较大的代表特征）
3. 方差过滤：保留方差较大的特征

目标：保留 20–30 个核心特征，各步骤阈值由实际数据分布决定。

**第二层**：对 20–30 个核心特征进行特征构造 + 特征变换（见下方），目标特征数控制在 40–90 个。

**第三层**：根据模型特性选择特征子集：

- Ridge / Lasso：正则化即特征选择，全量输入。
- XGBoost / LightGBM / Random Forest：对特征数量不敏感，全量输入。
- LSTM：过拟合风险更高，目标控制在 20 个特征以内。压缩方式：取 Lasso 保留的特征 ∪ XGBoost 特征重要性排名靠前的特征。

#### 2. 特征构造

源数据特征含义较黑盒，按时序变换从简构造，每簇给出 1–2 种处理方式：

| 特征簇 | 变换方式 |
|--------|---------|
| M* 市场动态 | 5 日滚动均值 |
| E* 宏观经济 | 21 / 63 期差分 |
| I* 利率 | 差分 1 期后滞后 5 日 |
| P* 价格/估值 | 偏离 21 日滚动均值 |
| V* 波动率 | 21 日滚动均值 |
| S* 情绪 | 滞后 1 日 |
| MOM* 动量 | 滞后 5 / 21 日 |

新增哑变量：高/低利率 regime，阈值 `risk_free_rate = 0.0001`（对应年化约 2.5%）。

注意：构造完成后，删除因窗口期不足产生的开头 NaN 行。

#### 3. 特征变换

**仅对线性模型（Ridge / Lasso / ElasticNet）和 LSTM** 执行 z-score 标准化，以加速梯度收敛。树模型（Random Forest / XGBoost / LightGBM）对特征尺度不敏感，跳过此步骤。

此步骤在每个 fold 内部执行：在当前 fold 训练集上 fit `StandardScaler`，再 transform 训练集和验证集。

### 3. 建模

#### 1. 问题定义

预测目标为 `market_forward_excess_returns`，问题为回归问题。

#### 2. 模型选择

Linear Regression（baseline）、Ridge、Lasso、ElasticNet、Random Forest、XGBoost、LightGBM、LSTM（特征见第三层）。

LSTM 仅为实验性候选，不必花过多时间调参。

#### 3. 训练

前 90% 采用 walk-forward 方式调超参，Optuna 贝叶斯优化，MLFlow 本地实验管理。

##### Walk-Forward 配置

| 参数 | 值 |
|------|-----|
| 窗口类型 | Expanding |
| 初始训练集大小 | 1512 交易日（约 6 年） |
| 验证集大小 | 由总数据量与fold数均匀决定 |
| Fold 数 | 调参：5 fold；最终评估：10 fold |
| Gap | 0（预测目标为次日收益，相邻 label 无时间重叠，无需间隔） |

调参完成后，以各模型的最优超参数在全部 90% 数据上重新训练最终模型，用于测试集评估

#### 4. 模型评估

MAE / MSE / RMSE / R²

#### 5. 诊断

- **偏差 & 方差**：学习曲线，对比训练集与验证集误差，评估过拟合程度。
- **残差诊断**：检验残差是否随机，识别系统性预测偏差。

#### 6. 仓位映射

将模型输出的连续预测值线性映射为合规仓位：

```
position = clip(1 + k × pred / rolling_std(pred, window=63), 0, 2)
```

- 中性仓位 = 1；预测为正则加仓，预测为负则减仓
- `rolling_std` 窗口为 63 交易日，用于对预测值做尺度归一化，使 k 的含义与预测值量级无关
- `k` 为激进程度系数，由 Optuna 搜索，搜索范围 [0.5, 3.0]

### 4. 回测（测试集测试）

```python
import numpy as np
import pandas as pd
import pandas.api.types

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Returns:
        float: The calculated adjusted Sharpe ratio.
    """

    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution
    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)
```

根据项目方提供的仓位规则和 adjusted Sharpe ratio 评估指标设计回测框架。

此外提供辅助指标：年化波动率、最大回撤、胜率、Calmar Ratio、Sortino Ratio、IC、ICIR、t 检验。
