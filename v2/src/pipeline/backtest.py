from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="An input array is constant")

MIN_POSITION = 0.0
MAX_POSITION = 2.0
TRADING_DAYS_PER_YEAR = 252


class ParticipantVisibleError(Exception):
    pass


def competition_score(
    forward_returns: pd.Series,
    risk_free_rate: pd.Series,
    positions: np.ndarray | pd.Series,
) -> float:
    pos = np.asarray(positions, dtype=float)

    if np.any(pos > MAX_POSITION):
        raise ParticipantVisibleError(f"Position {pos.max():.4f} exceeds maximum {MAX_POSITION}")
    if np.any(pos < MIN_POSITION):
        raise ParticipantVisibleError(f"Position {pos.min():.4f} below minimum {MIN_POSITION}")

    rf = risk_free_rate.values
    ret = forward_returns.values

    strat_ret = rf * (1 - pos) + pos * ret
    excess = strat_ret - rf
    # log-returns avoid complex numbers from negative cumulative returns
    mean_excess = float(np.exp(np.log1p(excess).sum() * (1 / len(excess))) - 1)
    strat_std = float(np.std(strat_ret, ddof=1))
    if strat_std == 0:
        raise ParticipantVisibleError("Strategy std is zero — division by zero")
    sharpe = mean_excess / strat_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    strat_vol = strat_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    mkt_excess = ret - rf
    mkt_mean_excess = float(np.exp(np.log1p(mkt_excess).sum() * (1 / len(mkt_excess))) - 1)
    mkt_std = float(np.std(ret, ddof=1))
    mkt_vol = mkt_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    if mkt_vol == 0:
        raise ParticipantVisibleError("Market vol is zero — division by zero")

    excess_vol = max(0.0, strat_vol / mkt_vol - 1.2)
    vol_penalty = 1.0 + excess_vol

    return_gap = max(0.0, (mkt_mean_excess - mean_excess) * 100 * TRADING_DAYS_PER_YEAR)
    return_penalty = 1.0 + (return_gap**2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return float(min(adjusted_sharpe, 1_000_000))


def compute_metrics(
    forward_returns: pd.Series,
    risk_free_rate: pd.Series,
    positions: np.ndarray | pd.Series,
    pred: np.ndarray | pd.Series | None = None,
    target: np.ndarray | pd.Series | None = None,
) -> dict:
    pos = np.asarray(positions, dtype=float)
    rf = risk_free_rate.values
    ret = forward_returns.values

    strat_ret = rf * (1 - pos) + pos * ret
    excess = strat_ret - rf

    strat_std_ann = float(np.std(strat_ret, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    ann_excess = float(np.exp(np.log1p(excess).sum() * (TRADING_DAYS_PER_YEAR / len(excess))) - 1)

    sharpe_raw = ann_excess / strat_std_ann if strat_std_ann > 0 else np.nan

    downside = excess[excess < 0]
    sortino_denom = float(np.std(downside, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 1 else np.nan
    sortino = ann_excess / sortino_denom if sortino_denom and sortino_denom > 0 else np.nan

    cum = np.cumprod(1 + strat_ret)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd = float(drawdowns.min())

    calmar = ann_excess / abs(max_dd) if max_dd < 0 else np.nan
    win_rate = float(np.mean(excess > 0))

    metrics = {
        "adjusted_sharpe": competition_score(pd.Series(ret), pd.Series(rf), pos),
        "sharpe_raw": sharpe_raw,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "ann_vol_pct": strat_std_ann * 100,
    }

    if pred is not None and target is not None:
        from scipy.stats import spearmanr, ttest_1samp
        p = np.asarray(pred, dtype=float)
        t_arr = np.asarray(target, dtype=float)
        valid = ~(np.isnan(p) | np.isnan(t_arr))
        if valid.sum() > 2:
            ic, _ = spearmanr(p[valid], t_arr[valid])
            ic_std = float(np.std(
                [spearmanr(p[valid][i:i+63], t_arr[valid][i:i+63])[0]
                 for i in range(0, valid.sum() - 63, 63)],
                ddof=1,
            )) if valid.sum() > 63 else np.nan
            icir = ic / ic_std if ic_std and ic_std > 0 else np.nan
            t_stat, p_val = ttest_1samp(p[valid] - t_arr[valid], 0)
            metrics.update({"ic": ic, "icir": icir, "t_stat": t_stat, "p_value": p_val})

    return metrics
