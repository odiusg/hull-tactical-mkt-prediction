import numpy as np
import pandas as pd

from .config import VOL_TARGETS, VOL_WINDOWS, VOL_SMOOTH_W
from .paths import RESULTS_DIR


# Position mapping
def linear_position_mapping(p, base=1.0, scale=1.0):
    edge = p - 0.5
    pos_raw = base + scale * edge
    return np.clip(pos_raw, 0, 2)


def sigmoid_position_mapping(p, scale=5.0):
    z = np.clip(scale * (p - 0.5), -10, 10)
    return 2.0 / (1 + np.exp(-z))


def tanh_position_mapping(p, scale=5.0):
    z = np.clip(scale * (p - 0.5), -10, 10)
    return 1 + np.tanh(z)


# Smoothing / vol targeting
def smooth_series(arr, window=5):
    return pd.Series(arr).rolling(window=window,
                                  min_periods=1).mean().to_numpy()


def apply_vol_targeting(positions, fwd_ret, vol_window, target_vol, max_scale=3.0):
    ret_series = pd.Series(fwd_ret)
    rolling_vol = ret_series.rolling(
        window=vol_window,
        min_periods=max(5, vol_window // 3)
    ).std()

    target_daily = target_vol / np.sqrt(252)
    scale = (target_daily / (rolling_vol + 1e-8)).shift(1)
    scale = scale.fillna(1).clip(0, max_scale)

    pos_scaled = positions * scale.to_numpy()
    return np.clip(pos_scaled, 0, 2)


# Kaggle-style adjusted Sharpe
def kaggle_adjusted_sharpe(solution, submission):
    sol = solution.copy()
    sol["position"] = submission["prediction"].values

    sol["strategy_returns"] = (
        sol["risk_free_rate"] * (1 - sol["position"]) +
        sol["position"] * sol["forward_returns"]
    )

    strategy_excess = sol["strategy_returns"] - sol["risk_free_rate"]
    strategy_cum = (1 + strategy_excess).prod()
    strategy_mean_exc = strategy_cum**(1/len(sol)) - 1
    strategy_std = sol["strategy_returns"].std()

    trading_days = 252
    raw_sharpe = strategy_mean_exc / strategy_std * np.sqrt(trading_days)

    market_exc = sol["forward_returns"] - sol["risk_free_rate"]
    market_cum = (1 + market_exc).prod()
    market_mean_exc = market_cum**(1/len(sol)) - 1
    market_std = sol["forward_returns"].std()

    strat_vol_pct = float(strategy_std * np.sqrt(trading_days) * 100)
    market_vol_pct = float(market_std * np.sqrt(trading_days) * 100)

    vol_penalty = 1 + max(0, strat_vol_pct / market_vol_pct - 1.2)
    return_gap = max(0, (market_mean_exc - strategy_mean_exc) * 100 * trading_days)
    return_penalty = 1 + (return_gap**2) / 100

    adj = raw_sharpe / (vol_penalty * return_penalty)

    return float(adj), {
        "raw_sharpe": float(raw_sharpe),
        "strategy_mean_excess_ret": float(strategy_mean_exc),
        "market_mean_excess_ret": float(market_mean_exc),
        "strategy_volatility_pct": strat_vol_pct,
        "market_volatility_pct": market_vol_pct,
        "vol_penalty": float(vol_penalty),
        "return_penalty": float(return_penalty),
        "adjusted_sharpe": float(adj),
    }


# Single backtest run
def backtest(name, pos_fn, p_up, fwd, rf, smooth_w, target_vol, vol_win):
    pos = pos_fn(p_up)

    if smooth_w > 1:
        pos = smooth_series(pos, smooth_w)

    if target_vol is not None:
        pos = apply_vol_targeting(pos, fwd, vol_win, target_vol)

    strategy_returns = rf * (1 - pos) + pos * fwd
    equity = np.cumprod(1 + strategy_returns)

    total_return = equity[-1] - 1
    daily_mean = strategy_returns.mean()
    daily_vol = strategy_returns.std()

    ann_ret = (1 + daily_mean)**252 - 1
    ann_vol = daily_vol * np.sqrt(252)
    max_dd = (equity / np.maximum.accumulate(equity) - 1).min()

    sol = pd.DataFrame({
        "row_id": np.arange(len(fwd)),
        "forward_returns": fwd,
        "risk_free_rate": rf,
    })
    sub = pd.DataFrame({
        "row_id": sol["row_id"],
        "prediction": pos,
    })

    adj, details = kaggle_adjusted_sharpe(sol, sub)

    return {
        "name": name,
        "positions": pos,
        "basic_stats": {
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "max_drawdown": max_dd,
            "total_return": total_return,
        },
        "kaggle_details": details,
    }


# High-level grid search backtest
def run_backtest(proba_test,
                 forward_returns_test,
                 risk_free_test,
                 vol_targets=None,
                 vol_windows=None,
                 smooth_w=None):
    if vol_targets is None:
        vol_targets = VOL_TARGETS
    if vol_windows is None:
        vol_windows = VOL_WINDOWS
    if smooth_w is None:
        smooth_w = VOL_SMOOTH_W

    mappings = {
        "Linear": lambda p: linear_position_mapping(p, 1, 1),
        "Sigmoid": lambda p: sigmoid_position_mapping(p, 5.0),
        "Tanh": lambda p: tanh_position_mapping(p, 5.0),
    }

    best_overall = None
    best_adj = -1
    all_results = []

    for name, fn in mappings.items():
        for tv in vol_targets:
            for vw in vol_windows:
                res = backtest(
                    name=f"{name} | target={tv:.2%}, win={vw}",
                    pos_fn=fn,
                    p_up=proba_test,
                    fwd=forward_returns_test,
                    rf=risk_free_test,
                    smooth_w=smooth_w,
                    target_vol=tv,
                    vol_win=vw,
                )
                all_results.append(res)

                adj = res["kaggle_details"]["adjusted_sharpe"]
                if adj > best_adj:
                    best_adj = adj
                    best_overall = res

    baseline_pos = np.ones_like(forward_returns_test)
    baseline_returns = risk_free_test * (1 - baseline_pos) + baseline_pos * forward_returns_test

    sol_base = pd.DataFrame({
        "row_id": np.arange(len(forward_returns_test)),
        "forward_returns": forward_returns_test,
        "risk_free_rate": risk_free_test,
    })
    sub_base = pd.DataFrame({
        "row_id": sol_base["row_id"],
        "prediction": baseline_pos,
    })

    baseline_adj, baseline_details = kaggle_adjusted_sharpe(sol_base, sub_base)

    print("\n===== FINAL RESULTS =====")
    print("\n[1] Market Baseline (position=1)")
    print("Adjusted Sharpe:", baseline_adj)
    print("Details:", baseline_details)

    print("\n[2] Best Strategy from Grid Search")
    print("Name:", best_overall["name"])
    print("Adjusted Sharpe:", best_overall["kaggle_details"]["adjusted_sharpe"])
    print("Details:", best_overall["kaggle_details"])
    print("Basic Stats:", best_overall["basic_stats"])

    return {
        "baseline": {
            "positions": baseline_pos,
            "adjusted_sharpe": baseline_adj,
            "details": baseline_details,
            "returns": baseline_returns,
        },
        "best": best_overall,
        "all_results": all_results,
    }
