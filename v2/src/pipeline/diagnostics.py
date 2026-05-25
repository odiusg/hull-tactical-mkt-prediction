from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def plot_learning_curves(
    results_df: pd.DataFrame,
    model_name: str,
    save_dir: Path | None = None,
) -> Path:
    df = results_df.copy()
    has_train = "train_mae" in df.columns and df["train_mae"].notna().any()

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(f"Walk-forward Learning Curves — {model_name.upper()}", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    x = df["train_size"].values
    xlabel = "Training set size (rows)"

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, df["mae"].values, "o-", color="#e07b54", label="Validation MAE")
    if has_train:
        ax.plot(x, df["train_mae"].values, "s--", color="#5b9bd5", label="Training MAE")
    ax.set_title("(A) MAE — bias/variance gap")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, df["r2"].values, "o-", color="#e07b54", label="Validation R²")
    if has_train:
        ax.plot(x, df["train_r2"].values, "s--", color="#5b9bd5", label="Training R²")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("(B) R² — predictive power")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("R²")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["adjusted_sharpe"].values]
    ax.bar(range(len(df)), df["adjusted_sharpe"].values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    mean_as = df["adjusted_sharpe"].mean()
    ax.axhline(mean_as, color="navy", linewidth=1.2, linestyle="--", label=f"Mean={mean_as:.3f}")
    ax.set_title("(C) Adjusted Sharpe per fold")
    ax.set_xlabel("Fold index")
    ax.set_ylabel("Adjusted Sharpe")
    ax.set_xticks(range(len(df)))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, df["n_features"].values, "o-", color="#9467bd")
    ax.set_title("(D) Engineered features selected per fold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# features")
    ax.grid(True, alpha=0.3)

    if has_train:
        avg_gap = (df["mae"] - df["train_mae"]).mean()
        if avg_gap > df["train_mae"].mean() * 0.5:
            diagnosis = "⚠ High variance (val MAE >> train MAE) — consider more regularisation"
        elif df["mae"].mean() > df["mae"].quantile(0.8):
            diagnosis = "⚠ High bias (both errors high) — consider richer features"
        else:
            diagnosis = "✓ Bias/variance balance appears reasonable"
        fig.text(0.5, 0.01, diagnosis, ha="center", fontsize=9, color="dimgray", style="italic")

    save_dir = Path(save_dir) if save_dir else Path(".")
    out = save_dir / f"learning_curves_{model_name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_residuals(
    pred: np.ndarray,
    actual: np.ndarray,
    model_name: str,
    save_dir: Path | None = None,
) -> Path:
    resid = actual - pred
    t = np.arange(len(resid))

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(f"Residual Diagnostics — {model_name.upper()}", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(t, resid, s=4, alpha=0.4, color="#5b9bd5")
    roll = pd.Series(resid).rolling(63, min_periods=10).mean()
    ax.plot(t, roll.values, color="#e07b54", linewidth=1.2, label="63-day rolling mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("(A) Residuals over time")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(resid, bins=50, color="#5b9bd5", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("(B) Residual distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(pred, resid, s=4, alpha=0.4, color="#9467bd")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("(C) Residuals vs. fitted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    cum_err = np.cumsum(resid)
    ax.plot(t, cum_err, color="#e07b54")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("(D) Cumulative error")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cumulative residual")
    ax.grid(True, alpha=0.3)

    save_dir = Path(save_dir) if save_dir else Path(".")
    out = save_dir / f"residuals_{model_name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_test_results(
    fwd: pd.Series,
    rf: pd.Series,
    positions: np.ndarray,
    pred: np.ndarray,
    actual: np.ndarray,
    model_name: str,
    save_dir: Path | None = None,
) -> list[Path]:
    """Individual backtest charts for the held-out test set. Returns list of saved Paths."""
    save_dir = Path(save_dir) if save_dir else Path(".")

    strat_ret = pd.Series(rf.values * (1 - positions) + positions * fwd.values, index=fwd.index)
    bh_ret    = fwd.copy()
    excess    = strat_ret - rf.values

    cum_strat   = (1 + strat_ret).cumprod()
    cum_bh      = (1 + bh_ret).cumprod()
    running_max = cum_strat.cummax()
    drawdown    = (cum_strat - running_max) / running_max
    roll_sharpe = (
        pd.Series(excess).rolling(63, min_periods=21).mean()
        / strat_ret.rolling(63, min_periods=21).std()
        * np.sqrt(252)
    )

    has_dates = isinstance(fwd.index, pd.DatetimeIndex)
    idx = fwd.index if has_dates else np.arange(len(fwd))
    paths: list[Path] = []

    def _save(fig, tag):
        out = save_dir / f"test_{tag}_{model_name}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    def _xdate(fig):
        if has_dates:
            fig.autofmt_xdate(rotation=30, ha="right")

    # Cumulative returns
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, cum_strat.values, color="#2ca02c", linewidth=1.4, label="Strategy")
    ax.plot(idx, cum_bh.values, color="#aec7e8", linewidth=1.0, linestyle="--", label="Buy & Hold")
    ax.set_title(f"Cumulative Returns — {model_name.upper()}")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _xdate(fig)
    _save(fig, "cumret")

    # Drawdown
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(idx, drawdown.values, 0, color="#d62728", alpha=0.6)
    ax.set_title(f"Drawdown — {model_name.upper()}")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3)
    _xdate(fig)
    _save(fig, "drawdown")

    # Rolling Sharpe
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, roll_sharpe.values, color="#5b9bd5", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(roll_sharpe.mean(), color="navy", linewidth=1.0, linestyle="--",
               label=f"Mean={roll_sharpe.mean():.2f}")
    ax.set_title(f"Rolling Sharpe (63-day) — {model_name.upper()}")
    ax.set_ylabel("Annualised Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _xdate(fig)
    _save(fig, "rolling_sharpe")

    # Monthly returns heatmap
    fig, ax = plt.subplots(figsize=(10, 4))
    if has_dates:
        monthly = strat_ret.resample("ME").apply(lambda x: float((1 + x).prod() - 1))
        pivot = pd.DataFrame({
            "year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values,
        }).pivot(index="year", columns="month", values="ret")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [month_labels[m - 1] for m in pivot.columns]
        vmax = np.nanpercentile(np.abs(pivot.values), 95)
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7,
                            color="black" if abs(v) < vmax * 0.6 else "white")
        plt.colorbar(im, ax=ax, format="{x:.0%}", shrink=0.8)
    else:
        chunk = max(1, len(strat_ret) // 20)
        periodic = [(1 + strat_ret.values[i:i+chunk]).prod() - 1
                    for i in range(0, len(strat_ret), chunk)]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in periodic]
        ax.bar(range(len(periodic)), periodic, color=colors, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_title(f"Monthly Returns — {model_name.upper()}")
    ax.grid(True, alpha=0.3)
    _save(fig, "monthly")

    # Position distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(positions, bins=40, color="#9467bd", edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--", label="Neutral (1.0)")
    ax.axvline(float(np.mean(positions)), color="#e07b54", linewidth=1.2,
               label=f"Mean={np.mean(positions):.2f}")
    ax.set_title(f"Position Distribution — {model_name.upper()}")
    ax.set_xlabel("Position size")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "positions")

    # IC scatter (prediction vs actual)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(pred, actual, s=5, alpha=0.3, color="#5b9bd5")
    ic = float(pd.Series(pred).corr(pd.Series(actual)))
    lim = max(np.percentile(np.abs(pred), 99), np.percentile(np.abs(actual), 99))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    m, b = np.polyfit(pred, actual, 1)
    xs = np.array([-lim, lim])
    ax.plot(xs, m * xs + b, color="#e07b54", linewidth=1.2, label=f"IC={ic:.3f}")
    ax.set_title(f"Prediction vs Actual (IC) — {model_name.upper()}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "ic_scatter")

    # Rolling volatility
    roll_vol_strat = strat_ret.rolling(63, min_periods=21).std() * np.sqrt(252) * 100
    roll_vol_bh    = bh_ret.rolling(63, min_periods=21).std() * np.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, roll_vol_strat.values, color="#2ca02c", linewidth=1.2, label="Strategy")
    ax.plot(idx, roll_vol_bh.values, color="#aec7e8", linewidth=1.0, linestyle="--", label="Buy & Hold")
    ax.set_title(f"Rolling Volatility (63-day, ann.) — {model_name.upper()}")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _xdate(fig)
    _save(fig, "rolling_vol")

    return paths
