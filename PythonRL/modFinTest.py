import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt




def backtest_multiple_paths(R_paths, r, tau=60, risk_aversion=0.1, ew_w=0.5, bins=40):
    N, T = R_paths.shape
    metrics = ["avg_yearly_return", "sharpe", "calmar", "max_drawdown"]
    strategies = ["CRP", "BH", "MVO", "EW"]

    results = {s: {m: np.zeros(N) for m in metrics} for s in strategies}

    for i in range(N):
        res_i = backtest_single_risky(R_paths[i], r, tau=tau,
                                      risk_aversion=risk_aversion, ew_w=ew_w)
        for s in strategies:
            for m in metrics:
                val = res_i[s][m]
                results[s][m][i] = np.nan if not np.isfinite(val) else val

    fig, axes = plt.subplots(len(metrics), len(strategies),
                             figsize=(14, 10), sharex=False, sharey=False)
                             
    

    for i, m in enumerate(metrics):
        for j, s in enumerate(strategies):
            ax = axes[i, j]
            data = results[s][m]
            data = data[np.isfinite(data)]

            if len(data) == 0:
                ax.text(0.5, 0.5, "no data", ha='center', va='center')
                continue

            ax.hist(data, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)

            # Robust range
            q1, q99 = np.nanquantile(data, [0.01, 0.99])
            if np.isfinite(q1) and np.isfinite(q99) and q1 != q99:
                ax.set_xlim(q1, q99)

            ax.set_title(f"{s} – {m}", fontsize=9)
            ax.tick_params(axis='x', labelbottom=True, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(alpha=0.3)

    # instead of tight_layout(), use manual spacing to avoid re-hiding ticks
    plt.subplots_adjust(wspace=0.35, hspace=0.45)
    fig.savefig("testPlot.png") 
    plt.show
    
    for ax in axes.flat:
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    for label in ax.get_xticklabels():
        label.set_visible(True)
    for label in ax.get_yticklabels():
        label.set_visible(True)

    # And now adjust spacing manually (NO tight_layout)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()

    return results

def backtest_single_risky(R, r, tau=60, risk_aversion=0.1, ew_w=0.5):
    """
    Backtest four strategies on a single risky asset + risk-free asset.
    """

    R = np.asarray(R).squeeze()
    if R.ndim != 1:
        raise ValueError("R must be 1D or shaped (1, T).")
    T = R.shape[0]
    if T <= tau:
        raise ValueError("T must be > tau.")
    n_periods = T - tau

    # convert annual risk-free to daily
    r_daily = (1.0 + r) ** (1.0 / 252.0) - 1.0

    # helper for metrics
    def compute_metrics(period_rets):
        wealth = np.cumprod(1.0 + period_rets)
        final_wealth = wealth[-1]
        cumulative_return = final_wealth - 1.0
        avg_yearly_return = final_wealth ** (252.0 / len(period_rets)) - 1.0
        ann_vol = np.std(period_rets, ddof=1) * np.sqrt(252.0) if len(period_rets) > 1 else 0.0
        sharpe = (avg_yearly_return - r) / ann_vol if ann_vol > 0 else np.nan
        running_max = np.maximum.accumulate(wealth)
        drawdowns = (running_max - wealth) / running_max
        max_dd = float(np.max(drawdowns))
        calmar = avg_yearly_return / max_dd if max_dd > 0 else np.nan
        return {
            "returns": period_rets,
            "wealth": wealth,
            "cumulative_return": cumulative_return,
            "avg_yearly_return": avg_yearly_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
        }

    # --- Strategy 1: CRP weight selection ---
    R_init = R[:tau]

    def avg_log_growth(w):
        vals = w * (1.0 + R_init) + (1.0 - w) * (1.0 + r_daily)
        if np.any(vals <= 0.0):
            return 1e6
        return -np.mean(np.log(vals))

    res = minimize_scalar(avg_log_growth, bounds=(0.0, 1.0), method="bounded")
    w_crp = float(np.clip(res.x, 0.0, 1.0))

    # CRP returns
    crp_period_rets = np.array([
        w_crp * (1.0 + R[t]) + (1.0 - w_crp) * (1.0 + r_daily) - 1.0
        for t in range(tau, T)
    ])

    # BH returns
    bh_period_rets = np.empty(n_periods)
    risky_units = w_crp
    rf_units = 1.0 - w_crp
    prev_wealth = 1.0
    for i, t in enumerate(range(tau, T)):
        risky_value = risky_units * (1.0 + R[t])
        rf_value = rf_units * (1.0 + r_daily)
        wealth = risky_value + rf_value
        bh_period_rets[i] = wealth / prev_wealth - 1.0
        prev_wealth = wealth

    # MVO returns
    mvo_period_rets = np.empty(n_periods)
    lam = float(risk_aversion)
    for i, t in enumerate(range(tau, T)):
        window = R[t - tau:t]
        mu = np.mean(window)
        var = np.var(window, ddof=1)
        if var <= 0.0:
            w_t = 1.0 if mu > r_daily else 0.0
        else:
            w_t = (mu - r_daily) / (2.0 * lam * var)
            w_t = float(np.clip(w_t, 0.0, 1.0))
        gross = w_t * (1.0 + R[t]) + (1.0 - w_t) * (1.0 + r_daily)
        mvo_period_rets[i] = gross - 1.0

    # EW returns
    ew_w = float(np.clip(ew_w, 0.0, 1.0))
    ew_period_rets = np.array([
        ew_w * (1.0 + R[t]) + (1.0 - ew_w) * (1.0 + r_daily) - 1.0
        for t in range(tau, T)
    ])

    results = {
        "CRP": compute_metrics(crp_period_rets),
        "BH": compute_metrics(bh_period_rets),
        "MVO": compute_metrics(mvo_period_rets),
        "EW": compute_metrics(ew_period_rets),
        "meta": {"w_crp_initial": w_crp, "r_daily": r_daily, "n_periods": n_periods, "tau": tau},
    }
    return results
