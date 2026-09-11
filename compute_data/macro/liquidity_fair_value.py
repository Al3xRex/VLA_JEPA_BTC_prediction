from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from scipy.optimize import minimize

from database_interraction import save_dataframe_to_table
# Allow running as a script (python macro/liquidity_fair_value.py) by adding project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from compute_data.macro.fred import DB_PATH, TABLE_NAME, construct_eco_data, load_btc_close
except ImportError:  # fallback for running as script
    from fred import DB_PATH, TABLE_NAME, construct_eco_data, load_btc_close  # type: ignore


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for liquidity plotting helpers.")


def safe_log(df: pd.DataFrame) -> pd.DataFrame:
    """Elementwise log with guard for non-positive entries."""
    return np.log(df.where(df > 0))


def make_slow_liquidity_features(
    raw_features: pd.DataFrame,
    resample_rule: str = "W-FRI",
    ewm_span: int | None = 120,
    yoy_periods: int | None = None,
    min_ewm_periods: int | None = None,
) -> pd.DataFrame:
    """
    Force liquidity drivers to move slower than price by:
      1) Resampling to lower frequency (weekly by default)
      2) Using YoY log changes
      3) Applying a long EWMA smoother
    """
    if resample_rule:
        features = raw_features.resample(resample_rule).last()
    else:
        features = raw_features.copy()

    if yoy_periods is None:
        if resample_rule and resample_rule.upper().startswith("M"):
            yoy_periods = 12
        elif resample_rule and resample_rule.upper().startswith("W"):
            yoy_periods = 52
        else:
            yoy_periods = 365  # daily fallback

    # YoY log changes (drop rows that cannot be computed cleanly)
    log_feats = safe_log(features)
    yoy = log_feats.diff(periods=yoy_periods)

    # Long EWMA to damp remaining high-frequency wiggles
    if ewm_span:
        if min_ewm_periods is None:
            min_ewm_periods = max(ewm_span // 4, 1)
        yoy = yoy.ewm(span=ewm_span, min_periods=min_ewm_periods).mean()

    return yoy.dropna(how="all")


def infer_steps_per_month(resample_rule: str | None) -> float:
    """Rough mapping from resample rule to steps per month for kappa scaling."""
    if not resample_rule:
        return 30.0
    rule = resample_rule.upper()
    if rule.startswith("W"):
        return 4.33
    if rule.startswith("M"):
        return 1.0
    return 30.0


def integrate_mean_reverting(
    ret_series: pd.Series,
    rho: float = 0.9,
    clip_k: float | None = None,
    base_level: float = 0.0,
) -> pd.Series:
    """
    Integrate returns into a mean-reverting level:
      L_t = rho * L_{t-1} + ret_t
    Optionally squash magnitude via tanh to avoid runaway contribution.
    """
    if ret_series.empty:
        return ret_series
    level_vals = []
    prev = base_level
    for _, ret in ret_series.items():
        lvl = rho * prev + ret
        if clip_k is not None and clip_k > 0:
            lvl = float(np.tanh(lvl / clip_k) * clip_k)
        level_vals.append(lvl)
        prev = lvl
    return pd.Series(level_vals, index=ret_series.index)


def compute_log_return_var(series: pd.Series) -> float:
    """
    Variance proxy for the observation noise.
    If the series is a price level (positive), use log returns.
    If the series can be negative (e.g., log returns already), fall back to plain variance.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return 1e-4
    if (clean <= 0).any():
        var = np.nanvar(clean.astype(float))
    else:
        log_ret = np.diff(np.log(clean.replace(0, np.nan))).astype(float)
        var = np.nanvar(log_ret)
    if not np.isfinite(var) or var <= 0:
        return 1e-4
    return float(var)


def standardize_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mu = values.mean()
    sigma = values.std(ddof=0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return (values - mu) / sigma


def split_train_test_index(
    index: pd.Index,
    train_frac: float,
    random_split: bool = False,
    random_state: int | None = None,
) -> tuple[pd.Index, pd.Index]:
    """
    Return train/test index slices with an optional random shuffle.
    If train_frac is outside (0, 1], the full index is used for training.
    """
    if train_frac <= 0 or train_frac > 1:
        return index, index[[]]

    split_idx = max(1, int(len(index) * train_frac))
    if random_split:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(index))
        train_idx = index.take(perm[:split_idx]).sort_values()
        test_idx = index.take(perm[split_idx:]).sort_values()
    else:
        train_idx = index[:split_idx]
        test_idx = index[split_idx:]
    return train_idx, test_idx


def run_kalman_fair_value(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    a: float = 0.99,
    b: np.ndarray | None = None,
    q: float = 1e-4,
    r: float = 1e-2,
    m0: float = 0.0,
    p0: float = 1.0,
) -> pd.DataFrame:
    """
    One-step Kalman recursion for a scalar state FV_t with linear predictors x_t.
    Returns a DataFrame with fair value (state mean), state variance, and innovations.
    """
    cols = [target_col] + list(feature_cols)
    data = df[cols].dropna().copy()
    if data.empty:
        raise ValueError("No data to run Kalman filter after dropping NaNs.")

    y = data[target_col].to_numpy(dtype=float)
    x = data[feature_cols].to_numpy(dtype=float)

    n_feat = x.shape[1]
    if b is None:
        b_vec = np.zeros(n_feat, dtype=float)
    else:
        b_vec = np.asarray(b, dtype=float).reshape(n_feat)

    m_prev = float(m0)
    p_prev = float(p0)

    m_list, p_list, e_list, s_list = [], [], [], []

    for i in range(len(data)):
        m_pred = a * m_prev + float(np.dot(b_vec, x[i]))
        p_pred = (a * a) * p_prev + q

        e = y[i] - m_pred
        s = p_pred + r
        k = p_pred / s

        m = m_pred + k * e
        p = (1.0 - k) * p_pred

        m_prev, p_prev = m, p

        m_list.append(m)
        p_list.append(p)
        e_list.append(e)
        s_list.append(s)

    out = pd.DataFrame(
        {
            "fair_value": m_list,
            "state_var": p_list,
            "innovation": e_list,
            "innovation_var": s_list,
        },
        index=data.index,
    )
    out.index.name = "Date" if data.index.name is None else data.index.name
    return out


def fit_kalman_fair_value(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    init_a: float = 0.99,
    init_b: np.ndarray | None = None,
    init_q: float | None = None,
    init_r: float | None = None,
    q_floor_factor: float = 1e-5,
    r_floor_factor: float = 5e-3,
    a_bounds: tuple[float, float] | None = (0.8, 1.05),
    fixed_a: float | None = None,
    recency_bias: float = 0.1,
    r_over_q_target: float = 100.0,
    ratio_penalty: float = 5.0,
    use_return_var_prior: bool = True,
    curvature_lambda: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit (a, b, Q, R) by maximizing the Gaussian log-likelihood via L-BFGS.
    curvature_lambda penalizes step-to-step changes in the filtered state
    (acts like a smoothness term when the state represents returns).
    Returns (filtered_df, best_params).
    """
    if target_col in feature_cols:
        raise ValueError("target_col must not be included in feature_cols.")

    cols = [target_col] + list(feature_cols)
    data = df[cols].dropna().copy()
    if data.empty:
        raise ValueError("No data to fit Kalman filter after dropping NaNs.")

    y = data[target_col].to_numpy(dtype=float)
    x = data[feature_cols].to_numpy(dtype=float)
    n_feat = x.shape[1]
    ret_var = compute_log_return_var(data[target_col])
    var_y = np.var(y)
    if init_r is None:
        init_r_val = max(ret_var if use_return_var_prior else 0.0, var_y * r_floor_factor, 1e-4)
    else:
        init_r_val = init_r
    if init_q is None:
        init_q_val = init_r_val / max(r_over_q_target, 1.0)
    else:
        init_q_val = init_q
    r_floor = max(ret_var if use_return_var_prior else 0.0, var_y * r_floor_factor, init_r_val * 0.5, 1e-6)
    q_floor = max(r_floor / max(r_over_q_target, 1.0), var_y * q_floor_factor, init_q_val * 0.25, 1e-8)
    t = np.arange(len(data), dtype=float)
    if len(data) > 1:
        scaled_t = (t - t.min()) / (t.max() - t.min())
    else:
        scaled_t = t
    weights = np.exp(recency_bias * (scaled_t - 1.0))
    weights = weights / weights.mean()

    if init_b is None:
        init_b_vec = np.zeros(n_feat, dtype=float)
    else:
        init_b_vec = np.asarray(init_b, dtype=float).reshape(n_feat)

    def unpack(theta: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        a = theta[0]
        b_vec = theta[1 : 1 + n_feat]
        q = q_floor + np.exp(theta[1 + n_feat])
        r = r_floor + np.exp(theta[2 + n_feat])
        return a, b_vec, q, r

    def neg_loglike(theta: np.ndarray) -> float:
        a, b_vec, q, r = unpack(theta)
        m_prev, p_prev = 0.0, 1.0
        ll = 0.0
        m_hist: list[float] = []
        for i in range(len(data)):
            m_pred = a * m_prev + float(np.dot(b_vec, x[i]))
            p_pred = (a * a) * p_prev + q
            s = p_pred + r
            if s <= 0:
                return 1e6
            e = y[i] - m_pred
            w = weights[i]
            ll += 0.5 * w * (np.log(s) + (e * e) / s)
            k = p_pred / s
            m_prev = m_pred + k * e
            p_prev = (1.0 - k) * p_pred
            m_hist.append(m_prev)
        log_ratio = np.log(max(r, 1e-12)) - np.log(max(q, 1e-12))
        shortfall = max(0.0, np.log(r_over_q_target) - log_ratio)
        ll += ratio_penalty * shortfall * shortfall
        if curvature_lambda > 0.0 and len(m_hist) > 1:
            delta_m = np.diff(np.array(m_hist))
            ll += curvature_lambda * float(np.mean(delta_m * delta_m))
        return ll

    theta0 = np.concatenate(
        (
            np.array([init_a if fixed_a is None else fixed_a]),
            init_b_vec,
            np.array([np.log(init_q_val), np.log(init_r_val)]),
        )
    )

    if fixed_a is not None:
        a_low, a_high = fixed_a, fixed_a
    else:
        a_low, a_high = a_bounds if a_bounds is not None else (-2.0, 2.0)

    bounds = [(a_low, a_high)] + [(None, None)] * n_feat + [(None, None), (None, None)]

    res = minimize(
        neg_loglike,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    a_hat, b_hat, q_hat, r_hat = unpack(res.x)

    filtered = run_kalman_fair_value(
        df=data,
        target_col=target_col,
        feature_cols=feature_cols,
        a=a_hat,
        b=b_hat,
        q=q_hat,
        r=r_hat,
    )

    params = {
        "a": float(a_hat),
        "b": b_hat.tolist(),
        "Q": float(q_hat),
        "R": float(r_hat),
        "success": bool(res.success),
        "message": res.message,
    }

    return filtered, params


def run_kalman_fair_value_with_deviation(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    b: np.ndarray | None,
    phi: float,
    q_fv: float,
    q_dev: float,
    r: float,
    m0_fv: float = 0.0,
    m0_dev: float = 0.0,
    p0: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Two-state filter:
      FV_t = FV_{t-1} + b^T x_t + w_t
      d_t = phi * d_{t-1} + u_t         (mean-reverting deviation)
      y_t = FV_t + d_t + v_t
    """
    cols = [target_col] + list(feature_cols)
    data = df[cols].dropna().copy()
    if data.empty:
        raise ValueError("No data to run 2-state Kalman filter after dropping NaNs.")

    y = data[target_col].to_numpy(dtype=float)
    x = data[feature_cols].to_numpy(dtype=float)
    n_feat = x.shape[1]
    b_vec = np.zeros(n_feat, dtype=float) if b is None else np.asarray(b, dtype=float).reshape(n_feat)

    m_prev = np.array([m0_fv, m0_dev], dtype=float)
    P_prev = np.eye(2) if p0 is None else np.asarray(p0, dtype=float).reshape(2, 2)

    A = np.array([[1.0, 0.0], [0.0, phi]], dtype=float)
    H = np.array([[1.0, 1.0]], dtype=float)
    Q = np.array([[q_fv, 0.0], [0.0, q_dev]], dtype=float)

    fv_list, dev_list, fv_plus_dev, p_fv_list, p_dev_list, p_cov_list, innov_list, innov_var_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(len(data)):
        fv_pred = m_prev[0] + float(np.dot(b_vec, x[i]))
        dev_pred = phi * m_prev[1]
        m_pred = np.array([fv_pred, dev_pred], dtype=float)

        P_pred = A @ P_prev @ A.T + Q

        y_pred = float((H @ m_pred).item())
        S = float((H @ P_pred @ H.T)[0, 0] + r)
        if S <= 0:
            S = 1e-8
        K = (P_pred @ H.T) / S  # shape (2, 1)
        innov = y[i] - y_pred
        m = m_pred + (K.flatten() * innov)
        P = (np.eye(2) - K @ H) @ P_pred

        m_prev, P_prev = m, P

        fv_list.append(m[0])
        dev_list.append(m[1])
        fv_plus_dev.append(m[0] + m[1])
        p_fv_list.append(P[0, 0])
        p_dev_list.append(P[1, 1])
        p_cov_list.append(P[0, 1])
        innov_list.append(innov)
        innov_var_list.append(S)

    out = pd.DataFrame(
        {
            "fair_value": fv_list,
            "deviation": dev_list,
            "fair_value_plus_dev": fv_plus_dev,
            "state_var_fv": p_fv_list,
            "state_var_dev": p_dev_list,
            "state_cov_fv_dev": p_cov_list,
            "innovation": innov_list,
            "innovation_var": innov_var_list,
        },
        index=data.index,
    )
    out.index.name = "Date" if data.index.name is None else data.index.name
    return out


def fit_kalman_fair_value_with_deviation(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    init_b: np.ndarray | None = None,
    init_phi: float = 0.6,
    init_q_fv: float | None = None,
    init_q_dev: float | None = None,
    init_r: float | None = None,
    phi_bound: float = 0.98,
    phi_min: float = 0.0,
    phi_penalty: float = 0.0,
    q_floor_factor: float = 1e-5,
    r_floor_factor: float = 5e-3,
    q_dev_floor_mult: float = 1.0,
    recency_bias: float = 0.1,
    r_over_q_target: float = 100.0,
    ratio_penalty: float = 5.0,
    use_return_var_prior: bool = True,
    curvature_lambda: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit a two-state model (fair value + deviation) via L-BFGS.
    The deviation state absorbs fast moves so FV stays sticky.
    """
    if target_col in feature_cols:
        raise ValueError("target_col must not be included in feature_cols.")

    cols = [target_col] + list(feature_cols)
    data = df[cols].dropna().copy()
    if data.empty:
        raise ValueError("No data to fit 2-state Kalman filter after dropping NaNs.")

    y = data[target_col].to_numpy(dtype=float)
    x = data[feature_cols].to_numpy(dtype=float)
    n_feat = x.shape[1]
    var_y = np.var(y)
    ret_var = compute_log_return_var(data[target_col])

    if init_r is None:
        init_r_val = max(ret_var if use_return_var_prior else 0.0, var_y * r_floor_factor, 1e-4)
    else:
        init_r_val = init_r
    if init_q_fv is None:
        init_q_fv_val = init_r_val / max(r_over_q_target, 1.0)
    else:
        init_q_fv_val = init_q_fv
    if init_q_dev is None:
        init_q_dev_val = init_r_val / 15.0
    else:
        init_q_dev_val = init_q_dev

    r_floor = max(ret_var if use_return_var_prior else 0.0, var_y * r_floor_factor, init_r_val * 0.5, 1e-6)
    q_fv_floor = max(r_floor / max(r_over_q_target, 1.0), var_y * q_floor_factor, init_q_fv_val * 0.25, 1e-8)
    q_dev_floor = max(r_floor / 25.0, var_y * q_floor_factor * 0.5, init_q_dev_val * 0.25, 1e-8)
    q_dev_floor *= max(q_dev_floor_mult, 1.0)
    phi_min = min(max(phi_min, 0.0), phi_bound - 1e-3)
    phi_penalty = max(phi_penalty, 0.0)

    t = np.arange(len(data), dtype=float)
    if len(data) > 1:
        scaled_t = (t - t.min()) / (t.max() - t.min())
    else:
        scaled_t = t
    weights = np.exp(recency_bias * (scaled_t - 1.0))
    weights = weights / weights.mean()

    b0 = np.zeros(n_feat, dtype=float) if init_b is None else np.asarray(init_b, dtype=float).reshape(n_feat)

    def map_phi(theta_phi: float) -> float:
        return float(phi_bound * np.tanh(theta_phi))

    def unpack(theta: np.ndarray) -> tuple[float, np.ndarray, float, float, float]:
        phi = map_phi(theta[0])
        b_vec = theta[1 : 1 + n_feat]
        q_fv = q_fv_floor + np.exp(theta[1 + n_feat])
        q_dev = q_dev_floor + np.exp(theta[2 + n_feat])
        r = r_floor + np.exp(theta[3 + n_feat])
        return phi, b_vec, q_fv, q_dev, r

    def neg_loglike(theta: np.ndarray) -> float:
        phi, b_vec, q_fv, q_dev, r = unpack(theta)
        A = np.array([[1.0, 0.0], [0.0, phi]], dtype=float)
        H = np.array([[1.0, 1.0]], dtype=float)
        Q = np.array([[q_fv, 0.0], [0.0, q_dev]], dtype=float)

        m_prev = np.zeros(2, dtype=float)
        P_prev = np.eye(2)
        ll = 0.0
        fv_hist: list[float] = []
        for i in range(len(data)):
            fv_pred = m_prev[0] + float(np.dot(b_vec, x[i]))
            dev_pred = phi * m_prev[1]
            m_pred = np.array([fv_pred, dev_pred], dtype=float)

            P_pred = A @ P_prev @ A.T + Q
            S = float((H @ P_pred @ H.T)[0, 0] + r)
            if S <= 0:
                return 1e6
            innov = y[i] - float((H @ m_pred).item())
            w = weights[i]
            ll += 0.5 * w * (np.log(S) + (innov * innov) / S)

            K = (P_pred @ H.T) / S
            m_prev = m_pred + (K.flatten() * innov)
            P_prev = (np.eye(2) - K @ H) @ P_pred
            fv_hist.append(m_prev[0])

        log_ratio = np.log(max(r, 1e-12)) - np.log(max(q_fv, 1e-12))
        shortfall = max(0.0, np.log(r_over_q_target) - log_ratio)
        ll += ratio_penalty * shortfall * shortfall
        if phi_min > 0.0 and phi_penalty > 0.0:
            phi_shortfall = max(0.0, phi_min - phi)
            ll += phi_penalty * phi_shortfall * phi_shortfall
        if curvature_lambda > 0.0 and len(fv_hist) > 1:
            delta_fv = np.diff(np.array(fv_hist))
            ll += curvature_lambda * float(np.mean(delta_fv * delta_fv))
        return ll

    if phi_min > 0.0:
        phi_init_low = min(phi_bound - 1e-3, max(0.0, phi_min) + 1e-3)
    else:
        phi_init_low = -phi_bound + 1e-3
    phi_init_clamped = np.clip(init_phi, phi_init_low, phi_bound - 1e-3)
    theta0 = np.concatenate(
        (
            np.array([np.arctanh(phi_init_clamped / phi_bound)]),
            b0,
            np.array(
                [
                    np.log(init_q_fv_val),
                    np.log(init_q_dev_val),
                    np.log(init_r_val),
                ]
            ),
        )
    )

    bounds = [(None, None)] * (1 + n_feat + 3)
    res = minimize(
        neg_loglike,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    phi_hat, b_hat, q_fv_hat, q_dev_hat, r_hat = unpack(res.x)

    filtered = run_kalman_fair_value_with_deviation(
        df=data,
        target_col=target_col,
        feature_cols=feature_cols,
        b=b_hat,
        phi=phi_hat,
        q_fv=q_fv_hat,
        q_dev=q_dev_hat,
        r=r_hat,
    )

    params = {
        "phi": float(phi_hat),
        "b": b_hat.tolist(),
        "Q_fv": float(q_fv_hat),
        "Q_dev": float(q_dev_hat),
        "R": float(r_hat),
        "success": bool(res.success),
        "message": res.message,
    }

    return filtered, params




def fit_and_plot(
    raw_features: pd.DataFrame,
    price: pd.Series,
    title: str,
    ax: plt.Axes | None = None,
    plot: bool = True,
    train_frac: float = 0.8,
    r_floor_factor: float = 5e-3,
    q_floor_factor: float = 1e-5,
    fixed_a: float | None = None,
    a_bounds: tuple[float, float] | None = (0.8, 1.05),
    recency_bias: float = 0.1,
    resid_scale: float = 1.0,
    standardize_features: bool = True,
    use_deviation_state: bool = True,
    r_over_q_target: float = 80.0,
    ratio_penalty: float = 5.0,
    slow_liquidity: bool = False,
    resample_rule: str | None = None,
    ewm_span: int | None = 60,
    yoy_periods: int | None = None,
    use_pc1: bool = True,
    use_dxy_split: bool = True,
    dxy_pattern: str = "dxy",
    include_lagged_price: bool = True,
    lagged_price_span: int = 60,
    lagged_price_scale: float = 1.25,
    secular_span_months: float = 15.0,
    rho_liq: float = 0.97,
    liq_clip_k: float | None = 1.3,
    curvature_lambda: float = 0.5,
    phi_min: float = 0.25,
    phi_penalty: float = 3.0,
    q_dev_floor_mult: float = 3.0,
    export_db_path: str | None = None,
    export_table: str = "value_liquidity_fv",
    random_split: bool = False,
    random_state: int | None = None,
) -> tuple[plt.Axes | None, dict]:
    """
    Model detrended BTC log returns on liquidity changes (optionally slowed), integrate the predicted cycles
    with mean reversion, and plot secular + liquidity + deviation decomposition.
    """
    # 1) optionally slow down liquidity
    if slow_liquidity:
        slow_features = make_slow_liquidity_features(
            raw_features,
            resample_rule=resample_rule,
            ewm_span=ewm_span,
            yoy_periods=yoy_periods,
        )
    else:
        if resample_rule:
            slow_features = raw_features.resample(resample_rule).last()
        else:
            slow_features = raw_features.copy()

    # 2) align price to same frequency and build secular baseline
    if resample_rule:
        price_rs = price.resample(resample_rule).last()
    else:
        price_rs = price.copy()
    log_price = np.log(price_rs)
    steps_per_month = infer_steps_per_month(resample_rule)
    secular_span_steps = max(5, int(round(secular_span_months * steps_per_month)))
    secular_log = log_price.ewm(span=secular_span_steps, min_periods=max(5, secular_span_steps // 4)).mean()
    detrended_log = (log_price - secular_log).dropna()
    log_ret = detrended_log.diff().rename("log_ret")

    merged = slow_features.join(log_ret, how="inner").dropna()
    if merged.empty:
        raise ValueError("No overlapping data between features and price after resampling/transform.")

    feature_cols = [c for c in merged.columns if c != "log_ret"]

    non_pc1_cols: list[str] = []
    if include_lagged_price:
        lagged_price = (
            log_ret.ewm(span=lagged_price_span, min_periods=max(5, lagged_price_span // 4)).mean().shift(1)
            * lagged_price_scale
        ).rename("lagged_price_ema")
        merged = merged.join(lagged_price, how="left")
        non_pc1_cols.append("lagged_price_ema")
        merged = merged.dropna(subset=["log_ret"] + non_pc1_cols)

    train_idx, test_idx = split_train_test_index(
        merged.index,
        train_frac=train_frac,
        random_split=random_split,
        random_state=random_state,
    )

    # Optional dimensionality reduction to kill collinearity
    if use_pc1 and len(feature_cols) > 1:
        dxy_cols = [c for c in feature_cols if dxy_pattern.lower() in c.lower()] if use_dxy_split else []
        liq_cols = [c for c in feature_cols if c not in dxy_cols + non_pc1_cols]
        new_cols = []
        merged_parts = []

        if len(liq_cols) > 1:
            train_feats = merged.loc[train_idx, liq_cols]
            mu_pc = train_feats.mean()
            sigma_pc = train_feats.std().replace(0, 1.0).fillna(1.0)
            train_std = (train_feats - mu_pc) / sigma_pc
            _, _, vh = np.linalg.svd(train_std.to_numpy(), full_matrices=False)
            pc1_weights = vh[0]
            pc1_all = ((merged[liq_cols] - mu_pc) / sigma_pc) @ pc1_weights
            merged_parts.append(pc1_all.rename("liquidity_pc1"))
            new_cols.append("liquidity_pc1")
        else:
            merged_parts.append(merged[liq_cols])
            new_cols.extend(liq_cols)

        if dxy_cols:
            merged_parts.append(merged[dxy_cols])
            new_cols.extend(dxy_cols)

        if non_pc1_cols:
            merged_parts.append(merged[non_pc1_cols])
            new_cols.extend(non_pc1_cols)

        merged = pd.concat(merged_parts + [merged["log_ret"]], axis=1)
        feature_cols = new_cols
    else:
        if include_lagged_price and "lagged_price_ema" not in feature_cols:
            feature_cols = feature_cols + non_pc1_cols

    if standardize_features:
        mu = merged.loc[train_idx, feature_cols].mean()
        sigma = merged.loc[train_idx, feature_cols].std().replace(0, 1.0).fillna(1.0)
        merged_std = merged.copy()
        merged_std[feature_cols] = (merged_std[feature_cols] - mu) / sigma
        merged = merged_std

    train = merged.loc[train_idx]
    test = merged.loc[test_idx]

    aligned_log_price_full = log_price.loc[merged.index]
    secular_aligned = secular_log.reindex(merged.index).ffill()

    if use_deviation_state:
        filtered_train, params = fit_kalman_fair_value_with_deviation(
            train,
            target_col="log_ret",
            feature_cols=feature_cols,
            q_floor_factor=q_floor_factor,
            r_floor_factor=r_floor_factor,
            phi_min=phi_min,
            phi_penalty=phi_penalty,
            q_dev_floor_mult=q_dev_floor_mult,
            recency_bias=recency_bias,
            r_over_q_target=r_over_q_target,
            ratio_penalty=ratio_penalty,
            curvature_lambda=curvature_lambda,
        )
        if not test.empty:
            last_cov = float(filtered_train["state_cov_fv_dev"].iloc[-1])
            p0 = np.array(
                [
                    [float(filtered_train["state_var_fv"].iloc[-1]), last_cov],
                    [last_cov, float(filtered_train["state_var_dev"].iloc[-1])],
                ]
            )
            filtered_test = run_kalman_fair_value_with_deviation(
                df=test,
                target_col="log_ret",
                feature_cols=feature_cols,
                b=np.array(params["b"]),
                phi=params["phi"],
                q_fv=params["Q_fv"],
                q_dev=params["Q_dev"],
                r=params["R"],
                m0_fv=float(filtered_train["fair_value"].iloc[-1]),
                m0_dev=float(filtered_train["deviation"].iloc[-1]),
                p0=p0,
            )
            filtered = pd.concat([filtered_train, filtered_test])
        else:
            filtered = filtered_train
        pred_col = "fair_value_plus_dev"
    else:
        filtered_train, params = fit_kalman_fair_value(
            train,
            target_col="log_ret",
            feature_cols=feature_cols,
            r_floor_factor=r_floor_factor,
            q_floor_factor=q_floor_factor,
            fixed_a=fixed_a,
            a_bounds=a_bounds,
            recency_bias=recency_bias,
            r_over_q_target=r_over_q_target,
            ratio_penalty=ratio_penalty,
            curvature_lambda=curvature_lambda,
        )
        if not test.empty:
            filtered_test = run_kalman_fair_value(
                df=test,
                target_col="log_ret",
                feature_cols=feature_cols,
                a=params["a"],
                b=np.array(params["b"]),
                q=params["Q"],
                r=params["R"],
                m0=float(filtered_train["fair_value"].iloc[-1]),
                p0=float(filtered_train["state_var"].iloc[-1]),
            )
            filtered = pd.concat([filtered_train, filtered_test])
        else:
            filtered = filtered_train
        pred_col = "fair_value"

    # integrate predicted returns to a level fair value curve with mean reversion, then add secular
    filtered = filtered.loc[log_price.index.intersection(filtered.index)]
    if filtered.empty:
        raise ValueError("Filtered output is empty after aligning with price index.")
    base_detrended = float(detrended_log.loc[filtered.index[0]]) if not detrended_log.empty else 0.0

    liq_level = integrate_mean_reverting(
        filtered["fair_value"],
        rho=rho_liq,
        clip_k=liq_clip_k,
        base_level=base_detrended,
    )
    fv_log_level = secular_aligned.reindex(liq_level.index).ffill() + liq_level

    liq_dev_level = integrate_mean_reverting(
        filtered[pred_col],
        rho=rho_liq,
        clip_k=liq_clip_k,
        base_level=base_detrended,
    )
    fv_plus_dev_log_level = secular_aligned.reindex(liq_dev_level.index).ffill() + liq_dev_level
    if use_deviation_state:
        dev_log_level = fv_plus_dev_log_level - fv_log_level
    else:
        dev_log_level = None

    aligned_log_price = log_price.loc[filtered.index]
    aligned_price = price_rs.reindex(filtered.index)
    fair_value_level = np.exp(fv_plus_dev_log_level).rename("fair_value")
    dev_osc = (aligned_price - fair_value_level).rename("dev_osc")
    dev_osc_z = standardize_series(dev_osc).rename("dev_osc_z")

    train_idx = filtered_train.index.intersection(filtered.index)
    resid_train = aligned_log_price.loc[train_idx] - fv_plus_dev_log_level.loc[train_idx]
    resid_std = resid_train.std()
    if np.isnan(resid_std) or resid_std <= 0:
        resid_std = 1e-6
    resid_std *= resid_scale

    btc_ret = log_price.diff()
    btc_ret_train_std = btc_ret.loc[train_idx].std()
    denom_vol = btc_ret_train_std if np.isfinite(btc_ret_train_std) and btc_ret_train_std > 0 else 1.0
    vol_adj_series = (btc_ret.rolling(window=12).std() / denom_vol).reindex(fv_plus_dev_log_level.index)
    vol_adj_series = vol_adj_series.fillna(1.0).clip(0.8, 1.3)
    sigma_series = resid_std * vol_adj_series

    band1_upper = fv_plus_dev_log_level + sigma_series
    band1_lower = fv_plus_dev_log_level - sigma_series
    band2_upper = fv_plus_dev_log_level + 2.0 * sigma_series
    band2_lower = fv_plus_dev_log_level - 2.0 * sigma_series

    band_plus_1 = np.exp(band1_upper).rename("band_plus_1")
    band_minus_1 = np.exp(band1_lower).rename("band_minus_1")
    band_plus_2 = np.exp(band2_upper).rename("band_plus_2")
    band_minus_2 = np.exp(band2_lower).rename("band_minus_2")

    if export_db_path:
        export_df = pd.concat(
            [
                fair_value_level.rename("fair_value"),
                dev_osc_z,
                band_plus_1,
                band_minus_1,
                band_plus_2,
                band_minus_2,
            ],
            axis=1,
        )
        export_df = export_df.dropna(subset=["fair_value", "dev_osc_z"])
        export_df.index.name = "Date"
        rows = save_dataframe_to_table(
            export_df,
            export_table,
            db_path=export_db_path,
            preserve_index=True,
        )
        print(f"Saved liquidity fair value to {export_db_path}: {export_table} ({rows} rows)")

    ax_out = None
    if plot:
        _require_matplotlib()
        ax_out = ax or plt.gca()
        ax_out.plot(aligned_log_price.index, np.exp(aligned_log_price), label="BTC close", color="black", linewidth=1.2)
        ax_out.plot(secular_aligned.index, np.exp(secular_aligned), label="Secular baseline", color="tab:gray", linestyle="--", linewidth=1.0)
        ax_out.plot(fv_plus_dev_log_level.index, np.exp(fv_plus_dev_log_level), label="Fair Value (sec + liq + dev)", color="tab:blue", linewidth=1.4)
        ax_out.plot(fv_log_level.index, np.exp(fv_log_level), label="Liquidity component (sec + liq)", color="tab:green", linewidth=1.2, alpha=0.9)
        shade_color = "tab:orange" if use_deviation_state else "tab:blue"
        if use_deviation_state and dev_log_level is not None:
            ax_out.plot(dev_log_level.index, np.exp(dev_log_level + secular_aligned.reindex(dev_log_level.index).ffill()), label="Deviation component", color="tab:red", linestyle="--", linewidth=1.0, alpha=0.6)

        ax_out.fill_between(fv_plus_dev_log_level.index, np.exp(band1_lower), np.exp(band1_upper), color=shade_color, alpha=0.12, label="±1σ (log resid)")
        ax_out.fill_between(fv_plus_dev_log_level.index, np.exp(band2_lower), np.exp(band2_upper), color=shade_color, alpha=0.07, label="±2σ (log resid)")
        ax_out.set_title(title)
        ax_out.set_xlabel("Date")
        ax_out.set_ylabel("Price")
        ax_out.set_yscale("log")
        ax_out.legend()
        ax_out.grid(True, alpha=0.3)
    return ax_out, params



def plot_liquidity_fair_value(
    fred_db_path: str = DB_PATH,
    fred_table: str = TABLE_NAME,
    price_db_path: str = "database/ohlcv.duckdb",
    price_table: str = "ohlcv",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    show: bool = False,
    save_path: str | None = None,
    train_frac: float = 0.8,
    r_floor_factor: float = 5e-3,
    q_floor_factor: float = 1e-5,
    fixed_a: float | None = None,
    a_bounds: tuple[float, float] | None = (0.8, 1.05),
    recency_bias: float = 0.08,
    resid_scale: float = 1.0,
    standardize_features: bool = True,
    use_deviation_state: bool = True,
    r_over_q_target: float = 80.0,
    slow_liquidity: bool = True,
    resample_rule: str | None = None,
    ewm_span: int | None = 60,
    yoy_periods: int | None = None,
    use_pc1: bool = True,
    use_dxy_split: bool = True,
    dxy_pattern: str = "dxy",
    include_lagged_price: bool = True,
    lagged_price_span: int = 45,
    lagged_price_scale: float = 1.4,
    secular_span_months: float = 12.0,
    rho_liq: float = 0.94,
    liq_clip_k: float | None = 1.6,
    curvature_lambda: float = 0.3,
    phi_min: float = 0.4,
    phi_penalty: float = 3.0,
    q_dev_floor_mult: float = 4.0,
    export_db_path: str | None = "orvian.duckdb",
    export_table: str = "value_liquidity_fv",
    random_split: bool = True,
    random_state: int | None = None,
) -> tuple[plt.Figure, dict]:
    """
    Build the liquidity DataFrame (WALCL - TGA - RRP + BTFP + PC + shifted M2 + 1/DXY),
    optionally slow it (resample + YoY + EWMA), fit on BTC log returns, integrate back to level,
    and render a single plot.
    """
    _require_matplotlib()
    price = load_btc_close(
        db_path=price_db_path,
        table_name=price_table,
        symbol=symbol,
        interval=interval,
    )
    end_date = price.index.max() if not price.empty else None
    liq_m2_dxy, _ = construct_eco_data(
        end_date=end_date,
        fred_db_path=fred_db_path,
        fred_table=fred_table,
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharex=False)
    _, params = fit_and_plot(
        liq_m2_dxy,
        price,
        title="Fair Value Model: Liquidity + M2 (shifted) + 1/DXY",
        ax=ax,
        train_frac=train_frac,
        r_floor_factor=r_floor_factor,
        q_floor_factor=q_floor_factor,
        fixed_a=fixed_a,
        a_bounds=a_bounds,
        recency_bias=recency_bias,
        resid_scale=resid_scale,
        standardize_features=standardize_features,
        use_deviation_state=use_deviation_state,
        r_over_q_target=r_over_q_target,
        slow_liquidity=slow_liquidity,
        resample_rule=resample_rule,
        ewm_span=ewm_span,
        yoy_periods=yoy_periods,
        use_pc1=use_pc1,
        use_dxy_split=use_dxy_split,
        dxy_pattern=dxy_pattern,
        include_lagged_price=include_lagged_price,
        lagged_price_span=lagged_price_span,
        lagged_price_scale=lagged_price_scale,
        secular_span_months=secular_span_months,
        rho_liq=rho_liq,
        liq_clip_k=liq_clip_k,
        curvature_lambda=curvature_lambda,
        phi_min=phi_min,
        phi_penalty=phi_penalty,
        q_dev_floor_mult=q_dev_floor_mult,
        export_db_path=export_db_path,
        export_table=export_table,
        random_split=random_split,
        random_state=random_state,
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, params


def store_liquidity_fair_value(
    *,
    orvian_db_path: str = "orvian.duckdb",
    table_name: str = "value_liquidity_fv",
    fred_db_path: str = DB_PATH,
    fred_table: str = TABLE_NAME,
    price_db_path: str = "database/ohlcv.duckdb",
    price_table: str = "ohlcv",
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    end_date: pd.Timestamp | str | None = None,
    **model_kwargs,
) -> dict:
    price = load_btc_close(
        db_path=price_db_path,
        table_name=price_table,
        symbol=symbol,
        interval=interval,
    )
    if end_date is not None:
        cutoff = pd.Timestamp(end_date)
        if cutoff.tz is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        price = price.loc[price.index <= cutoff]
    end_date = price.index.max() if not price.empty else None
    liq_m2_dxy, _ = construct_eco_data(
        end_date=end_date,
        fred_db_path=fred_db_path,
        fred_table=fred_table,
    )

    _, params = fit_and_plot(
        liq_m2_dxy,
        price,
        title="Fair Value Model: Liquidity + M2 (shifted) + 1/DXY",
        plot=False,
        export_db_path=orvian_db_path,
        export_table=table_name,
        **model_kwargs,
    )
    return params
