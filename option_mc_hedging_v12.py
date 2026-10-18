"""V12 - Independent Monte Carlo engine for a self-financing delta hedge.

Conventions: European option sold (short one or more contracts); shares held are
+quantity*Black-Scholes delta. The initial premium finances the initial hedge.
Cash earns/pays continuously compounded r between trading times. A trade in
shares costs abs(shares traded)*spot*transaction_cost_rate. Terminal shares are
liquidated, including transaction costs, and the option payoff is settled.
All results are monetary amounts, not percentage returns. No dividends.
"""
from dataclasses import dataclass
from math import erf, exp, log, sqrt
import numpy as np


@dataclass(frozen=True)
class HedgeConfig:
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 0.5
    rate: float = 0.03
    volatility: float = 0.20
    option_type: str = "call"
    quantity: float = 100.0
    trading_days_per_year: int = 252
    n_steps: int = 126
    n_paths: int = 10000
    rebalance_every: int = 1
    transaction_cost_rate: float = 0.0
    drift: float | None = None
    seed: int = 42


def validate(c: HedgeConfig) -> None:
    if c.spot <= 0 or c.strike <= 0 or c.maturity <= 0:
        raise ValueError("Spot, strike and maturity must be positive")
    if c.volatility <= 0 or not np.isfinite(c.volatility):
        raise ValueError("Volatility must be positive and finite")
    if c.option_type not in ("call", "put"):
        raise ValueError("option_type must be call or put")
    if c.quantity <= 0 or not np.isfinite(c.quantity):
        raise ValueError("quantity must be positive and finite")
    if not isinstance(c.n_steps, int) or c.n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not isinstance(c.n_paths, int) or c.n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    if not isinstance(c.rebalance_every, int) or c.rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer")
    if not (0 <= c.transaction_cost_rate < 1):
        raise ValueError("transaction_cost_rate must lie in [0,1)")
    if not np.isfinite(c.rate) or (c.drift is not None and not np.isfinite(c.drift)):
        raise ValueError("Rate and drift must be finite")
    if not isinstance(c.seed, (int, np.integer)) or c.seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _normal_cdf(x):
    # Python's math.erf is scalar; vectorized approximation via scipy when present.
    from scipy.special import ndtr
    return ndtr(x)


def bs_price_delta(spot, strike, tau, rate, vol, option_type):
    """Return Black-Scholes price and delta, with exact expiry handling."""
    s = np.asarray(spot, dtype=float)
    if tau <= 0:
        intrinsic = np.maximum(s - strike, 0) if option_type == "call" else np.maximum(strike - s, 0)
        delta = np.where(s > strike, 1.0, np.where(s < strike, 0.0, 0.5))
        if option_type == "put":
            delta = delta - 1.0
        return intrinsic, delta
    root = vol * sqrt(tau)
    d1 = (np.log(s / strike) + (rate + 0.5 * vol * vol) * tau) / root
    d2 = d1 - root
    disc = exp(-rate * tau)
    if option_type == "call":
        return s * _normal_cdf(d1) - strike * disc * _normal_cdf(d2), _normal_cdf(d1)
    return strike * disc * _normal_cdf(-d2) - s * _normal_cdf(-d1), _normal_cdf(d1) - 1.0


def simulate_paths(c: HedgeConfig, normals=None):
    """Exact GBM steps. Shape (n_steps+1, n_paths); drift defaults to r."""
    validate(c)
    if normals is None:
        normals = np.random.default_rng(c.seed).standard_normal((c.n_steps, c.n_paths))
    else:
        normals = np.asarray(normals, dtype=float)
        if normals.shape != (c.n_steps, c.n_paths) or not np.isfinite(normals).all():
            raise ValueError("normals must have shape (n_steps, n_paths) and be finite")
    dt = c.maturity / c.n_steps
    mu = c.rate if c.drift is None else c.drift
    increments = (mu - 0.5 * c.volatility**2) * dt + c.volatility * sqrt(dt) * normals
    prices = np.empty((c.n_steps + 1, c.n_paths))
    prices[0] = c.spot
    prices[1:] = c.spot * np.exp(np.cumsum(increments, axis=0))
    return prices


def hedge_on_paths(c: HedgeConfig, paths):
    """Self-financing hedge of SHORT options on a given common set of paths."""
    validate(c)
    paths = np.asarray(paths, dtype=float)
    if paths.shape != (c.n_steps + 1, c.n_paths) or not np.isfinite(paths).all() or np.any(paths <= 0):
        raise ValueError("Invalid paths shape or prices")
    dt = c.maturity / c.n_steps
    premium, delta0 = bs_price_delta(c.spot, c.strike, c.maturity, c.rate, c.volatility, c.option_type)
    shares = np.full(c.n_paths, float(c.quantity * delta0))
    cash = np.full(c.n_paths, float(c.quantity * premium - shares[0] * c.spot))
    initial_cost = np.abs(shares) * c.spot * c.transaction_cost_rate
    cash -= initial_cost
    trading_cost = initial_cost.copy()
    turnover = np.abs(shares) * c.spot
    for k in range(1, c.n_steps + 1):
        cash *= exp(c.rate * dt)
        if k < c.n_steps and k % c.rebalance_every == 0:
            tau = c.maturity - k * dt
            _, delta = bs_price_delta(paths[k], c.strike, tau, c.rate, c.volatility, c.option_type)
            new_shares = c.quantity * delta
            trade = new_shares - shares
            notional = np.abs(trade) * paths[k]
            fee = c.transaction_cost_rate * notional
            cash -= trade * paths[k] + fee
            trading_cost += fee
            turnover += notional
            shares = new_shares
    terminal = paths[-1]
    final_fee = c.transaction_cost_rate * np.abs(shares) * terminal
    trading_cost += final_fee
    turnover += np.abs(shares) * terminal
    payoff = c.quantity * (np.maximum(terminal - c.strike, 0) if c.option_type == "call" else np.maximum(c.strike - terminal, 0))
    pnl = cash + shares * terminal - final_fee - payoff
    return {
        "pnl": pnl,
        "terminal_spot": terminal.copy(),
        "premium_received": float(c.quantity * premium),
        "initial_shares": float(c.quantity * delta0),
        "trading_cost": trading_cost,
        "turnover": turnover,
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)) if c.n_paths > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(pnl**2))),
        "p05": float(np.quantile(pnl, 0.05)),
        "median": float(np.median(pnl)),
        "p95": float(np.quantile(pnl, 0.95)),
    }


def simulate_hedge(c: HedgeConfig):
    return hedge_on_paths(c, simulate_paths(c))


def compare_frequencies(c: HedgeConfig, frequencies=(1, 5, 21)):
    """All frequencies share EXACTLY the same simulated price trajectories."""
    paths = simulate_paths(c)
    from dataclasses import replace
    return {f: hedge_on_paths(replace(c, rebalance_every=f), paths) for f in frequencies}
