"""Diagnostic d'une trajectoire pour la couverture d'une option vendue.

Ce module ne modifie pas le moteur de calcul V12. Les positions, le cash,
les frais et le règlement final sont reconstruits à chaque pas afin de
contrôler leur concordance avec hedge_on_paths.
"""
from math import exp
import numpy as np
from option_mc_hedging_v12 import HedgeConfig, bs_price_delta, validate


def hedge_path_trace(c: HedgeConfig, paths, path_index: int = 0):
    """Historique d'une trajectoire: Delta unitaire, actions, cash, frais et trades.

    Les positions au temps k sont celles *après* un éventuel rééquilibrage.
    A maturité, les actions sont liquidées et le payoff réglé.
    """
    validate(c)
    paths = np.asarray(paths, dtype=float)
    if paths.shape != (c.n_steps + 1, c.n_paths):
        raise ValueError("Dimensions des trajectoires incompatibles")
    if not np.isfinite(paths).all() or np.any(paths <= 0):
        raise ValueError("Prix de trajectoires invalides")
    if not isinstance(path_index, (int, np.integer)) or not (0 <= path_index < c.n_paths):
        raise ValueError("Indice de trajectoire invalide")

    s = paths[:, path_index]
    dt = c.maturity / c.n_steps
    premium, delta0 = bs_price_delta(c.spot, c.strike, c.maturity, c.rate, c.volatility, c.option_type)
    shares = float(c.quantity * delta0)
    cash = float(c.quantity * premium - shares * c.spot)
    opening_fee = abs(shares) * c.spot * c.transaction_cost_rate
    cash -= opening_fee
    total_fees = opening_fee

    time = np.linspace(0, c.maturity, c.n_steps + 1)
    model_delta = np.empty(c.n_steps + 1)
    shares_held = np.empty(c.n_steps + 1)
    cash_balance = np.empty(c.n_steps + 1)
    traded_shares = np.zeros(c.n_steps + 1)
    fees_paid = np.zeros(c.n_steps + 1)
    is_rebalanced = np.zeros(c.n_steps + 1, dtype=bool)
    model_delta[0] = float(delta0)
    shares_held[0] = shares
    cash_balance[0] = cash
    traded_shares[0] = shares
    fees_paid[0] = opening_fee
    is_rebalanced[0] = True

    for k in range(1, c.n_steps + 1):
        cash *= exp(c.rate * dt)
        tau = c.maturity - k * dt
        _, current_delta = bs_price_delta(s[k], c.strike, tau, c.rate, c.volatility, c.option_type)
        model_delta[k] = float(current_delta)
        if k < c.n_steps and k % c.rebalance_every == 0:
            target = c.quantity * float(current_delta)
            trade = target - shares
            fee = abs(trade) * s[k] * c.transaction_cost_rate
            cash -= trade * s[k] + fee
            total_fees += fee
            shares = target
            traded_shares[k] = trade
            fees_paid[k] = fee
            is_rebalanced[k] = True
        shares_held[k] = shares
        cash_balance[k] = cash

    liquidation_fee = abs(shares) * s[-1] * c.transaction_cost_rate
    total_fees += liquidation_fee
    fees_paid[-1] += liquidation_fee
    traded_shares[-1] = -shares
    payoff = c.quantity * (max(s[-1] - c.strike, 0) if c.option_type == "call" else max(c.strike - s[-1], 0))
    final_pnl = cash + shares * s[-1] - liquidation_fee - payoff
    return {
        "time": time, "spot": s.copy(), "delta": model_delta,
        "shares": shares_held, "cash": cash_balance,
        "traded_shares": traded_shares, "fees_paid": fees_paid,
        "rebalanced": is_rebalanced,
        "final_pnl": float(final_pnl), "total_fees": float(total_fees),
        "terminal_payoff": float(payoff), "liquidation_fee": float(liquidation_fee),
        "premium_received": float(c.quantity * premium),
    }
