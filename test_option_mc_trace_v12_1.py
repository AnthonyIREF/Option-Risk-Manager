"""V12.1: independent path diagnostic vs the unmodified V12 financial engine."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from option_mc_hedging_v12 import HedgeConfig, simulate_paths, hedge_on_paths
from option_mc_trace_v12_1 import hedge_path_trace


def test_pathwise_pnl_and_fees():
    for option in ('call','put'):
        for frequency in (1,5,21):
            for fee in (0.,0.001):
                c=HedgeConfig(option_type=option,n_steps=42,n_paths=30,rebalance_every=frequency,
                              transaction_cost_rate=fee,seed=42,quantity=100.)
                paths=simulate_paths(c)
                result=hedge_on_paths(c,paths)
                for j in range(c.n_paths):
                    trace=hedge_path_trace(c,paths,j)
                    assert np.isclose(trace['final_pnl'],result['pnl'][j],atol=1e-8)
                    assert np.isclose(trace['total_fees'],result['trading_cost'][j],atol=1e-8)
                    assert np.isclose(trace['shares'][0],result['initial_shares'])
    print('PASS test_pathwise_pnl_and_fees')


def test_stepwise_rebalance():
    c=HedgeConfig(n_steps=42,n_paths=3,rebalance_every=5,seed=3)
    tr=hedge_path_trace(c,simulate_paths(c),0)
    assert np.all(tr['rebalanced'][[0,5,10,15,20,25,30,35,40]])
    assert not tr['rebalanced'][-1]
    for k in range(1,c.n_steps):
        if k%5 != 0:
            assert tr['shares'][k] == tr['shares'][k-1]
    assert tr['traded_shares'][-1] == -tr['shares'][-1]
    print('PASS test_stepwise_rebalance')


def test_put_short_negative_hedge():
    c=HedgeConfig(option_type='put',n_steps=21,n_paths=2,rebalance_every=1)
    tr=hedge_path_trace(c,simulate_paths(c),0)
    assert tr['shares'][0] < 0
    assert np.all((tr['delta']>=-1)&(tr['delta']<=0))
    print('PASS test_put_short_negative_hedge')


def test_invalid_inputs():
    c=HedgeConfig(n_steps=3,n_paths=2)
    paths=simulate_paths(c)
    for p,idx in ((paths,-1),(paths,2),(paths[:2],0)):
        try:
            hedge_path_trace(c,p,idx)
        except ValueError:
            continue
        raise AssertionError('invalid input accepted')
    print('PASS test_invalid_inputs')


def test_visual_structure_and_no_prior_modes_changed():
    src=Path(__file__).with_name('app_v12_1_visualisations_jupyter.txt').read_text(encoding='utf-8')
    prev=Path(__file__).with_name('app_v12_monte_carlo_jupyter.txt').read_text(encoding='utf-8')
    assert 'Suivi d’une couverture Delta sur une trajectoire' in src
    assert 'Compromis entre précision et coût de couverture' in src
    assert src[:src.index('# MODE 3 — MONTE-CARLO')] .replace('from option_mc_trace_v12_1 import hedge_path_trace\n','') == prev[:prev.index('# MODE 3 — MONTE-CARLO')]
    assert src.startswith('%%writefile app.py\n')
    print('PASS test_visual_structure_and_no_prior_modes_changed')


if __name__=='__main__':
    for fn in (test_pathwise_pnl_and_fees,test_stepwise_rebalance,test_put_short_negative_hedge,
               test_invalid_inputs,test_visual_structure_and_no_prior_modes_changed):
        fn()
    print('RESULTAT : 5/5 groupes réussis')
