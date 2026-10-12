"""
tyndp_battery_benchmark.py -- S.103: where does TYNDP's own planned battery fleet
sit relative to the mitigation frontier?

For each scenario: solve the cost-optimal baseline B_opt, then the same system
with battery power fixed at the TYNDP fleet (utility + prosumer, per node; energy
tied at 4 h as everywhere in the paper), all other flexibility cost-optimal.
Reports premium vs B_opt and the resulting VF_min / VF_gw / CR_min, so the point
can be placed on the frontier (make_paper_numbers.py compares it with the
cheapest frontier portfolio reaching the same VF_min).

TYNDP battery power: Installed-capacity rows are ENERGY (MWh) in the DE files ->
power = utility/4 h + prosumer(RETE)/3.2 h (checked against hourly max dispatch
in DE2040: DE 15.5 + ~47-53 GW, NL 7.7 + ~18.5-19.7 GW). NT2030 rows are MW.
Uses the same scenario conditioning as directed_search.py, incl. TYNDP_FIXES.
"""
import os, sys, json, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_representative_weeks
from scenarios import apply_scenario
from directed_search import LPOracle
import tyndp_fixes as _tf

def tyndp_battery_mw(scenario):
    X = json.load(open(_tf._EXTRAS)); S = X[scenario]
    out = {}
    for n in ('DE', 'CZ', 'DK', 'AT', 'NL'):
        b = S[n]['battery']
        out[n] = b['util_row'] if scenario == 'nt2030' else b['util_row'] / 4.0 + b['rete_row'] / 3.2
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', default='nt2030,de2035,de2040,de2050')
    ap.add_argument('--n-weeks', type=int, default=12)
    ap.add_argument('--climate-year', type=int, default=2009)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pecd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tyndp 2040', 'PECD')
    rows = []
    for sc in [s.strip() for s in a.scenarios.split(',')]:
        D = load_representative_weeks(os.path.join(root, 'time_series.xlsx'),
                                      os.path.join(root, 'data_input.xlsx'),
                                      n_weeks=a.n_weeks, nodes=['DE', 'CZ', 'DK', 'AT', 'NL'],
                                      climate_year=a.climate_year, pecd_dir=pecd)
        apply_scenario(D, sc)
        _tf.apply(D, sc)
        D['p2g_endogenous'] = True
        from data_pipeline import build_ev_2040, scale_demand_to_targets
        D['ev_capex'] = 0.0; build_ev_2040(D); scale_demand_to_targets(D)
        eta = D.get('sc_eta', 0.70); _EL = D.get('h2_elec_twh')
        D['h2_targets'] = {n: _EL[n] * 1e6 * eta for n in D['n'] if n in _EL}
        oracle = LPOracle(D)
        base = oracle.solve_cost_optimal()
        if base is None:
            raise SystemExit(f'{sc}: B_opt failed')
        rec = {'scenario': sc, 'case': 'B_opt', 'sys_cost_bn': base['sys_cost'] / 1e9, 'premium_bn': 0.0,
               'vf_min': oracle.compute_vf_min(base), 'vf_genw': base.get('vf_genw'),
               'cr_min': oracle.compute_cr_min(base)}
        rec.update({f'sto_{n.lower()}_mw': base.get('sto_mw' if n == 'DE' else f'sto_{n.lower()}_mw', 0.0) for n in D['n']})
        rows.append(rec)
        P = tyndp_battery_mw(sc)
        m = oracle.model
        for n, p in P.items():
            ub = m.K_sto_p[n].ub
            if ub is not None and p > ub + 1e-6:
                print(f'  [bench] {sc} {n}: TYNDP battery {p/1e3:.1f} GW exceeds potential '
                      f'{ub/1e3:.1f} GW -> capped (use TYNDP_FIXES incl. pots)')
                p = ub
            m.K_sto_p[n].setlb(p); m.K_sto_p[n].setub(p)
        r = oracle.solve_cost_optimal()
        if r is None:
            raise SystemExit(f'{sc}: benchmark solve failed')
        rec = {'scenario': sc, 'case': 'TYNDP_batteries', 'sys_cost_bn': r['sys_cost'] / 1e9,
               'premium_bn': (r['sys_cost'] - base['sys_cost']) / 1e9,
               'vf_min': oracle.compute_vf_min(r), 'vf_genw': r.get('vf_genw'),
               'cr_min': oracle.compute_cr_min(r)}
        rec.update({f'sto_{n.lower()}_mw': P[n] for n in D['n']})
        rows.append(rec)
        print(f"  [bench] {sc}: B_opt VF_min {rows[-2]['vf_min']:.4f} -> TYNDP batteries "
              f"VF_min {rec['vf_min']:.4f}, premium {rec['premium_bn']:.3f} bn")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                                f"tyndp_battery_benchmark_{os.environ.get('TYNDP_FIXES','none').replace(',','-')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False); print('written', out)

if __name__ == '__main__':
    main()
