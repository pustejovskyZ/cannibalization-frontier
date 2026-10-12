"""
grid_floor.py -- certified reduced-space frontier floor (progress.md S.56).

Evaluates the inner LP on a FULL FACTORIAL grid over 4 aggregate
flexibility scales (per-instrument fractions of each dimension's upper
bound) and extracts the least-cost envelope vs VF_min. Because the grid is
exhaustively enumerated (no heuristic search), its envelope is a CERTIFIED
upper bound on the true frontier *within the reduced space*:

    every point of the envelope is feasible and exactly costed; the
    full-dimensional CMA-ES frontier must be AT OR BELOW it wherever both
    are defined. If CMA-ES returns a costlier point at the same VF, the
    26-D search has demonstrably not converged (referee-facing evidence).

The reduced space scales each instrument class uniformly across nodes:
    x_i = f_class * ub_i   for i in {sto, sc, dsm+ev, ntc} dims.

Resumable: rows are appended to the output CSV after EVERY solve and
already-done combos are skipped on restart (sandbox/PC crash safe).

Usage (user's Gurobi machine; ~100 solves by default):
    python grid_floor.py                          # 5-node default
    python grid_floor.py --single-node
    python grid_floor.py --fracs-sto 0,0.25,0.5,0.75,1 --fracs-ntc 0,1
"""

import argparse
import itertools
import os
import time

import pandas as pd

from data_pipeline import load_representative_weeks
from scenarios import apply_scenario
from directed_search import LPOracle


def _parse_fracs(txt):
    return [float(v) for v in txt.split(',') if v.strip() != '']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--single-node', action='store_true')
    ap.add_argument('--nodes', type=str, default=None)
    ap.add_argument('--scenario', type=str, default='tyndp_de_2040')
    ap.add_argument('--n-weeks', type=int, default=6)
    ap.add_argument('--dk-h2-twh', type=float, default=5.0,
                    help='DK H2 mandate [TWh] -- MUST match the frontier run '
                         '(directed_search default 5.0) so B_opt and the '
                         'premium baseline are identical. (S.57b: grid_floor '
                         'previously used the data_pipeline default 0.0, '
                         'giving a different, easier B_opt.)')
    ap.add_argument('--fracs-sto', type=str,
                    default='0,0.1,0.2,0.35,0.5,0.7,0.85,1.0')
    ap.add_argument('--fracs-sc', type=str, default='0,0.33,0.66,1.0')
    ap.add_argument('--fracs-ntc', type=str, default='0,0.5,1.0')
    ap.add_argument('--fracs-dsmev', type=str, default='1.0',
                    help='DSM+EV scale (they saturate in every production '
                         'run -> default pinned at cap).')
    ap.add_argument('--out', type=str, default=None)
    # ---- S.86: config-match the headline runs (S.83 TODO) ----
    ap.add_argument('--climate-year', type=int, default=None)
    ap.add_argument('--demand-2040', action='store_true')
    ap.add_argument('--h2-2040', action='store_true')
    ap.add_argument('--ev-2040', action='store_true')
    ap.add_argument('--p2g-endogenous', action='store_true')
    args = ap.parse_args()

    if args.single_node:
        nodes = ['DE']
    elif args.nodes:
        nodes = [n.strip() for n in args.nodes.split(',')]
    else:
        nodes = ['DE', 'CZ', 'DK', 'AT', 'NL']

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pecd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tyndp 2040', 'PECD')
    D = load_representative_weeks(
        os.path.join(root, 'time_series.xlsx'),
        os.path.join(root, 'data_input.xlsx'),
        n_weeks=args.n_weeks, nodes=nodes,
        dk_h2_twh=args.dk_h2_twh,
        climate_year=args.climate_year,
        pecd_dir=(_pecd if args.climate_year is not None else None))
    apply_scenario(D, args.scenario)
    import tyndp_fixes as _tf
    _tf.apply(D, args.scenario)   # S.103 (env TYNDP_FIXES; unset = no change)
    # mirror directed_search main's scenario-conditioning (S.86, order matters)
    D['p2g_endogenous'] = bool(args.p2g_endogenous)
    if args.ev_2040:
        from data_pipeline import build_ev_2040
        D['ev_capex'] = 0.0
        build_ev_2040(D)
    if args.demand_2040:
        from data_pipeline import scale_demand_to_targets
        scale_demand_to_targets(D)
    if args.h2_2040:
        eta = D.get('sc_eta', 0.70)
        _EL = D.get('h2_elec_twh')
        if not _EL:
            from data_pipeline import TYNDP_DE2040_ELECTROLYSER_TWH as _EL
        D['h2_targets'] = {n: _EL[n] * 1e6 * eta for n in D['n'] if n in _EL}
        print('  [grid_floor] per-node H2 quotas set (TYNDP x eta)')

    oracle = LPOracle(D)
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output',
        f'grid_floor_{len(nodes)}node.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Baseline B_opt for the premium reference (matches directed_search)
    print('Solving cost-optimal baseline B_opt...')
    base = oracle.solve_cost_optimal()
    if base is None:
        raise SystemExit('B_opt solve failed')
    base_cost = base['sys_cost']
    print(f'  B_opt cost {base_cost/1e9:.3f} bn, '
          f'VF_min {oracle.compute_vf_min(base):.4f}')

    f_sto = _parse_fracs(args.fracs_sto)
    f_sc = _parse_fracs(args.fracs_sc)
    f_ntc = _parse_fracs(args.fracs_ntc)
    f_de = _parse_fracs(args.fracs_dsmev)
    combos = list(itertools.product(f_sto, f_sc, f_de, f_ntc))
    print(f'Grid: {len(f_sto)}x{len(f_sc)}x{len(f_de)}x{len(f_ntc)} '
          f'= {len(combos)} LP solves '
          f'({oracle.n_dims} dims reduced to 4 aggregate scales)')

    done = set()
    if os.path.exists(out_path):
        try:
            df_done = pd.read_csv(out_path)
            done = {(round(r.f_sto, 4), round(r.f_sc, 4),
                     round(r.f_dsmev, 4), round(r.f_ntc, 4))
                    for r in df_done.itertuples()}
            print(f'Resuming: {len(done)} combos already done in {out_path}')
        except Exception:
            pass

    t0 = time.time()
    n_new = 0
    for k, (fs, fp, fd, fn) in enumerate(combos):
        key = (round(fs, 4), round(fp, 4), round(fd, 4), round(fn, 4))
        if key in done:
            continue
        x = []
        for i, info in enumerate(oracle.dim_info):
            ub = oracle.bounds[i]
            if info['type'] == 'ntc':
                x.append(fn * ub)
            elif info['instr'] == 'sto':
                x.append(fs * ub)
            elif info['instr'] == 'sc':
                x.append(fp * ub)
            else:  # dsm / ev
                x.append(fd * ub)
        r = oracle(x)
        n_new += 1
        if r is None:
            row = {'f_sto': fs, 'f_sc': fp, 'f_dsmev': fd, 'f_ntc': fn,
                   'status': 'LP_FAIL', 'vf_min': None, 'vf_genw': None,
                   'cr_min': None, 'sys_cost': None, 'premium_bn': None,
                   'subsidy_bill_bn': None, 'cannib_loss_bn': None}
            for kk in oracle.x_keys:
                row[kk] = None
        else:
            row = {'f_sto': fs, 'f_sc': fp, 'f_dsmev': fd, 'f_ntc': fn,
                   'status': 'ok',
                   'vf_min': oracle.compute_vf_min(r),
                   'vf_genw': r.get('vf_genw'),
                   'cr_min': oracle.compute_cr_min(r),
                   'sys_cost': r['sys_cost'],
                   'premium_bn': (r['sys_cost'] - base_cost) / 1e9,
                   'subsidy_bill_bn': r.get('subsidy_bill_eur', 0) / 1e9,
                   'cannib_loss_bn': r.get('cannib_loss_eur', 0) / 1e9}
            for kk in oracle.x_keys:
                row[kk] = r.get(kk)
        pd.DataFrame([row]).to_csv(
            out_path, mode='a', index=False,
            header=not os.path.exists(out_path) or os.path.getsize(out_path) == 0)
        el = time.time() - t0
        if r is not None:
            print(f'  [{k+1}/{len(combos)}] sto={fs:.2f} sc={fp:.2f} '
                  f'dsmev={fd:.2f} ntc={fn:.2f} -> VF_min='
                  f'{row["vf_min"]:.4f}, premium={row["premium_bn"]:.2f} bn '
                  f'({el/max(n_new,1):.0f} s/solve)', flush=True)
        else:
            print(f'  [{k+1}/{len(combos)}] LP FAILED', flush=True)

    # ---- envelope report ----
    df = pd.read_csv(out_path)
    df = df[df['status'] == 'ok']
    print('\nCERTIFIED REDUCED-SPACE ENVELOPE (least premium s.t. VF_min >= theta):')
    print('  theta   premium_bn   (combo sto/sc/dsmev/ntc)')
    for theta in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                  0.75, 0.80, 0.85, 0.90]:
        feas = df[df['vf_min'] >= theta - 1e-9]
        if len(feas) == 0:
            print(f'  {theta:.2f}    -- unreachable in reduced space --')
            continue
        b = feas.loc[feas['sys_cost'].idxmin()]
        print(f'  {theta:.2f}   {b["premium_bn"]:9.3f}     '
              f'({b["f_sto"]:.2f}/{b["f_sc"]:.2f}/{b["f_dsmev"]:.2f}/{b["f_ntc"]:.2f})')
    print(f'\nWrote {out_path}. Compare: the CMA frontier must be AT OR '
          f'BELOW this envelope at every theta; if not, the 26-D search '
          f'has not converged there.')


if __name__ == '__main__':
    main()
