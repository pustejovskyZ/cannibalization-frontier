"""
node_factorial.py -- node-level and on-frontier interaction experiments (S.91).

Evaluates the inner LP on a small factorial over CHOSEN outer dimensions while
holding every other dimension at a BASE portfolio:
  --base bopt            base = the cost-optimal portfolio B_opt (solved here)
  --base-csv F --base-row i   base = row i of a frontier checkpoint/final CSV
Sweeps are given as  --sweep "dim=v1,v2,..."  in MW (absolute), or with
--relative as deltas added to the base value; values are clipped to [0, ub]
and the ACTUAL value used is recorded. Resumable: rows append after every
solve; done combos are skipped on restart.

Examples (de2040 production config):
  python node_factorial.py --scenario de2040 --climate-year 2009 --demand-2040
     --h2-2040 --ev-2040 --p2g-endogenous --n-weeks 12 --base bopt
     --sweep "sto_nl_mw=0,5000,10000,20000,30000,40000"
     --sweep "ntc_de_nl_mw=0,1500,3000" --sweep "ntc_dk_nl_mw=0,2000"
     --out output/node_factorial_de2040_bopt.csv
"""
import argparse, itertools, os, time
import pandas as pd

from data_pipeline import load_representative_weeks
from scenarios import apply_scenario
from directed_search import LPOracle


def _parse_sweep(txt):
    dim, vals = txt.split('=')
    return dim.strip(), [float(v) for v in vals.split(',') if v.strip() != '']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--single-node', action='store_true')
    ap.add_argument('--nodes', type=str, default=None)
    ap.add_argument('--scenario', type=str, default='tyndp_de_2040')
    ap.add_argument('--n-weeks', type=int, default=6)
    ap.add_argument('--dk-h2-twh', type=float, default=5.0)
    ap.add_argument('--climate-year', type=int, default=None)
    ap.add_argument('--demand-2040', action='store_true')
    ap.add_argument('--h2-2040', action='store_true')
    ap.add_argument('--ev-2040', action='store_true')
    ap.add_argument('--p2g-endogenous', action='store_true')
    ap.add_argument('--base', type=str, default='bopt', choices=['bopt', 'csv'])
    ap.add_argument('--base-csv', type=str, default=None)
    ap.add_argument('--base-row', type=int, default=0)
    ap.add_argument('--sweep', action='append', required=True,
                    help='dim=v1,v2,... (repeatable)')
    ap.add_argument('--relative', action='store_true',
                    help='sweep values are deltas added to the base value')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    nodes = ['DE'] if args.single_node else (
        [n.strip() for n in args.nodes.split(',')] if args.nodes
        else ['DE', 'CZ', 'DK', 'AT', 'NL'])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pecd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tyndp 2040', 'PECD')
    D = load_representative_weeks(
        os.path.join(root, 'time_series.xlsx'), os.path.join(root, 'data_input.xlsx'),
        n_weeks=args.n_weeks, nodes=nodes, dk_h2_twh=args.dk_h2_twh,
        climate_year=args.climate_year,
        pecd_dir=(pecd if args.climate_year is not None else None))
    apply_scenario(D, args.scenario)
    import tyndp_fixes as _tf
    _tf.apply(D, args.scenario)   # S.103 (env TYNDP_FIXES; unset = no change)
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

    oracle = LPOracle(D)
    keys = list(oracle.x_keys)
    print('  dims:', keys)

    # --- reference cost: always B_opt (premiums comparable with the paper) ---
    print('Solving cost-optimal baseline B_opt...')
    bopt = oracle.solve_cost_optimal()
    if bopt is None:
        raise SystemExit('B_opt solve failed')
    base_cost = bopt['sys_cost']
    print(f"  B_opt cost {base_cost/1e9:.3f} bn | VF_min {oracle.compute_vf_min(bopt):.4f}")

    # --- base portfolio ---
    if args.base == 'bopt':
        x0 = [float(bopt.get(k, 0.0) or 0.0) for k in keys]
        base_label = 'bopt'
    else:
        row = pd.read_csv(args.base_csv).iloc[args.base_row]
        x0 = [float(row.get(k, 0.0) or 0.0) for k in keys]
        base_label = f"{os.path.basename(args.base_csv)}#row{args.base_row}"
    print('  base portfolio:', base_label)
    for k, v in zip(keys, x0):
        if v > 0:
            print(f'    {k} = {v:,.0f}')

    sweeps = [_parse_sweep(s) for s in args.sweep]
    for dim, _ in sweeps:
        if dim not in keys:
            raise SystemExit(f"unknown dim '{dim}'; available: {keys}")
    idx = {k: i for i, k in enumerate(keys)}
    combos = list(itertools.product(*[v for _, v in sweeps]))
    print(f'  {len(combos)} combos over {[d for d, _ in sweeps]}')

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        prev = pd.read_csv(out_path)
        for _, r in prev.iterrows():
            done.add(tuple(round(float(r[f'req_{d}']), 3) for d, _ in sweeps))
        print(f'  resume: {len(done)} combos already done')

    t0 = time.time()
    for k, combo in enumerate(combos):
        key = tuple(round(float(v), 3) for v in combo)
        if key in done:
            continue
        x = list(x0)
        actual = {}
        for (dim, _), v in zip(sweeps, combo):
            i = idx[dim]
            val = x0[i] + v if args.relative else v
            val = min(max(val, 0.0), oracle.bounds[i])
            x[i] = val
            actual[dim] = val
        r = oracle(x)
        row = {'base': base_label}
        for (dim, _), v in zip(sweeps, combo):
            row[f'req_{dim}'] = v
            row[f'act_{dim}'] = actual[dim]
        if r is None:
            row.update({'status': 'LP_FAIL'})
        else:
            row.update({
                'status': 'ok',
                'vf_min': oracle.compute_vf_min(r),
                'vf_genw': r.get('vf_genw'),
                'cr_min': oracle.compute_cr_min(r),
                'sys_cost': r['sys_cost'],
                'premium_bn': (r['sys_cost'] - base_cost) / 1e9,
                'cannib_loss_bn': r.get('cannib_loss_eur', 0) / 1e9,
            })
            for kk in keys:
                row[kk] = r.get(kk)
            # per-pair value factors (needed to track the BINDING pair)
            for kk, vv in r.items():
                if isinstance(kk, str) and kk.startswith('vf_') and kk not in ('vf_min', 'vf_genw'):
                    row[kk] = vv
        pd.DataFrame([row]).to_csv(
            out_path, mode='a', index=False,
            header=not os.path.exists(out_path) or os.path.getsize(out_path) == 0)
        el = time.time() - t0
        vf = row.get('vf_min')
        print(f"  [{k+1}/{len(combos)}] " + ' '.join(f"{d}={actual[d]:,.0f}" for d, _ in sweeps)
              + (f" -> VF_min={vf:.4f} prem={row['premium_bn']:+.3f} bn" if vf is not None else '  LP_FAIL')
              + f"  ({el/60:.1f} min)")
    print('Done ->', out_path)


if __name__ == '__main__':
    main()
