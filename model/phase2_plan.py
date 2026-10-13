"""
phase2_plan.py -- derive the phase-2 experiments from the phase-1 frontiers.

Every choice that depends on the phase-1 results is made here by a fixed rule,
so the phase-2 runs and the paper numbers use the same definitions and no
value is picked by hand:

* elbow of a frontier: Kneedle rule (Satopaa et al. 2011) on the attained
  points including B_opt. Both axes are scaled to [0, 1] between the first and
  the last attained point; the elbow is the interior point with the largest
  gap between the chord and the curve (x - y). On the first campaign this
  gives the 0.738 point that the paper called the elbow.
* weak node W of a scenario: node of the binding (lowest) value factor at the
  highest value factor the frontier reaches (the ceiling point).
* ceiling: highest VF_min reached by any frontier row, attained or not.
* probe targets: ceiling + 0.02 and ceiling + 0.06, rounded up to 0.005.
* main link of W: its link to DE (the largest link otherwise), second link:
  the other link that touches W.
* sweep values: fixed fractions of the deployment potentials, as in the first
  campaign (storage 0, 1/8, 1/4, 1/2, 3/4, 1; main link 0, 1/2, 1; second link
  0, 1; far side DE storage 0, 20, 40, 80 GW x main link 0, 1/3, 2/3, 1).

    python phase2_plan.py            -> prints the plan, writes logs/phase2_plan.json
"""
import os, json, math
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
NODES = ['DE', 'CZ', 'DK', 'AT', 'NL']
TECH = ['pv', 'wind_on', 'wind_off']
SUFFIX = 'tfall'
TOL = 0.003                      # attained = within 0.003 of the target (paper rule)

# outer-search bounds of the production configuration (TYNDP_FIXES=all),
# printed by LPOracle: storage power potentials [MW] and NTC expansion [MW]
STO_POT = {'DE': 200000, 'CZ': 15000, 'DK': 40000, 'AT': 30000, 'NL': 40000}
STO_POT_2050 = dict(STO_POT, NL=50000)          # 'pots' token: NL 2050 -> 50 GW
LINKS = {('DE', 'CZ'): 3000, ('DE', 'DK'): 5000, ('DE', 'AT'): 4000,
         ('DE', 'NL'): 3000, ('DK', 'NL'): 2000, ('CZ', 'AT'): 1500}
STO_KEY = {'DE': 'sto_mw', 'CZ': 'sto_cz_mw', 'DK': 'sto_dk_mw', 'AT': 'sto_at_mw', 'NL': 'sto_nl_mw'}


def frontier_path(sc, tg='vf', suffix=SUFFIX, base=OUT):
    return os.path.join(base, f'directed_search_frontier_5node_{tg}_{sc}_cy2009_dhep_w12_{suffix}.csv')


def attained(d, m='vf_min'):
    """indices of interior rows that reached their target"""
    return [i for i, r in d.iterrows() if i > 0 and r[m] >= r['vf_target'] - TOL]


def kneedle(d, m='vf_min'):
    """row index of the elbow (None if fewer than two attained points)"""
    idx = [d.index[0]] + attained(d, m)
    if len(idx) < 3:
        return idx[-1] if len(idx) > 1 else None
    x = d.loc[idx, m].astype(float).tolist(); y = d.loc[idx, 'premium_bn'].astype(float).tolist()
    xn = [(v - x[0]) / (x[-1] - x[0]) for v in x]; yn = [(v - y[0]) / (y[-1] - y[0]) for v in y]
    gaps = [(xn[k] - yn[k], idx[k]) for k in range(1, len(idx) - 1)]
    return max(gaps)[1]


def binding(r):
    best = None
    for n in NODES:
        for t in TECH:
            v = r.get(f'vf_{t}_{n}', 0)
            if v and v > 0 and (best is None or v < best[0]):
                best = (float(v), t, n)
    return best


def up005(v):
    return math.ceil(round(v / 0.005, 6)) * 0.005


def links_of(w):
    ls = [l for l in LINKS if w in l]
    main = ('DE', w) if ('DE', w) in LINKS else max(ls, key=lambda l: LINKS[l])
    second = [l for l in ls if l != main]
    second = max(second, key=lambda l: LINKS[l]) if second else None
    return main, second


def ntc_key(l):
    return f'ntc_{l[0].lower()}_{l[1].lower()}_mw'


def r100(v):
    return float(round(v / 100.0) * 100)


def local_deltas(x, d, ub):
    """3-point design around x with step d, one-sided near the bounds"""
    if x + d > ub + 1e-6:
        return [-2 * d, -d, 0.0]
    if x - d < -1e-6:
        return [0.0, d, 2 * d]
    return [-d, 0.0, d]


def scenario_info(sc, base=OUT, suffix=SUFFIX):
    d = pd.read_csv(frontier_path(sc, 'vf', suffix, base))
    hi = attained(d)
    top = d['vf_min'].astype(float).idxmax()
    ceil = float(d.loc[top, 'vf_min'])
    w = binding(d.loc[top])[2]
    el = kneedle(d)
    return d, {'scenario': sc, 'ceiling': ceil, 'weak_node': w,
               'weak_tech': binding(d.loc[top])[1],
               'elbow_row': None if el is None else int(el),
               'elbow_target': None if el is None else round(float(d.loc[el, 'vf_target']), 4),
               'elbow_vf': None if el is None else float(d.loc[el, 'vf_min']),
               'first_row': int(hi[0]) if hi else None, 'last_row': int(hi[-1]) if hi else None,
               'probe_targets': [round(up005(ceil + 0.02), 3), round(up005(ceil + 0.06), 3)]}


def make_plan(base=OUT, suffix=SUFFIX):
    d40, s40 = scenario_info('de2040', base, suffix)
    d50, s50 = scenario_info('de2050', base, suffix)
    w = s40['weak_node']
    main, second = links_of(w)
    pot = STO_POT[w]
    plan = {'de2040': s40, 'de2050': s50, 'weak_node_de2040': w,
            'main_link': list(main), 'second_link': list(second) if second else None,
            'sto_key': STO_KEY[w], 'main_ntc_key': ntc_key(main),
            'second_ntc_key': ntc_key(second) if second else None}
    # lens 2: weak-node storage x main link x second link, from B_opt
    plan['lens2'] = {'sto': [r100(pot * f) for f in (0, 0.125, 0.25, 0.5, 0.75, 1.0)],
                     'main': [r100(LINKS[main] * f) for f in (0, 0.5, 1.0)],
                     'second': [r100(LINKS[second] * f) for f in (0, 1.0)] if second else []}
    # lens 3: local 3x3 designs at first attained point, elbow, last attained point
    loc = []
    for row in (s40['first_row'], s40['elbow_row'], s40['last_row']):
        if row is None:
            continue
        r = d40.loc[row]
        xs = float(r.get(STO_KEY[w], 0.0) or 0.0); xl = float(r.get(ntc_key(main), 0.0) or 0.0)
        loc.append({'row': int(row),
                    'sto': local_deltas(xs, r100(pot / 8.0), pot),
                    'main': local_deltas(xl, r100(LINKS[main] / 3.0), LINKS[main])})
    plan['lens3'] = sorted({x['row']: x for x in loc}.values(), key=lambda z: z['row'])
    # far side: DE storage x main link (W != DE), at the elbow and at B_opt
    plan['farside'] = None if w == 'DE' else {
        'sto': [0.0, 20000.0, 40000.0, 80000.0],
        'main': [r100(LINKS[main] * f) for f in (0, 1 / 3, 2 / 3, 1.0)],
        'rows': [s40['elbow_row'], 'bopt']}
    # ceiling probes: weak-node storage potential x2 (de2040, de2050) and 8-h storage (de2040)
    probes = []
    for sc, s in (('de2040', s40), ('de2050', s50)):
        wn = s['weak_node']; p = (STO_POT_2050 if sc == 'de2050' else STO_POT)[wn]
        for k, t in enumerate(s['probe_targets']):
            probes.append({'scenario': sc, 'variant': f'{wn} storage x2', 'target': t,
                           'env': {'CAP_OVERRIDES': f'{wn}.k_bar_sto_p={2*p:.0f};{wn}.k_bar_sto_e={8*p:.0f}'},
                           'tag': f'w2x_{sc}_{"ab"[k]}_{suffix}'})
    for k, t in enumerate(s40['probe_targets']):
        probes.append({'scenario': 'de2040', 'variant': '8-h storage', 'target': t,
                       'env': {'STO_EP_RATIO': '8'}, 'tag': f'ep8_de2040_{"ab"[k]}_{suffix}'})
    plan['probes'] = probes
    return plan


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else OUT
    p = make_plan(base)
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    if len(sys.argv) <= 1:
        json.dump(p, open(os.path.join(HERE, 'logs', 'phase2_plan.json'), 'w'), indent=1)
    print(json.dumps(p, indent=1))
