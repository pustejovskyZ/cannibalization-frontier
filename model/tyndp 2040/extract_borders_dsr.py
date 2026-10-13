"""S.106: TYNDP hourly neighbour prices for the residual borders, per-scenario
DSR price bands, and oil plants. Inputs: hourly marginal costs / DSR extracted
from the MMStandardOutputFile 'Hourly Market Data(emarket)' sheets (8736 h,
CY2009), TYNDP 2030 reference grid NTCs as weights, 'Yearly Outputs' for oil.
DE2040 hourly axis is offset +24 h vs PECD CY2009 (verified: wind-price corr
peaks at lag 24 for 2040, at lag 0 for NT2030/2035/2050) -> shifted here so all
stored series are aligned with the model's weather."""
import json, sys, numpy as np
from pyxlsb import open_workbook
import openpyxl
import os
# intermediate hourly files written by extract_hourly_*.py (see extract_all.py)
TMP = os.environ.get('TYNDP_TMP') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '_hourly')
MC = {k: os.path.join(TMP, v) for k, v in {'nt2030': 'mcnt.json', 'de2035': 'mc2035.json', 'de2040': 'mc2040.json', 'de2050': 'mc2050.json'}.items()}
DSRF = {k: os.path.join(TMP, v) for k, v in {'nt2030': 'mcnt.json', 'de2035': 'hmd2035.json', 'de2040': 'hmd2040.json', 'de2050': 'hmd2050.json'}.items()}
SHIFT = {'nt2030': 0, 'de2035': 0, 'de2040': 24, 'de2050': 0}
# neighbour weights = TYNDP 2030 reference-grid NTC (mean of both directions, MW)
W = {'DE': {'FR00': 4800, 'CH00': 4300, 'LUG1': 2400, 'PL00': 2500, 'NOS0': 1400, 'UK00': 1400, 'SE04': 1315, 'BE00': 1000},
     'DK': {'NOS0': 1632, 'UK00': 1400, 'SE04': 1500, 'SE03': 715},
     'AT': {'CH00': 1200, 'ITN1': 785},
     'NL': {'BE00': 3400, 'UK00': 1000, 'NOS0': 700}}

def load_mc(sc):
    J = json.load(open(MC[sc]))
    if 'names' in J:
        names = J['names']; rows = J['data']
        cols = {n: i for i, n in enumerate(names)}
    else:
        cols = {a: i for i, (a, b) in enumerate(J['cols']) if b.startswith('Marginal')}
        rows = J['data']
    A = np.array([[x if isinstance(x, (int, float)) else np.nan for x in r] for r in rows[:8736]], float)
    return {n: np.nan_to_num(A[:, i], nan=np.nanmedian(A[:, i])) for n, i in cols.items()}

def dsr_bands(sc):
    J = json.load(open(DSRF[sc])); cols = J['cols']
    A = np.array([[x if isinstance(x, (int, float)) else np.nan for x in r] for r in J['data'][:8736]], float)
    jd = [i for i, (a, b) in enumerate(cols) if a == 'DE00' and b.startswith('Demand Side Response') and 'Implicit' not in b][0]
    jp = [i for i, (a, b) in enumerate(cols) if a == 'DE00' and b.startswith('Marginal')][0]
    d, p = np.nan_to_num(A[:, jd]), np.nan_to_num(A[:, jp])
    cap = d.max()
    lo = d[(p >= 50) & (p < 70)].mean() / cap if ((p >= 50) & (p < 70)).any() else 0.09
    mid = d[(p >= 90) & (p < 130)].mean() / cap if ((p >= 90) & (p < 130)).any() else 0.31
    lo = min(max(lo, 0.0), 1.0); mid = min(max(mid, lo), 1.0)
    return {'cap_dispatch_max': float(cap), 'shares': [round(lo, 3), round(mid - lo, 3), round(1 - mid, 3)],
            'prices': [60.0, 110.0, 300.0], 'tyndp_twh': float(d.sum() / 1e6)}

def oil_mw(path):
    if path.endswith('xlsb'):
        with open_workbook(path) as wb, wb.get_sheet('Yearly Outputs') as s: rows = [[c.v for c in r] for r in s.rows()]
    else:
        rows = [list(r) for r in openpyxl.load_workbook(path, read_only=True, data_only=True)['Yearly Outputs'].iter_rows(values_only=True)]
    hi = [i for i, r in enumerate(rows) if r and r[0] == 'Output type'][0]; hdr = rows[hi]
    out = {n: 0.0 for n in ('DE', 'CZ', 'DK', 'AT', 'NL')}; sec = None
    for r in rows[hi + 1:]:
        if r[0]: sec = r[0]
        if sec and 'Installed' in sec and r[1] and str(r[1]).strip() in ('Light oil', 'Heavy oil old 1', 'Heavy oil old 2', 'Heavy oil new', 'Oil shale old', 'Oil shale new'):
            for j, h in enumerate(hdr):
                if isinstance(h, str) and h[:2] in out and 'EV ' not in h and isinstance(r[j], (int, float)):
                    out[h[:2]] += r[j]
    return out

MAIN = {'de2035': 'DE2035CY2009/MMStandardOutputFile_DE2035_Plexos_CY2009_v11_SoS.xlsb',
        'de2040': 'MMStandardOutputFile_DE2040_Plexos_CY2009_v11_SoS.xlsb',
        'de2050': 'DE2050CY2009/MMStandardOutputFile_DE2050_Plexos_CY2009_v11_SoS.xlsb',
        'nt2030': 'MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx'}
OILP = {'nt2030': 13.78, 'de2035': 0.5 * (10.09 + 9.61), 'de2040': 9.61, 'de2050': 9.12}   # light oil EUR/GJ
res = {}
base = sys.argv[1]
for sc in MC:
    mc = load_mc(sc); sh = SHIFT[sc]; res[sc] = {'borders': {}, 'shift_h': sh}
    for b, ws in W.items():
        tot = sum(ws.values())
        s = sum(ws[n] / tot * mc[n] for n in ws)
        s = np.roll(s, -sh)                       # aligned: model hour t <- TYNDP hour t+sh
        res[sc]['borders'][b] = {'weights': ws, 'price': [round(float(v), 2) for v in s],
                                 'mean': round(float(s.mean()), 2)}
    res[sc]['dsr'] = dsr_bands(sc)
    res[sc]['oil_mw'] = oil_mw(base + '/' + MAIN[sc])
    res[sc]['oil_price_eur_gj'] = OILP[sc]
    print(sc, {b: v['mean'] for b, v in res[sc]['borders'].items()}, res[sc]['dsr']['shares'], round(res[sc]['dsr']['tyndp_twh'], 1), 'TWh', {k: round(v) for k, v in res[sc]['oil_mw'].items()})
json.dump(res, open(sys.argv[2], 'w'))
