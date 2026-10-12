"""S.103: extract TYNDP 2024 inputs missing from the model (offshore hubs,
hydro storage, other generation, H2-CCGT, existing gas, DSR, batteries).
Sources: MMStandardOutputFile_<SC>_Plexos_CY2009_*.xlsb/xlsx (Yearly Outputs;
offshore files Hourly Data). Output: tyndp_extras.json (MW, GWh)."""
import json, collections, sys, os
from pyxlsb import open_workbook
import openpyxl
BASE = sys.argv[1]
MAIN = {'de2035': 'DE2035CY2009/MMStandardOutputFile_DE2035_Plexos_CY2009_v11_SoS.xlsb',
        'de2040': 'MMStandardOutputFile_DE2040_Plexos_CY2009_v11_SoS.xlsb',
        'de2050': 'DE2050CY2009/MMStandardOutputFile_DE2050_Plexos_CY2009_v11_SoS.xlsb',
        'nt2030': 'MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx'}
OFF = {'de2035': 'DE2035CY2009/MMStandardOutputFile_DE2035_Plexos_CY2009_offshore_v11_SoS.xlsb',
       'de2040': 'MMStandardOutputFile_DE2040_Plexos_CY2009_offshore_v11_SoS.xlsb',
       'de2050': 'DE2050CY2009/MMStandardOutputFile_DE2050_Plexos_CY2009_offshore_v11_SoS.xlsb'}
NODES = ('DE', 'CZ', 'DK', 'AT', 'NL')

def yearly(path):
    if path.endswith('.xlsb'):
        with open_workbook(path) as wb, wb.get_sheet('Yearly Outputs') as s:
            rows = [[c.v for c in r] for r in s.rows()]
    else:
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True)['Yearly Outputs']
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    hi = [i for i, r in enumerate(rows) if r and r[0] == 'Output type'][0]
    hdr = rows[hi]
    cols = collections.defaultdict(list)
    for j, h in enumerate(hdr):
        # main country sub-nodes only (00, RETE, SRES); EV sub-nodes excluded;
        # NT2030 offshore sub-nodes (DEKF, DKBH, ...) are already in the
        # country-node extraction and are not needed here.
        if isinstance(h, str) and h[:2] in NODES and 'EV ' not in h:
            cols[h[:2]].append((j, h))
    out = collections.defaultdict(dict); sec = None
    for r in rows[hi + 1:]:
        if r[0]: sec = r[0]
        if not r[1] or not sec: continue
        key = ('cap' if 'Installed' in sec else 'gen' if 'Annual generation' in sec else None)
        if key is None: continue
        for n, js in cols.items():
            v = sum((r[j] or 0) for j, h in js if isinstance(r[j], (int, float)))
            out[n][(key, str(r[1]).strip())] = v
            # utility vs prosumer battery energy split (Installed row = MWh)
            if key == 'cap' and str(r[1]).startswith('Battery Storage discharge'):
                out[n][('bat_util', '')] = sum((r[j] or 0) for j, h in js if isinstance(r[j], (int, float)) and h.endswith('00') or h in ('DKE1', 'DKW1'))
                out[n][('bat_rete', '')] = sum((r[j] or 0) for j, h in js if isinstance(r[j], (int, float)) and 'RETE' in h)
    return out

def g(d, key, lab):
    return float(sum(v for (k, l), v in d.items() if k == key and l.startswith(lab)))

res = {}
for sc, f in MAIN.items():
    Y = yearly(os.path.join(BASE, f)); res[sc] = {}
    for n in NODES:
        d = Y[n]
        res[sc][n] = {
          'hydro': {'res_mw': g(d, 'cap', 'Reservoir'),
                    'open_turb_mw': g(d, 'cap', 'Pump Storage - Open Loop (turbine)'),
                    'open_pump_mw': g(d, 'cap', 'Pump Storage - Open Loop (pump)'),
                    'closed_turb_mw': g(d, 'cap', 'Pump Storage - Closed Loop (turbine)'),
                    'closed_pump_mw': g(d, 'cap', 'Pump Storage - Closed Loop (pump)'),
                    'res_gwh': g(d, 'gen', 'Reservoir'),
                    'open_turb_gwh': g(d, 'gen', 'Pump Storage - Open Loop (turbine)'),
                    'open_pump_gwh': g(d, 'gen', 'Pump Storage - Open Loop (pump)')},
          'other_gwh': {'others_res': g(d, 'gen', 'Others renewable'),
                        'others_nonres': g(d, 'gen', 'Others non-renewable'),
                        'biofuel': g(d, 'gen', 'Hard Coal biofuel') + g(d, 'gen', 'Gas biofuel')},
          'other_mw': g(d, 'cap', 'Others renewable') + g(d, 'cap', 'Others non-renewable')
                      + g(d, 'cap', 'Hard Coal biofuel') + g(d, 'cap', 'Gas biofuel'),
          'h2ccgt_mw': g(d, 'cap', 'Hydrogen CCGT'),
          'dsr_mw': g(d, 'cap', 'Demand Side Response Explicit') if sc == 'nt2030' else g(d, 'cap', 'Demand Side Response'),
          'gas_mw': {'ccgt': g(d, 'cap', 'Gas CCGT') + g(d, 'cap', 'Gas conventional'),
                     'ocgt': g(d, 'cap', 'Gas OCGT')},
          'battery': {'cap_row_total': g(d, 'cap', 'Battery Storage discharge'),
                      'util_row': float(d.get(('bat_util', ''), 0)),
                      'rete_row': float(d.get(('bat_rete', ''), 0))},
        }
    print(sc, 'done')

# offshore hubs: grid-connected OHEL nodes, hourly availability = Gen + Dump
hubs = {}
for sc, f in OFF.items():
    with open_workbook(os.path.join(BASE, f)) as wb, wb.get_sheet('Hourly Data') as s:
        rows = [[c.v for c in r] for r in s.rows()]
    hi = [i for i, r in enumerate(rows) if r and r[0] == 'Hour'][0]; hdr = rows[hi]
    data = [r for r in rows[hi + 1:] if isinstance(r[0], (int, float))]
    hubs[sc] = {}
    for n in ('DE', 'DK', 'NL'):
        av = [0.0] * len(data)
        for hub in sorted({h[:-4] for h in hdr if isinstance(h, str) and h.startswith(n) and 'OHEL_' in h and h.endswith('_Gen')}):
            gi = hdr.index(hub + '_Gen'); di = hdr.index(hub + '_Dump')
            col = [(r[gi] or 0) + (r[di] or 0) for r in data]
            av = [a + b for a, b in zip(av, col)]
        # installed MW = sum over hubs of max availability (hits round capacities)
        mw = 0.0
        for hub in sorted({h[:-4] for h in hdr if isinstance(h, str) and h.startswith(n) and 'OHEL_' in h and h.endswith('_Gen')}):
            gi = hdr.index(hub + '_Gen'); di = hdr.index(hub + '_Dump')
            mw += max((r[gi] or 0) + (r[di] or 0) for r in data)
        prof = [round(a / mw, 5) if mw > 0 else 0.0 for a in av]
        hubs[sc][n] = {'mw': mw, 'twh': sum(av) / 1e6, 'n_hours': len(prof), 'profile': prof}
        print(sc, n, round(mw), 'MW', round(sum(av) / 1e6, 1), 'TWh', 'CF', round(sum(prof) / len(prof), 3))
for sc in res:
    if sc in hubs:
        for n in hubs[sc]: res[sc][n]['hub'] = hubs[sc][n]
# TYNDP 2024 'Final Fuel Prices 2022' sheet, Distributed Energy columns (NT: NT col)
H2_EUR_PER_GJ = {'nt2030': 20.25, 'de2035': 0.5 * (19.9612 + 16.227), 'de2040': 16.227, 'de2050': 12.52}
for sc in res: res[sc]['_h2_price_eur_mwh'] = round(H2_EUR_PER_GJ[sc] * 3.6, 2)
json.dump(res, open(sys.argv[2], 'w'))
print('written', sys.argv[2])
