"""Recreate the three basic TYNDP 2024 input files of the model:

  extracted_tyndp_scenarios.json  DE 2035/2040/2050 per node: installed PV,
                                  onshore, offshore, CCGT, OCGT, conventional
                                  gas, nuclear, lignite, hard coal [MW];
                                  demand and electrolysis load [GWh]
  extracted_nt2030.json           same for NT+ 2030 (gas = CCGT + OCGT, integers)
  ev_by_year.json                 passenger EV fleet per node and year:
                                  K_ev charging power [MW] (units x 7.4 kW),
                                  E battery energy [MWh] (units x kWh),
                                  elec annual charging demand [MWh]

Sources (TYNDP 2024 Scenarios, ENTSO-E and ENTSOG, CC BY 4.0):
  MMStandardOutputFile_<SC>_Plexos_CY2009_*.xlsb/xlsx, sheet 'Yearly Outputs'
  (all sub-nodes of a country: 00, SRES, RETE, EV, and DKE1/DKW1 for Denmark),
  EV Modelling Inputs/EV_VEHICLES.xlsx (Passenger, Distributed Energy).

    python extract_scenarios_ev.py [raw-data folder]     (default: this folder)
    python extract_scenarios_ev.py --check               compare with the JSON files
"""
import os, sys, json, math, collections

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = ('DE', 'CZ', 'DK', 'AT', 'NL')
MAIN = {'de2035': 'DE2035CY2009/MMStandardOutputFile_DE2035_Plexos_CY2009_v11_SoS.xlsb',
        'de2040': 'MMStandardOutputFile_DE2040_Plexos_CY2009_v11_SoS.xlsb',
        'de2050': 'DE2050CY2009/MMStandardOutputFile_DE2050_Plexos_CY2009_v11_SoS.xlsb',
        'nt2030': 'MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx'}
CAP = {'pv': ('Solar (Photovoltaic)',), 'wind_on': ('Wind Onshore',), 'wind_off': ('Wind Offshore',),
       'ccgt': ('Gas CCGT',), 'ocgt': ('Gas OCGT',), 'gasconv': ('Gas conventional',),
       'nuc': ('Nuclear',), 'lig': ('Lignite old', 'Lignite new', 'Lignite CCS'),
       'hc': ('Hard coal old', 'Hard coal new', 'Hard coal CCS')}


def yearly_rows(path):
    if path.endswith('.xlsb'):
        from pyxlsb import open_workbook
        with open_workbook(path) as wb, wb.get_sheet('Yearly Outputs') as s:
            return [[c.v for c in r] for r in s.rows()]
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)['Yearly Outputs']
    return [list(r) for r in ws.iter_rows(values_only=True)]


def country_table(rows):
    """{node: {(section, label): value summed over the node's columns}}"""
    hi = [i for i, r in enumerate(rows) if r and r[0] == 'Output type'][0]
    hdr = rows[hi]
    cols = collections.defaultdict(list)
    for j, h in enumerate(hdr):
        if isinstance(h, str) and h[:2] in NODES:
            cols[h[:2]].append(j)
    out = collections.defaultdict(dict); sec = None
    for r in rows[hi + 1:]:
        if r[0]:
            sec = str(r[0]).strip()
        lab = '' if r[1] is None else str(r[1]).strip()
        for n, js in cols.items():
            out[n][(sec, lab)] = sum(r[j] for j in js if isinstance(r[j], (int, float)))
    return out


def node_values(t, nt=False):
    capsec = 'Installed Capacities [MW]'
    def cap(prefixes):
        return float(sum(v for (s, l), v in t.items() if s == capsec and l.startswith(prefixes)))
    dem = sum(v for (s, l), v in t.items() if s.startswith('Native Demand'))
    el = sum(v for (s, l), v in t.items() if s.startswith('Annual generation') and l == 'Electrolyser (load)')
    v = {k: cap(p) for k, p in CAP.items()}
    v['demand_gwh'] = round(float(dem), 1); v['electro_gwh'] = round(float(el), 1)
    if nt:   # NT+ 2030 file of the first campaign: gas = CCGT + OCGT, integers
        v['gas'] = v.pop('ccgt') + v.pop('ocgt'); v.pop('gasconv')
        v['demand_gwh'], v['electro_gwh'] = float(dem), float(el)          # unrounded
        v = {k: int(round(v[k])) for k in ('pv', 'wind_on', 'wind_off', 'nuc', 'lig', 'hc', 'gas', 'demand_gwh', 'electro_gwh')}
    return v


def ev_fleet(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True)); h = rows[0]; c = {k: i for i, k in enumerate(h)}
    out = {}
    for r in rows[1:]:
        if r[c['EV TYPE']] != 'Passenger' or r[c['SCENARIO']] != 'Distributed Energy':
            continue
        n = str(r[c['NODE']])[:2]; y = str(int(r[c['YEAR']]))
        if n not in NODES or y not in ('2030', '2035', '2040', '2050'):
            continue
        u = float(r[c['UNITS [-]']])
        d = out.setdefault(y, {}).setdefault(n, {'K_ev': 0.0, 'E': 0.0, 'elec': 0.0})
        d['K_ev'] += u * float(r[c['MAX CHARGE RATE [kW]']]) / 1e3
        d['E'] += u * float(r[c['CAPACITY [kWh]']]) / 1e3
        d['elec'] += float(r[c['ELECTRICITY DEMAND [TWh]']]) * 1e6
    return {y: {n: {k: int(round(v)) for k, v in d.items()} for n, d in ys.items()} for y, ys in out.items()}


def build(base):
    sc = {}
    for s, f in MAIN.items():
        t = country_table(yearly_rows(os.path.join(base, f)))
        sc[s] = {n: node_values(t[n], nt=(s == 'nt2030')) for n in NODES}
        print('  read', f)
    ev = ev_fleet(os.path.join(base, 'EV Modelling Inputs', 'EV_VEHICLES.xlsx'))
    return {'extracted_tyndp_scenarios.json': {s: sc[s] for s in ('de2035', 'de2040', 'de2050')},
            'extracted_nt2030.json': sc['nt2030'], 'ev_by_year.json': ev}


def compare(a, b, path=''):
    """list of differences larger than rounding"""
    diffs = []
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                diffs.append((path + '/' + str(k), a.get(k), b.get(k)))
            else:
                diffs += compare(a[k], b[k], path + '/' + str(k))
    elif abs(float(a) - float(b)) > max(0.06, 1e-6 * abs(float(b))):
        diffs.append((path, a, b))
    return diffs


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    new = build(args[0] if args else HERE)
    if '--check' in sys.argv:
        for f, d in new.items():
            df = compare(d, json.load(open(os.path.join(HERE, f))))
            print(f, 'identical' if not df else f'{len(df)} differences: {df[:10]}')
    else:
        for f, d in new.items():
            json.dump(d, open(os.path.join(HERE, f), 'w'), indent=1)
            print('wrote', f)
