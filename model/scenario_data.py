"""
scenario_data.py -- per-year TYNDP 2024 Distributed Energy scenario bundles
(S.73 data-population). Centralises the PRIMARY-SOURCE-extracted, internally
consistent parameters for de2035 / de2040 / de2050 so apply_scenario() can set
a whole year at once. All VRE includes the RETE distributed-PV sub-node (the
S.73 fix: prior code dropped ~half the PV). Costs are year-specific DEA
(annualised, WACC 6%, one consistent edition). Sources: TYNDP output .xlsb
(extracted_tyndp_scenarios.json), EV-Modelling-Inputs (ev_by_year.json),
Commodity Prices Final.xlsx (CO2), DEA Technology Catalogue 2030/2040/2050.
"""
import json, os

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tyndp 2040')
_VRE = json.load(open(os.path.join(_DIR, 'extracted_tyndp_scenarios.json')))
_EV = json.load(open(os.path.join(_DIR, 'ev_by_year.json')))

WACC = 0.06
def _crf(n):
    r = WACC
    return r * (1 + r) ** n / ((1 + r) ** n - 1)
_LIFE = {'pv': 30, 'wind_on': 27, 'wind_off': 27, 'ccgt': 25, 'ocgt': 25,
         'sto_p': 15, 'sto_e': 20, 'sc': 25}

# DEA overnight CAPEX by year. PV/onshore/offshore/CCGT/OCGT EUR/kW (EUR2020);
# battery EUR/kWh(energy) & EUR/kW(power) EUR2015 -> x1.06 to EUR2020;
# electrolyser AEC EUR/kW_e (EUR2020). 2035 = linear 2030-2040 midpoint.
_OVN = {  # [2030, 2040, 2050]
    'pv':       [380, 320, 290],
    'wind_on':  [1147, 1110, 1090],
    'wind_off': [2393, 2141, 1986],
    'ccgt':     [883, 867, 851],
    'ocgt':     [468, 452, 436],
    'sto_e':    [62*1.06, 44*1.06, 35*1.06],   # EUR/kWh -> EUR/MWh after *1000
    'sto_p':    [160*1.06, 100*1.06, 60*1.06],  # EUR/kW
    'sc':       [550, 425, 300],
}
_FOM = {  # EUR/MW/yr by [2030,2040,2050]; electrolyser = 4%/yr of overnight
    'pv':       [9500, 8100, 7400],
    'wind_on':  [16663, 15965, 15602],
    'wind_off': [34000, 30000, 28000],
    'ccgt':     [29562, 28605, 27648],
    'ocgt':     [8236, 8065, 7893],
    'sto_p':    [572, 572, 572], 'sto_e': [0, 0, 0],
    'sc':       [22000, 17000, 12000],
}
_YR_IDX = {'2030': 0, '2040': 1, '2050': 2}

def _ovn(tech, year):
    if year == '2035':
        return 0.5 * (_OVN[tech][0] + _OVN[tech][1])
    return _OVN[tech][_YR_IDX[year]]
def _fom(tech, year):
    if year == '2035':
        return 0.5 * (_FOM[tech][0] + _FOM[tech][1])
    return _FOM[tech][_YR_IDX[year]]

def _costs(year):
    """Annualised CAPEX [EUR/MW/yr] + FOM, year-specific (DEA, CRF WACC 6%)."""
    cap = {}
    for t in ('pv', 'wind_on', 'wind_off', 'ccgt', 'ocgt'):
        cap[t] = _ovn(t, year) * 1000 * _crf(_LIFE[t])
    cap['lig'] = cap['nuc'] = cap['hc'] = 0
    fom = {t: _fom(t, year) for t in ('pv', 'wind_on', 'wind_off', 'ccgt', 'ocgt')}
    fom['lig'] = fom['nuc'] = fom['hc'] = 0
    sto_p = _ovn('sto_p', year) * 1000 * _crf(_LIFE['sto_p'])     # EUR/MW/yr
    sto_e = _ovn('sto_e', year) * 1000 * _crf(_LIFE['sto_e'])     # EUR/MWh/yr
    sc = _ovn('sc', year) * 1000 * _crf(_LIFE['sc'])
    return dict(capex={k: round(v) for k, v in cap.items()},
                fom={k: round(v) for k, v in fom.items()},
                sto_capex_p=round(sto_p), sto_capex_e=round(sto_e),
                sto_fom_p=572, sto_fom_e=0,
                sc_capex=round(sc), sc_fom=round(_fom('sc', year)))

CO2 = {'de2035': 100.5, 'de2040': 123.0, 'de2050': 168.0, 'nt2030': 70.0}
NODES = ('DE', 'CZ', 'DK', 'AT', 'NL')
_NT = None
def _nt2030():
    global _NT
    if _NT is None:
        _NT = json.load(open(os.path.join(_DIR, 'extracted_nt2030.json')))
    return _NT

def get_scenario(name):
    """Return a full override dict for a scenario (de2035/de2040/de2050/nt2030).
    nt2030 = National Trends 2030 near-term reference (different storyline; 2030
    DEA costs, CO2 70, includes residual coal). Disclose the storyline caveat."""
    if name == 'nt2030':
        year = '2030'
        v = _nt2030()
    else:
        key = {'de2035': 'de2035', 'de2040': 'de2040', 'de2050': 'de2050'}.get(name)
        if key is None:
            return None
        year = key[-4:]
        v = _VRE[key]
    ev = _EV[year]
    out = {'tau': CO2[name], 'year': year}
    out.update(_costs(year))
    out['vre'] = {n: {'pv': round(v[n]['pv']), 'wind_on': round(v[n]['wind_on']),
                      'wind_off': round(v[n]['wind_off'])} for n in NODES}
    out['fleet'] = {n: {'lig': round(v[n]['lig']), 'nuc': round(v[n]['nuc']),
                        'hc': round(v[n]['hc'])} for n in NODES}
    out['demand_twh'] = {n: round(v[n]['demand_gwh'] / 1e3, 1) for n in NODES}
    out['electro_twh'] = {n: round(v[n]['electro_gwh'] / 1e3, 1) for n in NODES}
    out['ev'] = {n: dict(K_ev=ev[n]['K_ev'], E=ev[n]['E'], elec=ev[n]['elec'])
                 for n in NODES}
    if name == 'de2050':
        # S.84.1: the DE2050 storyline gives AT 94.3 TWh electrolysis demand,
        # which needs >=10.8 GW P2G even at CF 100% -- above the default
        # k_bar_sc 10 GW (year-invariant potential), making B_opt INFEASIBLE.
        # 2050-only raise to 20 GW = symmetry with DK/NL potentials; implies
        # CF 54% for the quota (vs DK's 69% same year -> conservative). Other
        # scenarios untouched (caps slack >=2x) -> completed runs reproduce.
        out['caps_sc'] = {'AT': 20000}
    return out


# S.100: TYNDP 2024 OFFSHORE HUBS (grid-connected "OHEL" hub nodes), omitted by
# the original country-node extraction. Source: MMStandardOutputFile_<SC>_Plexos_
# CY2009_offshore_v11_SoS.xlsb, sheet 'Hourly Data': installed MW = max hourly
# availability (Gen+Dump, hits round capacities), TWh = annual availability.
# DRES hubs (dedicated to offshore electrolysis, not grid-connected) excluded,
# consistent with country-node H2 quotas. Hub CF ~0.55 (far offshore). Used only
# when env OFFSHORE_HUBS=capacity|energy (directed_search.py); unset = original.
HUBS_MW = {
    'de2035': {'DE': 26600 + 1270, 'DK': 5700,  'NL': 5035},
    'de2040': {'DE': 39900 + 1270, 'DK': 13912, 'NL': 5035},
    'de2050': {'DE': 51300 + 2457, 'DK': 18525, 'NL': 10735},
}
HUBS_TWH = {   # annual available energy (Gen + Dump), CY2009
    'de2035': {'DE': 127.1 + 5.2, 'DK': 29.7, 'NL': 23.7},
    'de2040': {'DE': 190.5 + 5.2, 'DK': 72.3, 'NL': 23.5},
    'de2050': {'DE': 248.5 + 10.1, 'DK': 96.6, 'NL': 50.5},
}
