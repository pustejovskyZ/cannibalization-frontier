"""
scenarios.py -- named year/scenario parameter bundles (progress.md S.40-43).

data_pipeline defaults = TYNDP 2024 Distributed Energy 2040 ('tyndp_de_2040').
apply_scenario(D, name) mutates a loaded D in place. 'legacy' restores the
original pre-TYNDP world (280 GW DE, 2/3-rule deployment, 2025 coal fleet,
2020 CAPEX, tau=80, old DSM/EV/storage/P2G caps, old NTC ceilings, fixed
non-investable residual links, no DK-NL link) so old results reproduce.
Only keys present in D are touched (works single-node and 5-node).
"""

_LEGACY = {
    'tau': 80.0,
    'capex': {'pv': 40000, 'wind_on': 85000, 'wind_off': 160000,
              'ccgt': 55000, 'ocgt': 35000, 'lig': 0, 'nuc': 0, 'hc': 0},
    'sto_capex_p': 15000, 'sto_capex_e': 8000, 'sc_capex': 65000,
    # legacy world had no explicit fixed O&M (it was implicit in the old
    # 2020-vintage capex constants) -> zero it for exact reproducibility
    'fom': {'pv': 0, 'wind_on': 0, 'wind_off': 0,
            'ccgt': 0, 'ocgt': 0, 'lig': 0, 'nuc': 0, 'hc': 0},
    'sto_fom_p': 0, 'sto_fom_e': 0, 'sc_fom': 0,
    'opex_gas': {'ccgt': 45, 'ocgt': 70},
    'dsm_capex': 15000, 'ev_capex': 8000,
    'vre': {
        'DE': {'pv': 100000, 'wind_on': 150000, 'wind_off': 30000},
        'CZ': {'pv': 10000,  'wind_on': 2000,   'wind_off': 0},
        'DK': {'pv': 8000,   'wind_on': 6667,   'wind_off': 10000},
        'AT': {'pv': 10000,  'wind_on': 5333,   'wind_off': 0},
        'NL': {'pv': 13333,  'wind_on': 6667,   'wind_off': 10000},
    },
    'fleet': {
        'DE': {'lig': 9300, 'nuc': 0,    'hc': 9800},
        'CZ': {'lig': 4760, 'nuc': 4055, 'hc': 251},
        'DK': {'lig': 0,    'nuc': 0,    'hc': 410},
        'AT': {'lig': 0,    'nuc': 0,    'hc': 0},
        'NL': {'lig': 0,    'nuc': 486,  'hc': 4608},
    },
    # old caps [MW]: DSM/EV + storage power/energy + P2G
    'caps': {
        'DE': {'k_bar_dsm': 8000, 'k_bar_ev': 20000},
        'CZ': {'k_bar_dsm': 1000, 'k_bar_ev': 3000,
               'k_bar_sto_p': 5000, 'k_bar_sto_e': 20000, 'k_bar_sc': 5000},
        'DK': {'k_bar_dsm': 1000, 'k_bar_ev': 3000,
               'k_bar_sto_p': 12000, 'k_bar_sto_e': 48000, 'k_bar_sc': 6000},
        'AT': {'k_bar_dsm': 800,  'k_bar_ev': 2000,
               'k_bar_sto_p': 10000, 'k_bar_sto_e': 40000, 'k_bar_sc': 3000},
        'NL': {'k_bar_dsm': 1500, 'k_bar_ev': 5000,
               'k_bar_sto_p': 10000, 'k_bar_sto_e': 40000, 'k_bar_sc': 5000},
    },
    'caps_top': {'k_bar_dsm': 8000, 'k_bar_ev': 20000},
    'ntc_fixed': {'DE_CZ': 2300, 'DE_DK': 3593, 'DE_AT': 7500,
                  'DE_NL': 5000, 'CZ_AT': 1100, 'DK_NL': 0},
    'ntc_max': {'DE_CZ': 3000, 'DE_DK': 7000, 'DE_AT': 10000,
                'DE_NL': 8000, 'CZ_AT': 2000, 'DK_NL': 0},
    'resid_ntc': {'DK': 4800, 'AT': 2650, 'NL': 3400},
}

KNOWN = ('tyndp_de_2040', 'de2035', 'de2040', 'de2050', 'nt2030', 'legacy')
_ALIAS = {'tyndp_de_2040': 'de2040'}


def _apply_year(D, s):
    """Apply a per-year TYNDP bundle (scenario_data.get_scenario) onto D:
    corrected VRE (incl RETE PV), fleet, year-specific DEA costs, CO2, and
    stash demand / electrolyser / EV targets for the build helpers (S.73)."""
    D['tau'] = s['tau']
    if 'capex' in D: D['capex'].update(s['capex'])
    if 'fom' in D: D['fom'].update(s['fom'])
    for k in ('sto_capex_p', 'sto_capex_e', 'sto_fom_p', 'sto_fom_e',
              'sc_capex', 'sc_fom'):
        if k in s: D[k] = s[k]
    # DE global VRE/fleet + per-node
    D['k_bar_vre'] = dict(s['vre']['DE'])
    D['k_exist_inf'] = dict(s['fleet']['DE'])
    for n, npar in D.get('node_params', {}).items():
        if n in s['vre']:
            npar['k_bar_vre'] = dict(s['vre'][n])
        if n in s['fleet']:
            npar['k_exist_inf'] = dict(s['fleet'][n])
    # stash per-scenario targets for scale_demand_to_targets / build_ev / H2
    D['demand_targets'] = dict(s['demand_twh'])
    D['h2_elec_twh'] = dict(s['electro_twh'])
    D['ev_params'] = {n: dict(v) for n, v in s['ev'].items()}
    D['_scen_year'] = s['year']
    for n, cap in s.get('caps_sc', {}).items():   # S.84.1 (de2050 AT P2G)
        npar = D.get('node_params', {}).get(n)
        if npar is not None:
            print(f"  [scenario] {n} k_bar_sc {npar.get('k_bar_sc')} -> "
                  f"{cap:,.0f} MW (year-{s['year']} electrolyser potential)")
            npar['k_bar_sc'] = float(cap)
    print(f"  [scenario] Distributed Energy {s['year']} applied (corrected VRE "
          f"incl RETE; DE PV={s['vre']['DE']['pv']/1000:.0f} GW; tau={s['tau']}; "
          f"year-specific DEA costs)")
    return D


def apply_scenario(D, name='tyndp_de_2040'):
    yr = _ALIAS.get(name, name)
    if yr in ('de2035', 'de2040', 'de2050', 'nt2030'):
        from scenario_data import get_scenario
        return _apply_year(D, get_scenario(yr))
    if name != 'legacy':
        raise ValueError(f"unknown scenario '{name}'; known: {KNOWN} "
                         f"(tyndp_nt_2030 not yet populated)")
    b = _LEGACY
    D['tau'] = b['tau']
    if 'capex' in D:
        D['capex'].update(b['capex'])
    for k in ('sto_capex_p', 'sto_capex_e', 'sc_capex', 'dsm_capex', 'ev_capex',
              'sto_fom_p', 'sto_fom_e', 'sc_fom'):
        if k in D and k in b:
            D[k] = b[k]
    if 'fom' in D:
        D['fom'].update(b['fom'])
    if 'opex' in D:
        D['opex'].update(b['opex_gas'])
    if 'k_bar_vre' in D:
        D['k_bar_vre'] = dict(b['vre']['DE'])
    if 'k_exist_inf' in D:
        D['k_exist_inf'] = dict(b['fleet']['DE'])
    for k, v in b['caps_top'].items():
        if k in D:
            D[k] = v
    for n, npar in D.get('node_params', {}).items():
        if n in b['vre']:
            npar['k_bar_vre'] = dict(b['vre'][n])
        if n in b['fleet']:
            npar['k_exist_inf'] = dict(b['fleet'][n])
        if n in b['caps']:
            npar.update(b['caps'][n])
    for lname, link in D.get('links', {}).items():
        if lname in b['ntc_fixed']:
            link['ntc_fixed'] = b['ntc_fixed'][lname]
        if lname in b['ntc_max']:
            link['ntc_max'] = b['ntc_max'][lname]
    # legacy: DE had no residual interconnection (isolated single-node / 4-link
    # 5-node) -> drop it so legacy reproduces pre-residual results.
    D.get('resid_intercon', {}).pop('DE', None)
    # legacy: residual links were fixed (non-investable) -> ntc_max == ntc
    for rn, rd in D.get('resid_intercon', {}).items():
        if rn in b.get('resid_ntc', {}):
            rd['ntc'] = b['resid_ntc'][rn]
        rd['ntc_max'] = rd['ntc']
    print("  [scenario] LEGACY applied (pre-TYNDP: 280 GW DE, 2025 fleet, "
          "tau=80, 2020 CAPEX, old caps, fixed residual links, no DE residual)")
    return D
