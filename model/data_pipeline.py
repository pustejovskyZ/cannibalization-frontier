"""
Shared Data Pipeline v12 -- Representative Weeks via k-Medoids
"""

import pandas as pd
import numpy as np


def _parse_excel(ts_path, node='DE'):
    raw = pd.read_excel(ts_path, sheet_name='basic', header=None)
    nodes  = raw.iloc[5, :].values
    labels = raw.iloc[7, :].values
    data   = raw.iloc[8:, :].copy()
    data.columns = range(data.shape[1])
    data = data.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

    def find_col(n, dtype):
        for i in range(1, len(labels)):
            if str(nodes[i]).strip() == n and str(labels[i]).strip() == dtype:
                return i
        return None

    wind_off_col = find_col(node, 'wind_off')
    if wind_off_col is not None:
        wind_off = data[wind_off_col].values[:8760]
        wind_off = np.where(np.isnan(wind_off), 0.0, wind_off)
    else:
        wind_off = np.zeros(8760)

    ror_col = find_col(node, 'ror')
    if ror_col is not None:
        ror = data[ror_col].values[:8760]
        ror = np.where(np.isnan(ror), 0.0, ror)
    else:
        ror = np.zeros(8760)

    rsvr_col = find_col(node, 'rsvr')
    if rsvr_col is not None:
        rsvr = data[rsvr_col].values[:8760]
        rsvr = np.where(np.isnan(rsvr), 0.0, rsvr)
    else:
        rsvr = np.zeros(8760)

    return {
        'demand':   data[find_col(node, 'demand')].values[:8760],
        'pv':       data[find_col(node, 'pv')].values[:8760],
        'wind_on':  data[find_col(node, 'wind_on')].values[:8760],
        'wind_off': wind_off,
        'ror':      ror,
        'rsvr':     rsvr,
    }


def _kmedoids(X, k, max_iter=100, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
    medoids = rng.choice(n, size=k, replace=False)
    for _ in range(max_iter):
        dists_to_medoids = D[:, medoids]
        labels = np.argmin(dists_to_medoids, axis=1)
        improved = False
        for c in range(k):
            members = np.where(labels == c)[0]
            best_cost = sum(D[m, medoids[c]] for m in members)
            best_medoid = medoids[c]
            for candidate in members:
                cost = sum(D[m, candidate] for m in members)
                if cost < best_cost - 1e-10:
                    best_cost = cost
                    best_medoid = candidate
                    improved = True
            medoids[c] = best_medoid
        if not improved:
            break
    labels = np.argmin(D[:, medoids], axis=1)
    inertia = sum(D[i, medoids[labels[i]]] for i in range(n))
    return medoids, labels, inertia


def _load_pecd_cf(pecd_dir, climate_year, nodes, fleet='2030'):
    """Load PECD v3.1 hourly capacity factors (PV / wind onshore / offshore)
    for one CLIMATE YEAR and our nodes, for the multi-weather-year robustness
    (Phase B). PECD parquet is long: [country, year, month, day, hour, cf].
    Returns {node: {'pv','wind_on','wind_off'}} with >=8760-length arrays
    (offshore -> zeros for landlocked AT/CZ). Uses predicate pushdown on
    `year` so we never load the full 12M-row table. Source: ENTSO-E PECD
    v3.1 (Zenodo 5780185, CC-BY); the database TYNDP 2024 uses."""
    import pandas as _pd, numpy as _np, os as _os
    files = {'pv': f'PECD-2021.3-country-LFSolarPV-{fleet}.parquet',
             'wind_on': f'PECD-2021.3-country-Onshore-{fleet}.parquet',
             'wind_off': f'PECD-2021.3-country-Offshore-{fleet}.parquet'}
    out = {n: {} for n in nodes}
    for tech, fn in files.items():
        path = _os.path.join(pecd_dir, fn)
        if not _os.path.exists(path):
            for n in nodes:
                out[n][tech] = _np.zeros(8760)
            continue
        df = _pd.read_parquet(path, columns=['country', 'year', 'month', 'day', 'hour', 'cf'],
                              filters=[('year', '==', float(climate_year))])
        df = df.sort_values(['country', 'month', 'day', 'hour'])
        for n in nodes:
            sub = df[df['country'] == n]
            out[n][tech] = (sub['cf'].to_numpy() if len(sub) >= 8760
                            else _np.zeros(8760))
    return out


def _load_pecd_demand(pecd_dir, climate_year, nodes):
    """Load PECD v3.1 hourly DEMAND SHAPE (dem_MW) for one climate year and
    our nodes (S.75). Same long format [country,year,month,day,hour,dem_MW],
    predicate-pushdown on year. Returns {node: 8760-array} (present-level
    shape; the LEVEL is rescaled to the TYNDP target later by
    scale_demand_to_targets). climate years 1982-2016; returns None-entries
    for years/nodes not present (caller keeps DIETER demand then)."""
    import pandas as _pd, numpy as _np, os as _os
    path = _os.path.join(pecd_dir, 'PECD-country-demand_national_estimates-2030.parquet')
    out = {n: None for n in nodes}
    if not _os.path.exists(path):
        return out
    df = _pd.read_parquet(path, columns=['country', 'year', 'month', 'day', 'hour', 'dem_MW'],
                          filters=[('year', '==', float(climate_year))])
    df = df.sort_values(['country', 'month', 'day', 'hour'])
    for n in nodes:
        sub = df[df['country'] == n]
        if len(sub) >= 8760:
            out[n] = sub['dem_MW'].to_numpy()
    return out


def load_representative_weeks(ts_path, di_path=None, node='DE', n_weeks=6, nodes=None,
                              dk_h2_twh=0.0, dk_nordic_mult=1.0, ntc_cost_mult=1.0,
                              enable_heat=False, dk_heat_twh=15.0,
                              dk_hp_mw=2000.0, dk_heat_storage_mwh=30000.0,
                              dk_cop=3.0, climate_year=None, pecd_dir=None):
    """Load 6-representative-week dataset.

    Sensitivity parameters (Tier 1 / Tier 2 levers).

    Scalar capex / opex / efficiency / capacity values are HARDCODED in this
    function (NOT read from data_input.xlsx -- the di_path arg is vestigial) and
    are then overridden by apply_scenario (TYNDP/DEA year-specific) for
    VRE/fleet/costs/CO2. The multipliers below ONLY scale at default 1.0x.

    Tier 1 / Tier 2 sensitivity flags:

      dk_h2_twh         DK-specific annual H2 demand quota in TWh_H2.
                        0 disables the constraint. Tier 1 #1.

      dk_nordic_mult    Multiplier on the DK residual NTC (default 4800 MW
                        to Nordic; ENTSO-E 2024 NTC reference). Default
                        1.0x. Test 1.5x and 2.0x. Tier 1 #3.

      ntc_cost_mult     Multiplier on annualized NTC capex (322 EUR/MW-km/yr
                        default; from data_input.xlsx 'spatial' sheet).
                        Default 1.0x. Test 0.5x and 2.0x. Tier 2 #8.

    Tier 2 #6 heat-coupling defaults (sources cited; data_input.xlsx 'heat'
    sheet is residual GAMS-DIETER residential-building data, marked
    "Sheet not used in this application", and is not the right reference
    for district-heating HP parameters).

      enable_heat       If True, attach a DK electric heat-pump load with
                        thermal storage. All HP/storage parameters fixed
                        (no investment decision in Tier 2 #6).

      dk_heat_twh = 15  Annual DK HP-supplied heat (TWh_th). DK total DH
                        demand is ~30 TWh/yr (Dansk Fjernvarme statistics
                        2023, https://www.danskfjernvarme.dk/). 15 TWh =
                        50 % HP share, matching DEA "Klimastatus 2023"
                        2030 ambition for HP penetration in DH.

      dk_hp_mw = 2000   HP electric capacity (MW_e). DK installed large
                        DH HP in 2024: ~600 MW_e (Dansk Fjernvarme).
                        2030 ambition: 1500-2500 MW_e (Energinet "System
                        Perspektiv 2050", 2023). 2000 MW_e is the upper
                        end of the 2030 ambition.

      dk_heat_storage_mwh = 30000  Thermal-storage capacity (MWh_th).
                        Existing DK pit storages: Vojens 200 GWh,
                        Marstal 75 GWh, Aalborg 75 GWh, Copenhagen CTR
                        ~120 GWh, total Danish DH ~500 GWh. The HP-fed
                        share is much smaller; 30 GWh = ~15 h of full
                        HP heat output is consistent with daily/weekly-
                        cycling tank/pit storage on the HP segment.
                        DEA Technology Catalogue 142 ("Pit Storage").

      dk_cop = 3.0      Annual seasonal coefficient of performance.
                        DEA Technology Catalogue 40 ("Large Heat Pumps,
                        air to water"): annual SCOP 2.5-3.5 in DK
                        climate (Table 4, 2022 update). 3.0 is a
                        conservative midpoint.
                        https://ens.dk/en/our-services/technology-catalogues/technology-catalogue-generation-electricity-and-district-heating
    """
    if nodes is None:
        nodes = [node]

    print(f"  Loading time series from {ts_path}...")
    print(f"  Nodes: {nodes}")
    ts_all = {n: _parse_excel(ts_path, n) for n in nodes}
    # Multi-weather-year (Phase B): replace VRE capacity factors with PECD for
    # the chosen climate year (demand shape, run-of-river and reservoir inflow
    # stay from DIETER; demand LEVEL is set later by --demand-2040). Documented
    # limitation: demand-shape weather-correlation not yet swapped (needs PECD
    # demand parquet).
    if climate_year is not None and pecd_dir is not None:
        import numpy as _np
        pecd = _load_pecd_cf(pecd_dir, climate_year, nodes)
        for n in nodes:
            for tech in ('pv', 'wind_on', 'wind_off'):
                arr = pecd[n].get(tech)
                if arr is not None and len(arr) >= 8760:
                    ts_all[n][tech] = _np.asarray(arr, dtype=float)[:8760]
        # demand SHAPE from the same climate year (level rescaled later); if the
        # year is absent (post-2016) keep DIETER demand for that node.
        pdem = _load_pecd_demand(pecd_dir, climate_year, nodes)
        _dem_swapped = []
        for n in nodes:
            if pdem.get(n) is not None and len(pdem[n]) >= 8760:
                ts_all[n]['demand'] = _np.asarray(pdem[n], dtype=float)[:8760]
                _dem_swapped.append(n)
        print(f"  PECD climate year {climate_year}: VRE CF replaced "
              f"(PV/onshore/offshore) for {nodes}; demand shape replaced for "
              f"{_dem_swapped or 'none (kept DIETER)'}")
    ts = ts_all[node]

    n_full = 52 * 168
    demand_weeks = ts['demand'][:n_full].reshape(52, 168)
    pv_weeks     = ts['pv'][:n_full].reshape(52, 168)
    wind_weeks   = ts['wind_on'][:n_full].reshape(52, 168)

    def normalize(arr):
        mu, sigma = arr.mean(), arr.std()
        return (arr - mu) / (sigma + 1e-10), mu, sigma

    d_norm, d_mu, d_sig = normalize(demand_weeks)
    p_norm, p_mu, p_sig = normalize(pv_weeks)
    w_norm, w_mu, w_sig = normalize(wind_weeks)
    X = np.hstack([d_norm, p_norm, w_norm])

    print(f"  Clustering 52 weeks into {n_weeks} representative weeks (k-medoids)...")
    medoid_idx, labels, inertia = _kmedoids(X, n_weeks)
    medoid_idx = np.sort(medoid_idx)
    D_mat = np.zeros((52, len(medoid_idx)))
    for j, mi in enumerate(medoid_idx):
        D_mat[:, j] = np.sqrt(np.sum((X - X[mi])**2, axis=1))
    labels = np.argmin(D_mat, axis=1)
    cluster_sizes = np.array([np.sum(labels == c) for c in range(n_weeks)])
    print(f"  Cluster sizes: {cluster_sizes.tolist()} (total={cluster_sizes.sum()} weeks)")

    hours = list(range(n_weeks * 168))
    weight_dict = {}
    week_info = []
    d_dict = {n: {} for n in nodes}
    phi_pv_dict = {n: {} for n in nodes}
    phi_wind_on_dict = {n: {} for n in nodes}
    phi_wind_off_dict = {n: {} for n in nodes}
    phi_ror_dict = {n: {} for n in nodes}
    rsvr_inflow_dict = {n: {} for n in nodes}

    all_demand_weeks = {}
    all_pv_weeks = {}
    all_wind_on_weeks = {}
    all_wind_off_weeks = {}
    all_ror_weeks = {}
    all_rsvr_weeks = {}
    for n in nodes:
        all_demand_weeks[n]  = ts_all[n]['demand'][:n_full].reshape(52, 168)
        all_pv_weeks[n]      = ts_all[n]['pv'][:n_full].reshape(52, 168)
        all_wind_on_weeks[n] = ts_all[n]['wind_on'][:n_full].reshape(52, 168)
        all_wind_off_weeks[n] = ts_all[n]['wind_off'][:n_full].reshape(52, 168)
        all_ror_weeks[n]     = ts_all[n]['ror'][:n_full].reshape(52, 168)
        all_rsvr_weeks[n]    = ts_all[n]['rsvr'][:n_full].reshape(52, 168)

    for c, mi in enumerate(medoid_idx):
        w = cluster_sizes[c]
        base_h = c * 168
        for t_local in range(168):
            h_global = base_h + t_local
            weight_dict[h_global] = float(w)
            for n in nodes:
                d_dict[n][h_global]            = float(all_demand_weeks[n][mi, t_local])
                phi_pv_dict[n][h_global]       = float(all_pv_weeks[n][mi, t_local])
                phi_wind_on_dict[n][h_global]  = float(all_wind_on_weeks[n][mi, t_local])
                phi_wind_off_dict[n][h_global] = float(all_wind_off_weeks[n][mi, t_local])
                phi_ror_dict[n][h_global]      = float(all_ror_weeks[n][mi, t_local])
                rsvr_inflow_dict[n][h_global]  = float(all_rsvr_weeks[n][mi, t_local])
        week_info.append({
            'cluster_id': c, 'medoid_week': int(mi), 'cluster_size': int(w),
            'avg_demand': float(demand_weeks[mi].mean()),
            'avg_pv': float(pv_weeks[mi].mean()),
            'avg_wind': float(wind_weeks[mi].mean()),
        })

    total_weighted_hours = sum(weight_dict[h] for h in hours)
    print(f"  Total weighted hours: {total_weighted_hours:.0f} (real: 8760)")
    for wi in week_info:
        print(f"    Week {wi['cluster_id']}: medoid=week {wi['medoid_week']}, "
              f"size={wi['cluster_size']}, demand={wi['avg_demand']:,.0f} MW, "
              f"PV={wi['avg_pv']:.3f}, Wind={wi['avg_wind']:.3f}")

    result = dict(
        h=hours, h_set=set(hours), n=nodes,
        d=d_dict,
        phi_pv=phi_pv_dict,
        phi_wind=phi_wind_on_dict,
        phi_wind_on=phi_wind_on_dict,
        phi_wind_off=phi_wind_off_dict,
        phi_ror=phi_ror_dict,
        rsvr_inflow=rsvr_inflow_dict,
        weights=weight_dict,
        week_info=week_info,
        n_repr_weeks=n_weeks,
        # Kotzur (2018) seasonal-storage linking: chronological 52-week -> rep-week
        # map (week_labels[p] in 0..n_weeks-1) and hours-per-rep-week. Rep week r
        # occupies hours [r*168, r*168+167]. Used by model_builder to thread an
        # inter-period storage state through the real-year order of the weeks.
        week_labels=[int(x) for x in labels],
        repr_hours=168,
        tech_vre=['pv', 'wind_on', 'wind_off'],
        tech_flex=['ccgt', 'ocgt'],
        tech_inf=['lig', 'nuc', 'hc'],
        capex={'pv': 25400, 'wind_on': 76200, 'wind_off': 135000,  # 2040 DEA, WACC 6% (S.40.4)
               'ccgt': 69000, 'ocgt': 35000, 'lig': 0, 'nuc': 0, 'hc': 0},  # gas 2040 DEA (S.48)
        # Fixed O&M [EUR/MW/yr], DEA Technology Catalogue 2040 central values
        # (datasheets, constant 2020 EUR; S.56). PV sheet 22 (8.1k incl. land
        # rent), onshore sheet 20 (16.0k), offshore AC fixed sheet 21 (30.0k),
        # CCGT sheet 05 CC extraction (28.6k, 2030/2050 interpolated), OCGT
        # sheet 52 pure-power peaker (8.1k). Enters BOTH the LP objective
        # (investable techs) and the cost-recovery AC (all VRE), closing the
        # CR-overstatement gap flagged in the S.56 audit.
        fom={'pv': 8100, 'wind_on': 16000, 'wind_off': 30000,
             'ccgt': 28600, 'ocgt': 8100, 'lig': 0, 'nuc': 0, 'hc': 0},
        opex={'pv': 0, 'wind_on': 0, 'wind_off': 0,  # gas opex 2040: TYNDP DE2040 NG 4.07 EUR/GJ (S.48)
              'ccgt': 30, 'ocgt': 45, 'lig': 25, 'nuc': 10, 'hc': 30},
        c_curt={'pv': 0.5, 'wind_on': 0.5, 'wind_off': 0.5},
        epsilon={'pv': 0, 'wind_on': 0, 'wind_off': 0,
                 'ccgt': 0.37, 'ocgt': 0.55, 'lig': 1.0, 'nuc': 0.0, 'hc': 0.73},
        tau=123.0,  # TYNDP 2024 Distributed Energy 2040 (S.39.4)
        sto_capex_p=14400, sto_capex_e=11300,  # 2040 Li-ion, WACC 6% (S.40.4)
        # Battery fixed O&M: DEA storage catalogue convention = 2.9%/yr of
        # specific investment (NREL ATB uses 2.5%). Applied to the implied
        # overnight of each component (power ~184 EUR/kW, energy ~100 EUR/kWh
        # at WACC 6%): ~5.0k EUR/MW/yr + ~2.9k EUR/MWh/yr (S.56).
        sto_fom_p=5000, sto_fom_e=2900,
        sto_c_ch=0.5, sto_c_dis=0.5,
        eta_ch=0.92, eta_dis=0.92, phi_s=0.0002,
        sc_capex=39200, sc_opex=2.0, sc_eta=0.70,  # 2040 electrolyser, WACC 6% (S.40.4)
        sc_fom=16000,  # DEA renewable-fuels cat.: AEC 4%/yr of capex, 1 GW scale, 2040 (S.56)
        sc_h2_target=20_000_000,
        v_ls=10000,
        reserve_frac=0.03,
        fast_reserve_frac=0.003,
        rho={'ccgt': 0.15, 'ocgt': 1.0, 'lig': 0.04, 'nuc': 0.04, 'hc': 0.06},
        k_exist_inf={'lig': 0, 'nuc': 0, 'hc': 0},
        mu_inf={'lig': 0.4, 'nuc': 0.80, 'hc': 0.40},
        k_exist_vre={'pv': 0, 'wind_on': 0, 'wind_off': 0},
        k_bar_vre={'pv': 178506, 'wind_on': 168500, 'wind_off': 21723},
        k_bar_flex={'ccgt': 80000, 'ocgt': 50000},
        k_bar_sto_p=200000, k_bar_sto_e=800000,
        k_bar_sc=200000,
        dsm_capex=10000, dsm_opex=10.0, dsm_eta=0.95,  # 2040 enabling (S.48)
        k_bar_dsm=16000,  # 2040 DSR potential (S.41.5)
        ev_capex=4000, ev_opex=5.0,  # 2040 V2G enabling (S.48)
        ev_eta_ch=0.96, ev_eta_dis=0.96, ev_phi_s=0.0,
        ev_ep_ratio=4.0,
        k_bar_ev=40000,  # 2040 EV smart-charge/V2G flex (S.41.5)
        k_ror={'DE': 3955, 'CZ': 314, 'DK': 7, 'AT': 7157, 'NL': 28},  # MW: recalibrated so model energy = TYNDP DE2040 ROR at CF 0.478 (progress.md S.81.1)
        rsvr_params={
            'AT': {'power_mw': 4788, 'energy_mwh': 2394000,
                   'eta_ch': 0.90, 'eta_dis': 0.90, 'opex': 1.0},
        },
    )

    # ---- Today's installed flexibility fleet (B_today, progress.md S.36) --
    # Per-node currently installed flexibility, in MW, for the transition
    # baseline. Used only by directed_search.LPOracle.make_today_vector();
    # has no effect on the cost-optimal baseline or the global frontier.
    #
    # Scope: model-relevant *grid-scale* capacity. Residential / behind-the-
    # meter batteries are excluded (not centrally dispatchable in the model).
    # Pumped hydro is NOT included here -- AT pumped hydro is represented
    # separately via rsvr_params; DE/CZ/DK/NL pumped hydro is a known model
    # gap (see progress.md T3.4).
    #
    # Sources (values are end-2024 / early-2025, rounded):
    #   sto  grid-scale battery power. DE 2.3 GW operational large-scale
    #        (Clean Energy Wire / Modo Energy, 2024-25). NL ~0.4 GW utility-
    #        scale (SolarPower Europe Battery Outlook 2025). AT/CZ utility-
    #        scale ~0.1-0.15 GW (most of AT 1.1 GWh / CZ ~2.3 GWh fleet is
    #        residential). DK ~0.1 GW utility-scale.
    #   sc   electrolysis/P2G. DE 0.18 GW operational (Clean Energy Wire,
    #        2024). NL ~0.1 GW (EU Hydrogen Observatory). DK ~0.05 GW.
    #        CZ/AT ~0.
    #   dsm  market-active demand-side load shifting. Order-of-magnitude
    #        estimates -- today's *contracted/active* DSM is small relative
    #        to technical potential (DE McKinsey/Agora 5-7 GW *potential*).
    #        ESTIMATE, flagged for firmer sourcing.
    #   ev   dispatchable EV smart-charging fleet. ~0 everywhere: EnWG 14a
    #        (Jan 2024) is grid-protection throttling, not a market
    #        flexibility resource; no enrolled V1G/V2G fleet at scale.
    # NOTE: the DE scenario mandates 20 TWh/yr electrolytic H2; today's
    # 0.18 GW cannot deliver it, so B_today's DE P2G is raised to the
    # H2-feasibility minimum inside make_today_vector() (see T2.1).
    # S.57 revision (mid-2026 figures; sources in progress.md S.57):
    #   sto  DE 2.7 GW (Modo Energy 02/2026; battery-charts grid-scale);
    #        NL ~0.6 GW (CBS + Rabobank); AT ~0.06 GW (grid-scale genuinely
    #        tiny; most AT storage residential); DK ~0.13 GW; CZ ~0.1 GW
    #        (NOTE: CZ jumps to ~0.45 GW end-2026 -- sensitivity case).
    #   sc   DE 181 MW (EWI/CLEW); DK 90 MW (Kasso 52 + HySynergy 20 + ...);
    #        AT 15 MW (OMV 10 + H2FUTURE 6); NL ~25 MW operational only
    #        (Holland Hydrogen 1 200 MW arrives late 2026).
    #   dsm  market-active (not technical potential); CZ trimmed to 100
    #        (aggregation only legalized 2025, Lex OZE III).
    #   ev   0 everywhere confirmed (largest EU V2G fleet ~1 MW, Utrecht).
    result['k_today_flex'] = {
        'DE': {'sto': 2700.0, 'sc': 180.0, 'dsm': 1500.0, 'ev': 0.0},
        'CZ': {'sto': 100.0,  'sc': 0.0,   'dsm': 100.0,  'ev': 0.0},
        'DK': {'sto': 130.0,  'sc': 90.0,  'dsm': 100.0,  'ev': 0.0},
        'AT': {'sto': 60.0,   'sc': 15.0,  'dsm': 150.0,  'ev': 0.0},
        'NL': {'sto': 600.0,  'sc': 25.0,  'dsm': 250.0,  'ev': 0.0},
    }

    # EV availability profile
    ev_avail = {}
    for c, wi in enumerate(week_info):
        base_h = c * 168
        for t_local in range(168):
            h_global = base_h + t_local
            day_of_week = t_local // 24
            hour_of_day = t_local % 24
            if day_of_week >= 5:
                avail = 0.70
            elif 8 <= hour_of_day < 18:
                avail = 0.30
            else:
                avail = 0.80
            ev_avail[h_global] = avail
    result['ev_avail'] = {n: dict(ev_avail) for n in nodes}

    # ---- SENSITIVITY PARAMS (Tier 1 / Tier 2 levers) -----------------
    # Stored on the result dict so model_builder, LPOracle and the results
    # writer can all see them. Defaults reproduce the baseline behaviour
    # documented in progress.md sections 31-33.
    result['sc_h2_target_dk'] = float(dk_h2_twh) * 1_000_000.0  # TWh -> MWh
    result['dk_nordic_mult'] = float(dk_nordic_mult)
    result['ntc_cost_mult'] = float(ntc_cost_mult)
    result['enable_heat'] = bool(enable_heat)
    result['dk_heat_twh'] = float(dk_heat_twh)
    result['dk_hp_mw'] = float(dk_hp_mw)
    result['dk_heat_storage_mwh'] = float(dk_heat_storage_mwh)
    result['dk_cop'] = float(dk_cop)
    if dk_h2_twh > 0:
        print(f"  DK H2 quota: {dk_h2_twh:.1f} TWh/yr (Tier 1 #1)")
    if abs(dk_nordic_mult - 1.0) > 1e-6:
        print(f"  DK-Nordic NTC multiplier: {dk_nordic_mult:.2f}x (Tier 1 #3)")
    if abs(ntc_cost_mult - 1.0) > 1e-6:
        print(f"  NTC capex multiplier: {ntc_cost_mult:.2f}x (Tier 2 #8)")

    # ---- DK heat-pump demand profile (Tier 2 #6) ---------------------
    # Synthesised from a seasonal sinusoid (peak in mid-January, trough in
    # mid-July) modulated by a small daily morning peak. Profile depends
    # only on the calendar position of each clustered medoid week, so
    # other sensitivity variants (which keep the same clustering) see the
    # same per-hour heat demand. Annual integral targets dk_heat_twh.
    if 'DK' in nodes and enable_heat:
        avg_mw = float(dk_heat_twh) * 1_000_000.0 / 8760.0
        seasonal_amp = avg_mw * 0.75   # +-75% of mean -> peak/trough ~ 3:1
        daily_amp_frac = 0.15
        heat_demand_dk = {}
        for c, mi in enumerate(medoid_idx):
            base_h = c * 168
            week_in_year = int(mi)  # 0..51
            for t_local in range(168):
                h_global = base_h + t_local
                day_in_week = t_local // 24
                hour_in_day = t_local % 24
                day_of_year = week_in_year * 7 + day_in_week
                seasonal = np.cos(2 * np.pi * (day_of_year - 14) / 365.0)
                daily = np.cos(2 * np.pi * (hour_in_day - 8) / 24.0)
                heat_demand_dk[h_global] = float(
                    (avg_mw + seasonal_amp * seasonal) *
                    (1.0 + daily_amp_frac * daily)
                )
        weighted_total = sum(weight_dict[h] * heat_demand_dk[h] for h in hours)
        result['heat_demand_dk'] = heat_demand_dk
        print(f"  DK heat coupling enabled (Tier 2 #6):")
        print(f"    HP capacity: {dk_hp_mw:,.0f} MW_e (fixed), COP={dk_cop:.1f}")
        print(f"    Thermal storage: {dk_heat_storage_mwh:,.0f} MWh_th")
        print(f"    Annual target: {dk_heat_twh:.1f} TWh_th, "
              f"avg {avg_mw:,.0f} MW_th, peak ~{(avg_mw+seasonal_amp)*1.15:,.0f} MW_th")
        print(f"    Weighted-clustered total: {weighted_total/1e6:.2f} TWh_th "
              f"(should be ~{dk_heat_twh:.1f})")

    # Transmission links (multi-node only)
    if len(nodes) > 1:
        result['links'] = {}
        _NTC_CAPEX = 322.0 * float(ntc_cost_mult)  # EUR/MW-km/yr (Tier 2 #8)

        if 'CZ' in nodes:
            result['links']['DE_CZ'] = {
                'from': 'DE', 'to': 'CZ', 'inc': {'DE': 1, 'CZ': -1},
                'ntc_fixed': 2600, 'ntc_max': 5600, 'dist_km': 399,
                'capex_per_mw_km_yr': _NTC_CAPEX, 'loss_per_1000km': 0.02,
            }
        if 'DK' in nodes:
            result['links']['DE_DK'] = {
                'from': 'DE', 'to': 'DK', 'inc': {'DE': 1, 'DK': -1},
                'ntc_fixed': 4100, 'ntc_max': 9100, 'dist_km': 573,
                'capex_per_mw_km_yr': _NTC_CAPEX, 'loss_per_1000km': 0.02,
            }
        if 'AT' in nodes:
            result['links']['DE_AT'] = {
                'from': 'DE', 'to': 'AT', 'inc': {'DE': 1, 'AT': -1},
                'ntc_fixed': 7500, 'ntc_max': 11500, 'dist_km': 464,
                'capex_per_mw_km_yr': _NTC_CAPEX, 'loss_per_1000km': 0.02,
            }
        if 'NL' in nodes:
            result['links']['DE_NL'] = {
                'from': 'DE', 'to': 'NL', 'inc': {'DE': 1, 'NL': -1},
                'ntc_fixed': 5000, 'ntc_max': 8000, 'dist_km': 326,
                'capex_per_mw_km_yr': _NTC_CAPEX, 'loss_per_1000km': 0.02,
            }
        if 'DK' in nodes and 'NL' in nodes:
            result['links']['DK_NL'] = {
                'from': 'DK', 'to': 'NL', 'inc': {'DK': 1, 'NL': -1},
                'ntc_fixed': 700, 'ntc_max': 2700, 'dist_km': 620,
                'capex_per_mw_km_yr': _NTC_CAPEX, 'loss_per_1000km': 0.02,
            }
        if 'CZ' in nodes and 'AT' in nodes:
            result['links']['CZ_AT'] = {
                'from': 'CZ', 'to': 'AT', 'inc': {'CZ': 1, 'AT': -1},
                'ntc_fixed': 900, 'ntc_max': 2400, 'dist_km': 599,
                'capex_per_mw_km_yr': _NTC_CAPEX, 'loss_per_1000km': 0.02,
            }

        # Node-specific parameters from DIETER Technologies sheet
        result['node_params'] = {
            'DE': {
                'k_bar_vre': {'pv': 178506, 'wind_on': 168500, 'wind_off': 21723},
                'k_bar_flex': {'ccgt': 80000, 'ocgt': 50000},
                'k_exist_inf': {'lig': 0, 'nuc': 0, 'hc': 0},
                'k_bar_sto_p': 200000, 'k_bar_sto_e': 800000,
                'k_bar_sc': 200000, 'k_bar_dsm': 16000, 'k_bar_ev': 40000,
            },
            'CZ': {
                'k_bar_vre': {'pv': 12164, 'wind_on': 3411, 'wind_off': 0},
                'k_bar_flex': {'ccgt': 5000, 'ocgt': 5000},
                'k_exist_inf': {'lig': 0, 'nuc': 5187, 'hc': 0},
                'k_bar_sto_p': 15000, 'k_bar_sto_e': 60000,
                'k_bar_sc': 15000, 'k_bar_dsm': 2000, 'k_bar_ev': 6000,
            },
        }
        if 'DK' in nodes:
            result['node_params']['DK'] = {
                'k_bar_vre': {'pv': 29579, 'wind_on': 12499, 'wind_off': 14242},
                'k_bar_flex': {'ccgt': 5000, 'ocgt': 3000},
                'k_exist_inf': {'lig': 0, 'nuc': 0, 'hc': 0},
                'k_bar_sto_p': 40000, 'k_bar_sto_e': 160000,
                'k_bar_sc': 20000, 'k_bar_dsm': 2000, 'k_bar_ev': 6000,
            }
        if 'AT' in nodes:
            result['node_params']['AT'] = {
                'k_bar_vre': {'pv': 25140, 'wind_on': 32939, 'wind_off': 0},
                'k_bar_flex': {'ccgt': 5000, 'ocgt': 5000},
                'k_exist_inf': {'lig': 0, 'nuc': 0, 'hc': 0},
                'k_bar_sto_p': 30000, 'k_bar_sto_e': 120000,
                'k_bar_sc': 10000, 'k_bar_dsm': 1600, 'k_bar_ev': 4000,
            }
        if 'NL' in nodes:
            result['node_params']['NL'] = {
                'k_bar_vre': {'pv': 40706, 'wind_on': 15100, 'wind_off': 33000},
                'k_bar_flex': {'ccgt': 8000, 'ocgt': 8000},
                'k_exist_inf': {'lig': 0, 'nuc': 0, 'hc': 0},
                'k_bar_sto_p': 40000, 'k_bar_sto_e': 160000,
                'k_bar_sc': 20000, 'k_bar_dsm': 3000, 'k_bar_ev': 10000,
            }

        # Residual interconnection
        result['resid_intercon'] = {}
        if 'DK' in nodes:
            dk_nordic_ntc = int(round(5450 * float(dk_nordic_mult)))
            result['resid_intercon']['DK'] = {
                'ntc': dk_nordic_ntc, 'price_export': 30.0, 'price_import': 40.0,
                'ntc_max': 14550, 'dist_km': 500, 'capex_per_mw_km_yr': _NTC_CAPEX,
            }
            print(f"  DK residual interconnection: {dk_nordic_ntc} MW to Nordic (NO+SE+UK)"
                  f"{' [Tier 1 #3 multiplier '+format(dk_nordic_mult,'.2f')+'x]' if abs(dk_nordic_mult-1.0)>1e-6 else ''}")
        if 'AT' in nodes:
            result['resid_intercon']['AT'] = {
                'ntc': 2075, 'price_export': 35.0, 'price_import': 45.0,
                'ntc_max': 4725, 'dist_km': 300, 'capex_per_mw_km_yr': _NTC_CAPEX,
            }
            print(f"  AT residual interconnection: 2075 MW to Alpine (CH+IT)")
        if 'NL' in nodes:
            result['resid_intercon']['NL'] = {
                'ntc': 5100, 'price_export': 32.0, 'price_import': 42.0,
                'ntc_max': 13100, 'dist_km': 200, 'capex_per_mw_km_yr': _NTC_CAPEX,
            }
            print(f"  NL residual interconnection: 5100 MW to BE+NO+UK")
        for lname, ldata in result['links'].items():
            print(f"  Transmission: {lname} link, NTC={ldata['ntc_fixed']} MW (fixed)")

    # DE residual interconnection to the rest of Europe (FR, BE, CH, PL, SE, NO,
    # UK, LU). DE total cross-border NTC ~39 GW (TYNDP 2024 2030 ref grid),
    # ~67 GW with 2040 candidate projects. Without this, single-node DE is an
    # unrealistic electrical island that must absorb every PV midday surplus
    # domestically -> VF biased LOW / storage need biased HIGH. The residual =
    # DE total minus the explicitly-modelled DE links present, so single-node
    # DE gets the full ~39/67 GW and the 5-node gets DE-to-non-modelled only.
    # CAVEAT (disclose in paper): fixed export price cannot capture neighbour
    # correlation (EU-wide midday solar glut); set conservatively as a low
    # "dump" price to avoid over-valuing midday export. Investable (residual
    # expansion = transmission-as-mitigation lever).
    result.setdefault('links', {})
    result.setdefault('resid_intercon', {})
    _de_links = [l for l in result['links'].values() if 'DE' in (l.get('from'), l.get('to'))]
    _de_modelled = sum(l['ntc_fixed'] for l in _de_links)
    _de_modelled_max = sum(l.get('ntc_max', l['ntc_fixed']) for l in _de_links)
    _de_resid = max(0.0, 39000.0 - _de_modelled)
    _de_resid_max = max(_de_resid, 67000.0 - _de_modelled_max)
    if _de_resid > 0:
        result['resid_intercon']['DE'] = {
            'ntc': _de_resid, 'price_export': 20.0, 'price_import': 75.0,
            'ntc_max': _de_resid_max, 'dist_km': 400,
            'capex_per_mw_km_yr': 322.0 * float(ntc_cost_mult),
        }
        print(f"  DE residual interconnection: {_de_resid:.0f} MW to rest-of-Europe "
              f"(investable to {_de_resid_max:.0f}; export 20 / import 75 EUR/MWh)")

    return result


# ---- TYNDP 2024 Distributed Energy 2040 final electricity demand [TWh/yr] ----
# Per-country FINAL demand incl. EV + heat-pump + industrial electrification,
# EXCL. electrolysis (the model adds electrolysis separately via the H2 quota).
# Source: official ENTSO-E TYNDP 2024 output file
# MMStandardOutputFile_DE2040_Plexos_CY2009 ("Native Demand" summed across each
# country's market nodes incl. EV + RETE electrification nodes), cross-checked
# vs the Scenario Report Data Figures (EU27 final demand 4091 TWh). See
# progress.md S.61. The model's base demand was only the TYNDP "main node"
# (DE 565 TWh) -> it dropped ~295 TWh of 2040 electrification -> overstated
# penetration/cannibalization. Scaling base demand to these totals fixes it.
TYNDP_DE2040_DEMAND_TWH = {
    'DE': 939.7, 'CZ': 98.0, 'DK': 47.7, 'AT': 126.5, 'NL': 231.6,
}

# TYNDP 2024 DE2040 electrolyser ELECTRICITY consumption [TWh elec/yr], same
# source as the demand dict. Model H2 quota is in H2 OUTPUT, so multiply by
# sc_eta (0.7): H2_out = elec * eta. CAVEAT: DK (85.7 TWh elec) >> DK demand
# because TYNDP makes DK a hydrogen EXPORT hub; our model has no H2 trade, so
# this becomes a large domestic flexible load that strongly lifts DK VF (S.61).
TYNDP_DE2040_ELECTROLYSER_TWH = {
    'DE': 152.4, 'CZ': 5.1, 'DK': 85.7, 'AT': 36.4, 'NL': 61.0,
}


def scale_demand_to_targets(D, targets_twh=None, verbose=True):
    """Scale each node's hourly base-demand profile so its WEIGHTED ANNUAL
    energy matches a target [TWh/yr] (default = TYNDP DE2040 final demand,
    excl. electrolysis). Preserves the hourly SHAPE; only rescales the level.
    Mutates D['d'] in place and returns the applied per-node scale factors
    (S.61 demand-vintage fix)."""
    if targets_twh is None:
        targets_twh = D.get('demand_targets') or TYNDP_DE2040_DEMAND_TWH
    # When EV is modelled as a separate flexible load (ev_2040), its charging
    # energy is added back via the EV driving sink, so the BASE target must
    # exclude it to avoid double counting (S.63).
    ev_elec = D.get('ev_elec_twh', {}) if D.get('ev_2040') else {}
    W = D['weights']
    factors = {}
    for n in D['n']:
        if n not in targets_twh:
            continue
        cur_mwh = sum(W[t] * D['d'][n][t] for t in D['h'])
        if cur_mwh <= 0:
            continue
        tgt_mwh = (float(targets_twh[n]) - float(ev_elec.get(n, 0.0))) * 1e6
        f = tgt_mwh / cur_mwh
        for t in D['h']:
            D['d'][n][t] *= f
        factors[n] = f
        if verbose:
            print(f"  [demand-2040] {n}: {cur_mwh/1e6:.1f} -> "
                  f"{tgt_mwh/1e6:.1f} TWh/yr (x{f:.3f})")
    return factors


# ============================================================================
# EV 2040 (TYNDP DE2040 EV-Modelling-Inputs) — realistic flexible-load remodel
# ============================================================================
# Per-node PASSENGER EV fleet (S.63 audit, read from EV-Modelling-Inputs):
#   K_ev = fleet x 7.4 kW charger (MW, aggregate charge/discharge power; V2G);
#   E    = fleet battery (MWh); elec = annual charging demand (MWh);
#   minSOC 20%. Source: EV_VEHICLES.xlsx / EV_CHARGING STATIONS.xlsx,
#   2040 "Distributed Energy". CHARGE/DISCHARGE eff 0.94 (charging stations).
EV2040 = {
    'DE': {'K_ev': 282011, 'E': 3163102, 'elec': 84210829},
    'CZ': {'K_ev': 18423,  'E': 206636,  'elec': 6312808},
    'DK': {'K_ev': 15173,  'E': 170179,  'elec': 7081934},
    'AT': {'K_ev': 35509,  'E': 398274,  'elec': 9487254},
    'NL': {'K_ev': 47536,  'E': 533179,  'elec': 19082943},
}
EV2040_MINSOC = 0.20
EV2040_ETA = 0.94
# 24h profiles (TYNDP AVAILABILITY_PROFILES prosumer, DRIVING_PROFILES passenger)
EV_AVAIL_WD = [0.716,0.727,0.736,0.724,0.604,0.254,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.254,0.462,0.519,0.564,0.608,0.646,0.679,0.701]
EV_AVAIL_WE = [0.652,0.66,0.67,0.67,0.642,0.57,0.392,0.278,0.247,0.247,0.247,0.247,0.278,0.347,0.386,0.415,0.455,0.485,0.522,0.551,0.574,0.599,0.626,0.641]
EV_DRIVE_WD = [0.0022,0.0018,0.0017,0.0015,0.0013,0.0024,0.0053,0.0129,0.0146,0.0135,0.0137,0.0139,0.0119,0.0111,0.011,0.0112,0.0121,0.0122,0.0102,0.0069,0.0048,0.0034,0.0032,0.0027]
EV_DRIVE_WE = [0.0013,0.0012,0.0012,0.0012,0.0011,0.001,0.0009,0.0011,0.0015,0.0017,0.0023,0.0026,0.0023,0.0015,0.0015,0.0017,0.0018,0.0021,0.0021,0.0018,0.0012,0.001,0.0009,0.0009]


def build_ev_2040(D, verbose=True):
    """Populate D with the realistic 2040 EV representation (S.63): fixed fleet
    power/energy, hourly plug-in availability, and an hourly DRIVING-demand
    sink scaled to the TYNDP annual EV charging energy. Sets ev_2040=True so
    model_builder/LPOracle switch to the driving-load EV. Driving weekly shape
    sums ~1.0 over 168 h, so drive[t]=shape[h_in_week]*elec/52 gives weighted
    annual = elec (MWh)."""
    H = int(D.get('repr_hours', 168))
    EVP = D.get('ev_params') or EV2040   # per-scenario fleet (S.73) else 2040
    D['ev_2040'] = True
    D['ev_minsoc'] = EV2040_MINSOC
    D['ev_eta'] = EV2040_ETA
    D['ev_K'] = {}; D['ev_E'] = {}; D['ev_elec_twh'] = {}
    D['ev_drive'] = {}; D['ev_avail2040'] = {}
    for n in D['n']:
        if n not in EVP:
            continue
        p = EVP[n]
        D['ev_K'][n] = float(p['K_ev']); D['ev_E'][n] = float(p['E'])
        D['ev_elec_twh'][n] = p['elec'] / 1e6
        drive = {}; avail = {}
        for t in D['h']:
            hw = t % H; dow = (hw // 24) % 7; hod = hw % 24
            wknd = dow >= 5
            shp = (EV_DRIVE_WE if wknd else EV_DRIVE_WD)[hod]
            drive[t] = shp * p['elec'] / 52.0       # MWh in this hour = MW
            avail[t] = (EV_AVAIL_WE if wknd else EV_AVAIL_WD)[hod]
        D['ev_drive'][n] = drive
        D['ev_avail2040'][n] = avail
        if verbose:
            print(f"  [ev-2040] {n}: K_ev={p['K_ev']/1000:.1f} GW, "
                  f"E={p['E']/1e6:.1f} TWh, drive={p['elec']/1e6:.1f} TWh/yr")
    return D
