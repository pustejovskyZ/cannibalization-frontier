"""
tyndp_fixes.py -- S.103: bring the remaining TYNDP 2024 system components into
the model (proofread finding, progress.md S.99-S.102). Applied right after
apply_scenario() in directed_search.py, grid_floor.py and node_factorial.py.

Controlled by env TYNDP_FIXES (comma list or 'all'); unset/empty = no change,
bit-identical to the original production model.

  hubs   offshore hubs (grid-connected OHEL nodes): TYNDP installed MW and the
         hubs' own CY2009 hourly availability, blended into the node offshore
         profile on the SAME medoid weeks (capacity, energy and shape = TYNDP).
  hydro  reservoir + pumped hydro for DE, CZ, AT at TYNDP turbine/pump power;
         natural inflow = reservoir generation + net open-loop generation
         (turbine - pump x 0.81) + AT pondage; inflow shape = AT reservoir
         inflow series (Alpine). Energy: PHS 6 h (DE), 5 h (CZ) of turbine
         power, reservoirs 168 h; AT keeps its calibrated 2.394 TWh.
  other  'Others renewable' + 'Others non-renewable' + biofuel plants as a flat
         must-run injection equal to TYNDP annual generation (TYNDP profile is
         flat, cf. Hourly Market Data).
  h2     TYNDP hydrogen-fired CCGT fleet as existing capacity, fuel cost =
         TYNDP hydrogen price / 0.58, no CO2.
  gas    TYNDP existing natural-gas fleet (CCGT/conventional, OCGT) as sunk
         existing capacity; further gas investment stays LP-endogenous.
  dsr    TYNDP explicit demand-side response (load reduction) as three price
         bands calibrated to the DE2040 price-DSR relation (12% @60, 35% @110,
         53% @300 EUR/MWh). Load SHIFTING (our DSM) is unchanged.
  pots   battery-power potentials raised where below TYNDP's own planned battery
         fleet (utility/4 h + prosumer/3.2 h from the Installed-energy rows),
         rounded up to the next 5 GW (binding only for NL 2050: 40 -> 50 GW).
  fuel   gas-plant fuel cost from the TYNDP 2024 gas BLEND price of the
         scenario (natural gas + biomethane + synthetic methane; 'Final Fuel
         Prices 2022' sheet, Gas row: NT2030 8.845, DE2030 5.857, DE2040
         10.438, DE2050 14.804 EUR/GJ; DE2035 = mean of 2030/2040), divided by
         the efficiency implied by the model's emission factors (CCGT 0.546,
         OCGT 0.367), and carbon charged on the fossil share only (TYNDP blend
         shares: DE2030 74%, DE2040 35%, DE2050 2%, DE2035 54.5%; NT2030 82%
         backed out of its NG/biomethane blend price).
  oil    TYNDP oil plants (light/heavy oil; DK ~0.75 GW all years, DE 1.6 GW
         NT2030 / 0.7 GW DE2035) as existing capacity, TYNDP light-oil price /
         0.35, 0.76 tCO2/MWh_el.
  borders residual borders priced HOURLY at TYNDP's own neighbour marginal
         costs (NTC-weighted: DE<-FR,CH,LU,PL,NO,UK,SE4,BE; DK<-NO,UK,SE4,SE3;
         AT<-CH,IT-N; NL<-BE,UK,NO), import +1 / export -1 EUR/MWh; DE2040
         series shifted +24 h to the PECD axis (tyndp_borders_dsr.json,
         extract_borders_dsr.py). Overrides the static prices and the
         'fuel' DE import price.
  dsr bands now calibrated per scenario on TYNDP's DE00 price-DSR relation.
Batteries are NOT fixed (they are the paper's choice variable); see
tyndp_battery_benchmark in directed_search for the TYNDP battery point.
"""
import os, json

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXTRAS = os.path.join(_HERE, 'tyndp 2040', 'tyndp_extras.json')
ALL = ('hubs', 'hydro', 'other', 'h2', 'gas', 'dsr', 'pots', 'fuel', 'oil', 'borders')
_BORDERS = os.path.join(_HERE, 'tyndp 2040', 'tyndp_borders_dsr.json')
PHS_HOURS = {'DE': 6.0, 'CZ': 5.0}
RES_HOURS = 168.0
ETA_PHS = 0.90                       # per direction (AT calibration)
H2CCGT_EFF = 0.58
DSR_BANDS = (('dsr1', 0.12, 60.0), ('dsr2', 0.35, 110.0), ('dsr3', 0.53, 300.0))


def requested():
    raw = os.environ.get('TYNDP_FIXES', '').strip().lower()
    if not raw:
        return ()
    if raw == 'all':
        return ALL
    out = tuple(t.strip() for t in raw.split(',') if t.strip())
    bad = [t for t in out if t not in ALL]
    if bad:
        raise SystemExit(f"TYNDP_FIXES: unknown {bad}; allowed {ALL} or 'all'")
    return out


def _node_inf(D, n):
    """per-node existing-capacity dict used by model_builder (_np fallback)."""
    npar = D.setdefault('node_params', {}).setdefault(n, {})
    if 'k_exist_inf' not in npar:
        npar['k_exist_inf'] = dict(D.get('k_exist_inf', {}))
    return npar['k_exist_inf']


def _add_inf_tech(D, g, opex, eps=0.0, rho=1.0, mu=0.0):
    if g not in D['tech_inf']:
        D['tech_inf'] = list(D['tech_inf']) + [g]
    for key, val in (('capex', 0.0), ('fom', 0.0), ('opex', opex), ('epsilon', eps),
                     ('rho', rho), ('mu_inf', mu)):
        D.setdefault(key, {})[g] = val
    D.setdefault('k_exist_inf', {}).setdefault(g, 0.0)


def _set_exist(D, n, g, mw):
    _node_inf(D, n)[g] = float(mw)
    if n == 'DE':                       # DE also reads the global dict
        D['k_exist_inf'][g] = float(mw)


def apply(D, scenario, fixes=None):
    fixes = requested() if fixes is None else tuple(fixes)
    if not fixes:
        return D
    if D.get('_tyndp_fixes_applied'):
        raise RuntimeError('tyndp_fixes.apply called twice on the same data')
    D['_tyndp_fixes_applied'] = tuple(fixes)
    X = json.load(open(_EXTRAS))
    try:
        from scenarios import _ALIAS
        scenario = _ALIAS.get(scenario, scenario)
    except ImportError:
        pass
    if scenario not in X:
        raise SystemExit(f"TYNDP_FIXES: no extras for scenario '{scenario}'")
    S = X[scenario]
    W = D['weights']; H = D['h']; wsum = sum(W[t] for t in H)
    nodes = list(D['n'])
    print(f"  ** TYNDP_FIXES={','.join(fixes)} ({scenario}) **")

    # S.108: the TYNDP hourly series (hub availability, neighbour prices) exist
    # for climate year 2009 only. For other weather years (robustness runs) the
    # hub output follows the node's own offshore profile of that year, scaled to
    # TYNDP's annual hub energy, and the border prices keep TYNDP's
    # month-by-hour-of-day pattern instead of the CY2009 chronology.
    cy = D.get('climate_year')
    other_cy = cy is not None and int(cy) != 2009

    if 'hubs' in fixes:
        med = {wi['cluster_id']: wi['medoid_week'] for wi in D['week_info']}
        for n in nodes:
            hub = S.get(n, {}).get('hub')
            if not hub or hub['mw'] <= 0:
                continue
            prof = hub['profile']
            if scenario == 'de2040':
                # TYNDP's DE2040 hourly files are offset by +24 h against PECD
                # CY2009 and match it less well even after the shift (corr
                # 0.76-0.84 vs 0.87-0.99 for DE2035/DE2050, whose hub series
                # agree with each other at 0.99-1.00). Use the mean per-MW hub
                # availability of the DE2035 and DE2050 files (same CY2009),
                # rescaled to the DE2040 annual hub capacity factor (S.105).
                p35 = X['de2035'][n]['hub']['profile']; p50 = X['de2050'][n]['hub']['profile']
                pm = [0.5 * (a + b) for a, b in zip(p35, p50)]
                k = (sum(prof) / len(prof)) / (sum(pm) / len(pm))
                prof = [min(1.0, k * v) for v in pm]
            assert len(prof) == 52 * 168, len(prof)
            ph = {t: prof[med[t // 168] * 168 + t % 168] for t in H}
            if other_cy:
                ph = {t: D['phi_wind_off'][n][t] for t in H}
            targets = [D['node_params'][n]['k_bar_vre']] if n in D.get('node_params', {}) else []
            if n == 'DE' and all(D['k_bar_vre'] is not tt for tt in targets):
                targets.append(D['k_bar_vre'])
            # rescale so the representative weeks deliver TYNDP's annual hub energy
            _e = sum(W[t] * ph[t] for t in H) * hub['mw']
            if _e > 0:
                _k = hub['twh'] * 1e6 / _e
                ph = {t: min(1.0, _k * ph[t]) for t in H}
            k0 = float(targets[0]['wind_off']); k1 = k0 + hub['mw']
            po = D['phi_wind_off'][n]
            # alignment guard: hub and node offshore weather must co-move
            import math
            xs = [ph[t] for t in H]; ys = [po[t] for t in H]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            sx = math.sqrt(sum((a - mx) ** 2 for a in xs)); sy = math.sqrt(sum((b - my) ** 2 for b in ys))
            corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0.0
            if corr < 0.75 and not other_cy:
                raise SystemExit(f'hubs {n}: hub profile not aligned with node offshore weather (corr {corr:.2f})')
            D['phi_wind_off'][n] = {t: (k0 * po[t] + hub['mw'] * ph[t]) / k1 for t in H}
            for tt in targets:
                tt['wind_off'] = k1
            cf_old = sum(W[t] * po[t] for t in H) / wsum
            cf_hub = sum(W[t] * ph[t] for t in H) / wsum
            if other_cy:
                print(f"     hubs {n}: CY{cy}: node offshore profile scaled to TYNDP hub energy")
            print(f"     hubs {n}: wind_off {k0/1e3:.1f} -> {k1/1e3:.1f} GW "
                  f"(corr {corr:.2f}; hub CF {cf_hub:.3f} on rep. weeks vs node {cf_old:.3f}; "
                  f"TYNDP hub {hub['twh']:.1f} TWh)")

    if 'hydro' in fixes:
        shape = D['rsvr_inflow'].get('AT', {})
        ssum = sum(W[t] * shape.get(t, 0.0) for t in H)
        assert ssum > 0, 'AT reservoir inflow shape missing'
        rp_all = D.setdefault('rsvr_params', {})
        for n in nodes:
            h = S[n]['hydro']
            turb = h['res_mw'] + h['open_turb_mw'] + h['closed_turb_mw']
            if turb <= 0:
                continue
            pump = h['open_pump_mw'] + h['closed_pump_mw']
            inflow_gwh = h['res_gwh'] + max(0.0, h['open_turb_gwh'] - h['open_pump_gwh'] * ETA_PHS ** 2)
            if n == 'AT':                                   # pondage (daily storage)
                pond_mw = 1149.0
                pond_gwh = {'nt2030': 6465.0, 'de2035': 6252.0, 'de2040': 6241.0, 'de2050': 6102.0}[scenario]
                turb += pond_mw; inflow_gwh += pond_gwh
                energy = rp_all.get('AT', {}).get('energy_mwh', 2394000.0)
            else:
                energy = PHS_HOURS[n] * (h['open_turb_mw'] + h['closed_turb_mw']) + RES_HOURS * h['res_mw']
            # inflow is stored and released at eta_dis: gross it up so the model
            # delivers TYNDP's natural hydro generation (audit M1)
            inflow_gwh = inflow_gwh / ETA_PHS
            scale = inflow_gwh * 1e3 / ssum
            D['rsvr_inflow'][n] = {t: scale * shape.get(t, 0.0) for t in H}
            rp_all[n] = {'power_mw': turb, 'pump_mw': pump, 'energy_mwh': energy,
                         'eta_ch': ETA_PHS, 'eta_dis': ETA_PHS, 'opex': 1.0}
            print(f"     hydro {n}: turbine {turb/1e3:.1f} GW, pump {pump/1e3:.1f} GW, "
                  f"energy {energy/1e3:.0f} GWh, inflow {inflow_gwh/1e3:.1f} TWh")

    if 'other' in fixes:
        _add_inf_tech(D, 'oth', opex=0.0, eps=0.0, rho=1.0, mu=1.0)
        for n in nodes:
            gwh = sum(S[n]['other_gwh'].values())
            _set_exist(D, n, 'oth', gwh * 1e3 / 8760.0)
            if gwh > 0:
                print(f"     other {n}: flat {gwh*1e3/8760/1e3:.2f} GW ({gwh/1e3:.1f} TWh)")

    if 'h2' in fixes:
        price = S['_h2_price_eur_mwh']
        _add_inf_tech(D, 'h2', opex=price / H2CCGT_EFF, eps=0.0, rho=0.15, mu=0.0)
        for n in nodes:
            _set_exist(D, n, 'h2', S[n]['h2ccgt_mw'])
        D.setdefault('reserve_inf_techs', [])
        if 'h2' not in D['reserve_inf_techs']:
            D['reserve_inf_techs'].append('h2')
        print(f"     h2: H2 price {price:.1f} EUR/MWh_H2 -> {price/H2CCGT_EFF:.0f} EUR/MWh_el; "
              + ', '.join(f"{n} {S[n]['h2ccgt_mw']/1e3:.1f} GW" for n in nodes if S[n]['h2ccgt_mw']))

    if 'gas' in fixes:
        for n in nodes:
            for g in ('ccgt', 'ocgt'):
                _set_exist(D, n, g, S[n]['gas_mw'][g])
        print('     gas existing: ' + ', '.join(
            f"{n} {S[n]['gas_mw']['ccgt']/1e3:.1f}+{S[n]['gas_mw']['ocgt']/1e3:.1f} GW" for n in nodes))

    if 'pots' in fixes:
        import math
        for n in nodes:
            b = S[n]['battery']
            p = b['util_row'] if scenario == 'nt2030' else b['util_row'] / 4.0 + b['rete_row'] / 3.2
            npar = D['node_params'][n]
            cur = npar.get('k_bar_sto_p', D.get('k_bar_sto_p'))
            if p > cur:
                new = 5000.0 * math.ceil(p / 5000.0)
                ratio = npar.get('k_bar_sto_e', 4 * cur) / cur
                npar['k_bar_sto_p'] = new; npar['k_bar_sto_e'] = ratio * new
                print(f"     pots {n}: battery potential {cur/1e3:.0f} -> {new/1e3:.0f} GW "
                      f"(TYNDP fleet {p/1e3:.1f} GW)")

    if 'fuel' in fixes:
        GAS = {'nt2030': 8.845, 'de2035': 0.5 * (5.857 + 10.438), 'de2040': 10.438, 'de2050': 14.804}
        FOSSIL = {'nt2030': 0.82, 'de2035': 0.545, 'de2040': 0.35, 'de2050': 0.02}
        pg = GAS[scenario] * 3.6                       # EUR/MWh_th
        old = (D['opex']['ccgt'] + D['epsilon']['ccgt'] * D['tau'],
               D['opex']['ocgt'] + D['epsilon']['ocgt'] * D['tau'])
        for g, eta, eps in (('ccgt', 0.546, 0.37), ('ocgt', 0.367, 0.55)):
            D['opex'][g] = pg / eta
            D['epsilon'][g] = eps * FOSSIL[scenario]
        new = (D['opex']['ccgt'] + D['epsilon']['ccgt'] * D['tau'],
               D['opex']['ocgt'] + D['epsilon']['ocgt'] * D['tau'])
        rde = D.get('resid_intercon', {}).get('DE')
        if rde is not None:
            # DE rest-of-Europe import price was set to the CCGT marginal cost
            # (75 ~ 75.5 EUR/MWh, progress.md S.43); keep that link (S.105)
            rde['price_import'] = round(new[0], 1)
            print(f'     fuel: DE residual import price -> {rde["price_import"]:.1f} EUR/MWh (= CCGT cost)')
        print(f"     fuel: gas blend {GAS[scenario]:.2f} EUR/GJ, fossil {FOSSIL[scenario]:.0%}; "
              f"CCGT {old[0]:.0f} -> {new[0]:.0f}, OCGT {old[1]:.0f} -> {new[1]:.0f} EUR/MWh (fuel+CO2)")

    XB = json.load(open(_BORDERS)) if os.path.exists(_BORDERS) else {}
    if 'dsr' in fixes:
        bands = DSR_BANDS
        if scenario in XB:
            _sh = XB[scenario]['dsr']['shares']; _pr = XB[scenario]['dsr']['prices']
            bands = tuple((f'dsr{i+1}', float(_sh[i]), float(_pr[i])) for i in range(3))
            print(f"     dsr bands ({scenario}): " + ', '.join(f'{s:.0%}@{p:.0f}' for _, s, p in bands))
        for g, share, price in bands:
            _add_inf_tech(D, g, opex=price, eps=0.0, rho=1.0, mu=0.0)
            for n in nodes:
                _set_exist(D, n, g, share * S[n]['dsr_mw'])
        print('     dsr: ' + ', '.join(f"{n} {S[n]['dsr_mw']/1e3:.1f} GW" for n in nodes if S[n]['dsr_mw']))
    if 'oil' in fixes and scenario in XB:
        po = XB[scenario]['oil_price_eur_gj'] * 3.6
        _add_inf_tech(D, 'oil', opex=po / 0.35, eps=0.2668 / 0.35, rho=1.0, mu=0.0)
        for n in nodes:
            _set_exist(D, n, 'oil', XB[scenario]['oil_mw'].get(n, 0.0))
        D.setdefault('reserve_inf_techs', [])
        if 'oil' not in D['reserve_inf_techs']:
            D['reserve_inf_techs'].append('oil')
        print('     oil: ' + ', '.join(f"{n} {XB[scenario]['oil_mw'][n]/1e3:.2f} GW" for n in nodes if XB[scenario]['oil_mw'].get(n)) + f" @ {po/0.35:.0f} EUR/MWh fuel")

    if 'borders' in fixes and scenario in XB:
        med = {wi['cluster_id']: wi['medoid_week'] for wi in D['week_info']}
        for rn, rd in D.get('resid_intercon', {}).items():
            bd = XB[scenario]['borders'].get(rn)
            if bd is None:
                continue
            pr = bd['price']
            assert len(pr) == 52 * 168
            ser = {t: pr[med[t // 168] * 168 + t % 168] for t in H}
            if other_cy:
                import calendar
                mon = []
                for d_ in range(365):
                    m_ = 1
                    while d_ >= sum(calendar.monthrange(2009, k)[1] for k in range(1, m_ + 1)):
                        m_ += 1
                    mon.append(m_)
                acc = {}
                for h_, v_ in enumerate(pr):
                    key = (mon[min(h_ // 24, 364)], h_ % 24)
                    a_, c_ = acc.get(key, (0.0, 0)); acc[key] = (a_ + v_, c_ + 1)
                prof_mh = {k_: a_ / c_ for k_, (a_, c_) in acc.items()}
                ser = {}
                for t in H:
                    h_ = med[t // 168] * 168 + t % 168
                    ser[t] = prof_mh[(mon[min(h_ // 24, 364)], h_ % 24)]
            rd['price_import_series'] = {t: ser[t] + 1.0 for t in H}
            rd['price_export_series'] = {t: ser[t] - 1.0 for t in H}
            mean_rw = sum(W[t] * ser[t] for t in H) / wsum
            rd['price_import'] = mean_rw + 1.0; rd['price_export'] = mean_rw - 1.0
            print(f"     borders {rn}: {'month-by-hour TYNDP pattern (CY ' + str(cy) + ')' if other_cy else 'hourly TYNDP neighbour price'}, mean {bd['mean']:.1f} EUR/MWh (rep. weeks {mean_rw:.1f})")
    return D
