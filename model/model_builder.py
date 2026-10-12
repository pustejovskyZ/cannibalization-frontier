"""
Shared Model Builder -- Central Planner LP with Weighted Hours
==============================================================
Single source of truth for the Pyomo model. Both validate_proxy.py
and directed_search.py import build_model() from here.

Key features:
  - Weighted OPEX (representative weeks x cluster weights)
  - Cyclic SOC within each representative week
  - Ramping skips at week boundaries
  - Minimum annual H2 production constraint (forces P2G usage)
  - DSM (demand-side management) as simplified load shifting
  - EV smart charging (temporally-constrained virtual battery)
  - Multi-node support with transmission links (lossless, bidirectional)
  - Run-of-river hydro as must-take zero-cost generation (v12)
  - Pumped hydro reservoir with natural inflow (v12, AT)
  - Full conventional mix: lignite, nuclear, hard coal (v12)
"""

import pyomo.environ as pyo


def build_model(D, vre_ub):
    """
    Central planner LP with weighted representative weeks.

    Parameters:
      D:       data dict from data_pipeline.load_representative_weeks()
      vre_ub:  dict mapping VRE tech -> upper bound on K_new (MW).
               For multi-node, can be {tech: cap} (applied to all nodes)
               or {node: {tech: cap}} for per-node caps.

    Returns:
      Pyomo ConcreteModel with all constraints, ready to solve.
    """
    m = pyo.ConcreteModel('DIETER-reprweeks')
    W = D['weights']
    n_repr_weeks = D['n_repr_weeks']
    node_params = D.get('node_params', None)
    links = D.get('links', {})

    # ---- Sets ----
    m.h    = pyo.Set(initialize=D['h'])
    m.n    = pyo.Set(initialize=D['n'])
    m.VRE  = pyo.Set(initialize=D['tech_vre'])
    m.FLEX = pyo.Set(initialize=D['tech_flex'])
    m.INF  = pyo.Set(initialize=D['tech_inf'])
    m.G    = m.VRE | m.FLEX | m.INF

    # Day indices within the representative weeks (for DSM daily balance).
    n_days = n_repr_weeks * 7
    m.DAYS = pyo.RangeSet(0, n_days - 1)

    # Transmission link set (for multi-node)
    m.LINKS = pyo.Set(initialize=list(links.keys()))

    # ---- Variables ----
    m.P       = pyo.Var(m.n, m.G,   m.h, domain=pyo.NonNegativeReals)
    m.C_VRE   = pyo.Var(m.n, m.VRE, m.h, domain=pyo.NonNegativeReals)
    m.LS      = pyo.Var(m.n, m.h,         domain=pyo.NonNegativeReals)
    m.K_new   = pyo.Var(m.n, m.G,         domain=pyo.NonNegativeReals)
    m.K_sto_p = pyo.Var(m.n,              domain=pyo.NonNegativeReals)
    m.K_sto_e = pyo.Var(m.n,              domain=pyo.NonNegativeReals)
    m.K_sc    = pyo.Var(m.n,              domain=pyo.NonNegativeReals)
    m.E_ch    = pyo.Var(m.n, m.h, domain=pyo.NonNegativeReals)
    m.E_dis   = pyo.Var(m.n, m.h, domain=pyo.NonNegativeReals)
    m.SOC     = pyo.Var(m.n, m.h, domain=pyo.Reals)  # intra-period SOC (Kotzur); physical level enforced below
    m.Q_sc    = pyo.Var(m.n, m.h, domain=pyo.NonNegativeReals)

    # DSM variables
    m.DSM_up   = pyo.Var(m.n, m.h, domain=pyo.NonNegativeReals)
    m.DSM_down = pyo.Var(m.n, m.h, domain=pyo.NonNegativeReals)
    m.K_dsm    = pyo.Var(m.n, domain=pyo.NonNegativeReals)

    # EV smart charging variables
    m.EV_ch    = pyo.Var(m.n, m.h, domain=pyo.NonNegativeReals)
    m.EV_dis   = pyo.Var(m.n, m.h, domain=pyo.NonNegativeReals)
    m.EV_SOC   = pyo.Var(m.n, m.h, domain=pyo.Reals)  # intra-period (Kotzur)
    m.K_ev     = pyo.Var(m.n, domain=pyo.NonNegativeReals)

    # Transmission flow: F[l,t] > 0 means flow from +1 node to -1 node
    m.F = pyo.Var(m.LINKS, m.h, domain=pyo.Reals)

    # Endogenous NTC investment
    m.NTC_invest = pyo.Var(m.LINKS, domain=pyo.NonNegativeReals)

    # ---- DK heat-pump coupling (Tier 2 #6, optional) ---------------------
    # Activated by data_pipeline when enable_heat=True. Variables exist
    # only on the HEAT_NODES set (currently always ['DK'] when enabled).
    # Q_hp[n,t]      MW_e drawn by heat pumps (electric load on node)
    # SOC_heat[n,t]  MWh_th in thermal storage
    # Q_heat_bkp[n,t] MW_th gas/biomass backup, included with high cost so
    #                the LP only activates it when HP+storage cannot meet
    #                heat demand (prevents infeasibility from any corner).
    heat_enabled = bool(D.get('enable_heat', False)) and 'DK' in D['n']
    heat_nodes = ['DK'] if heat_enabled else []
    m.HEAT_NODES = pyo.Set(initialize=heat_nodes)
    m.Q_hp       = pyo.Var(m.HEAT_NODES, m.h, domain=pyo.NonNegativeReals)
    m.SOC_heat   = pyo.Var(m.HEAT_NODES, m.h, domain=pyo.Reals)  # intra-period (Kotzur)
    m.Q_heat_bkp = pyo.Var(m.HEAT_NODES, m.h, domain=pyo.NonNegativeReals)

    # ---- Pumped hydro reservoir variables (v12) ----
    rsvr_params = D.get('rsvr_params', {})
    rsvr_nodes = [n for n in D['n'] if n in rsvr_params]
    m.RSVR_NODES = pyo.Set(initialize=rsvr_nodes)
    m.E_rsvr_ch  = pyo.Var(m.RSVR_NODES, m.h, domain=pyo.NonNegativeReals)
    m.E_rsvr_dis = pyo.Var(m.RSVR_NODES, m.h, domain=pyo.NonNegativeReals)
    m.SOC_rsvr   = pyo.Var(m.RSVR_NODES, m.h, domain=pyo.Reals)  # intra-period (Kotzur)

    # Run-of-river data (v12): must-take zero-cost generation
    k_ror = D.get('k_ror', {})
    phi_ror = D.get('phi_ror', {})
    # Fail loudly if a node lacks RoR data: a silent 0 would drop cheap must-run
    # baseload and inflate that node's VF (audit S.81.2).
    for _n in D['n']:
        assert _n in k_ror and _n in phi_ror, (
            f"node {_n} missing run-of-river data (k_ror/phi_ror)")

    # Residual interconnection
    resid = D.get('resid_intercon', {})
    resid_nodes = list(resid.keys())
    m.RESID_NODES = pyo.Set(initialize=resid_nodes)
    m.F_resid_exp = pyo.Var(m.RESID_NODES, m.h, domain=pyo.NonNegativeReals)
    m.F_resid_imp = pyo.Var(m.RESID_NODES, m.h, domain=pyo.NonNegativeReals)
    m.NTC_resid_invest = pyo.Var(m.RESID_NODES, domain=pyo.NonNegativeReals)

    # ---- Helper: get per-node parameter ----
    def _np(node, key, default=None):
        if node_params and node in node_params and key in node_params[node]:
            return node_params[node][key]
        return D.get(key, default)

    # ---- Bounds (per-node) ----
    for n in m.n:
        node_vre_ub = _np(n, 'k_bar_vre', vre_ub)
        for v in m.VRE:
            if isinstance(vre_ub, dict) and n in vre_ub and isinstance(vre_ub[n], dict):
                m.K_new[n, v].setub(vre_ub[n][v])
            else:
                m.K_new[n, v].setub(node_vre_ub[v])
        node_flex = _np(n, 'k_bar_flex', D['k_bar_flex'])
        for f in m.FLEX:
            m.K_new[n, f].setub(node_flex[f])
        for g in m.INF:
            m.K_new[n, g].fix(0)
        m.K_sto_p[n].setub(_np(n, 'k_bar_sto_p', D['k_bar_sto_p']))
        m.K_sto_e[n].setub(_np(n, 'k_bar_sto_e', D['k_bar_sto_e']))
        m.K_sc[n].setub(_np(n, 'k_bar_sc', D.get('k_bar_sc', 200000)))
        m.K_dsm[n].setub(_np(n, 'k_bar_dsm', D.get('k_bar_dsm', 16000)))
        m.K_ev[n].setub(_np(n, 'k_bar_ev', D.get('k_bar_ev', 40000)))

    # Transmission NTC bounds
    for l_name, l_data in links.items():
        ntc_max = l_data.get('ntc_max', 50000)
        ntc_fixed = l_data['ntc_fixed']
        m.NTC_invest[l_name].setub(max(0, ntc_max - ntc_fixed))

    def ntc_flow_ub_rule(m, l, t):
        ntc_fixed = links[l]['ntc_fixed']
        return m.F[l, t] <= ntc_fixed + m.NTC_invest[l]
    m.NtcFlowUB = pyo.Constraint(m.LINKS, m.h, rule=ntc_flow_ub_rule)

    def ntc_flow_lb_rule(m, l, t):
        ntc_fixed = links[l]['ntc_fixed']
        return m.F[l, t] >= -(ntc_fixed + m.NTC_invest[l])
    m.NtcFlowLB = pyo.Constraint(m.LINKS, m.h, rule=ntc_flow_lb_rule)

    # Residual interconnection NTC bounds -- INVESTABLE (progress.md S.43):
    # the model can expand the link to non-modelled neighbours up to ntc_max.
    for rn, rd in resid.items():
        ntc_r = rd['ntc']
        resid_max = rd.get('ntc_max', ntc_r)
        m.NTC_resid_invest[rn].setub(max(0.0, resid_max - ntc_r))

    def resid_exp_ub(m, rn, t):
        return m.F_resid_exp[rn, t] <= resid[rn]['ntc'] + m.NTC_resid_invest[rn]
    m.ResidExpUB = pyo.Constraint(m.RESID_NODES, m.h, rule=resid_exp_ub)

    def resid_imp_ub(m, rn, t):
        return m.F_resid_imp[rn, t] <= resid[rn]['ntc'] + m.NTC_resid_invest[rn]
    m.ResidImpUB = pyo.Constraint(m.RESID_NODES, m.h, rule=resid_imp_ub)

    # ---- Dual suffix for shadow prices ----
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # ================================================================
    # CONSTRAINTS
    # ================================================================

    # Energy balance -- includes DSM, EV, transmission, ror, reservoir, HP
    def energy_balance_rule(m, n, t):
        net_import = sum(
            links[l]['inc'].get(n, 0) * m.F[l, t]
            for l in links if n in links[l]['inc']
        ) if links else 0
        resid_net = 0
        if n in resid:
            resid_net = m.F_resid_imp[n, t] - m.F_resid_exp[n, t]
        # Run-of-river: must-take zero-cost generation (v12)
        ror_gen = k_ror.get(n, 0) * phi_ror.get(n, {}).get(t, 0)
        # Pumped hydro reservoir (v12)
        rsvr_net = 0
        if n in rsvr_params:
            rsvr_net = m.E_rsvr_dis[n, t] - m.E_rsvr_ch[n, t]
        # Heat-pump electricity load (Tier 2 #6, only on HEAT_NODES)
        hp_load = m.Q_hp[n, t] if (heat_enabled and n in heat_nodes) else 0
        return (sum(m.P[n, g, t] for g in m.G) + m.E_dis[n, t]
                + m.LS[n, t] + m.DSM_down[n, t] + m.EV_dis[n, t]
                + net_import + resid_net + ror_gen + rsvr_net
                == D['d'][n][t] + m.E_ch[n, t] + m.Q_sc[n, t]
                + m.DSM_up[n, t] + m.EV_ch[n, t] + hp_load)
    m.EnergyBalance = pyo.Constraint(m.n, m.h, rule=energy_balance_rule)

    # VRE availability
    def vre_rule(m, n, v, t):
        if v == 'pv':
            phi = D['phi_pv'][n][t]
        elif v == 'wind_off':
            phi = D['phi_wind_off'][n][t]
        else:
            phi = D['phi_wind_on'][n][t] if 'phi_wind_on' in D else D['phi_wind'][n][t]
        k_exist = D['k_exist_vre'].get(v, 0)
        return m.P[n, v, t] + m.C_VRE[n, v, t] == phi * (k_exist + m.K_new[n, v])
    m.VreAvail = pyo.Constraint(m.n, m.VRE, m.h, rule=vre_rule)

    # Conventional capacity
    def conv_cap_rule(m, n, g, t):
        k_exist = _np(n, 'k_exist_inf', D['k_exist_inf']).get(g, 0)
        return m.P[n, g, t] <= k_exist + m.K_new[n, g]
    m.ConvCap = pyo.Constraint(m.n, m.FLEX | m.INF, m.h, rule=conv_cap_rule)

    # Ramping -- skip at week boundaries
    def ramp_up_rule(m, n, g, t):
        if t % 168 == 0:
            return pyo.Constraint.Skip
        k_exist = _np(n, 'k_exist_inf', D['k_exist_inf']).get(g, 0)
        return m.P[n, g, t] - m.P[n, g, t-1] <= D['rho'][g] * (k_exist + m.K_new[n, g])
    def ramp_down_rule(m, n, g, t):
        if t % 168 == 0:
            return pyo.Constraint.Skip
        k_exist = _np(n, 'k_exist_inf', D['k_exist_inf']).get(g, 0)
        return m.P[n, g, t-1] - m.P[n, g, t] <= D['rho'][g] * (k_exist + m.K_new[n, g])
    m.RampUp   = pyo.Constraint(m.n, m.FLEX | m.INF, m.h, rule=ramp_up_rule)
    m.RampDown = pyo.Constraint(m.n, m.FLEX | m.INF, m.h, rule=ramp_down_rule)

    # Must-run for inflexible plants
    def must_run_rule(m, n, g, t):
        k_exist = _np(n, 'k_exist_inf', D['k_exist_inf']).get(g, 0)
        return m.P[n, g, t] >= D['mu_inf'][g] * k_exist
    m.MustRun = pyo.Constraint(m.n, m.INF, m.h, rule=must_run_rule)

    # ---- OPERATING RESERVE ----
    reserve_frac = D.get('reserve_frac', 0.03)
    if reserve_frac > 0:
        def reserve_rule(m, n, t):
            demand = D['d'][n][t]
            sto_headroom = m.K_sto_p[n] - m.E_dis[n, t]
            dsm_headroom = m.K_dsm[n] - m.DSM_down[n, t]
            flex_headroom = sum(
                _np(n, 'k_exist_inf', D['k_exist_inf']).get(g, 0) + m.K_new[n, g] - m.P[n, g, t]
                for g in list(m.FLEX) + [g2 for g2 in D.get('reserve_inf_techs', []) if g2 in m.INF]
            )
            # Reservoir discharge headroom (v12)
            rsvr_headroom = 0
            if n in rsvr_params:
                rsvr_headroom = rsvr_params[n]['power_mw'] - m.E_rsvr_dis[n, t]
            return (sto_headroom + dsm_headroom + flex_headroom + rsvr_headroom
                    >= reserve_frac * demand)
        m.Reserve = pyo.Constraint(m.n, m.h, rule=reserve_rule)

    # ---- FAST-RESPONSE RESERVE (storage-only) ----
    fast_reserve_frac = D.get('fast_reserve_frac', 0.003)
    if fast_reserve_frac > 0:
        def fast_reserve_rule(m, n, t):
            demand = D['d'][n][t]
            sto_headroom = m.K_sto_p[n] - m.E_dis[n, t]
            return sto_headroom >= fast_reserve_frac * demand
        m.FastReserve = pyo.Constraint(m.n, m.h, rule=fast_reserve_rule)
        m.FastReserve_nodes = list(m.n)

    # ---- RAMPING COST for conventional plants ----
    ramp_cost = D.get('ramp_cost', None)
    if ramp_cost:
        m.Ramp_up_v   = pyo.Var(m.n, m.FLEX, m.h, domain=pyo.NonNegativeReals)
        m.Ramp_down_v = pyo.Var(m.n, m.FLEX, m.h, domain=pyo.NonNegativeReals)

        def ramp_track_up(m, n, g, t):
            if t % 168 == 0:
                return pyo.Constraint.Skip
            return m.Ramp_up_v[n, g, t] >= m.P[n, g, t] - m.P[n, g, t-1]
        def ramp_track_down(m, n, g, t):
            if t % 168 == 0:
                return pyo.Constraint.Skip
            return m.Ramp_down_v[n, g, t] >= m.P[n, g, t-1] - m.P[n, g, t]
        m.RampTrackUp   = pyo.Constraint(m.n, m.FLEX, m.h, rule=ramp_track_up)
        m.RampTrackDown = pyo.Constraint(m.n, m.FLEX, m.h, rule=ramp_track_down)

    # Storage SOC -- Kotzur et al. (2018) linked representative periods, applied
    # UNIFORMLY to every storage type (battery, reservoir, EV, heat) through the
    # helper below. Each storage's INTRA-period state resets to 0 at every
    # rep-week boundary (non-cyclic); an INTER-period state is threaded through
    # the real-year chronological order of the 52 weeks (annual-cyclic), so
    # energy crosses week boundaries and the per-week over-crediting artifact is
    # removed. Physical level  A[p,h] = (1-phi)^(h+1) * SOC_inter[p] + SOC_intra[rep,h];
    # 0 <= A <= cap enforced via per-rep intra extremes (EXACT when phi=0,
    # <= (1-(1-phi)^H) conservative otherwise).
    WL = D.get('week_labels')
    if WL is not None:
        Hk = int(D.get('repr_hours', 168))
        n_rep_k = int(D.get('n_repr_weeks', len(m.h) // Hk))
        P_k = len(WL)
        m.REP = pyo.Set(initialize=range(n_rep_k))
        m.PER = pyo.Set(initialize=range(P_k))

        def _kotzur_link(soc, flow_fn, cap_fn, nodes, phi, tag):
            """Linked-period dynamics on intra-period state soc[n,t] (Reals).
            flow_fn(n,t): net stored-energy inflow at hour t (charge-discharge[+inflow]).
            cap_fn(n): energy-capacity expression (upper bound on physical level).
            phi: per-hour self-discharge (0 = lossless). tag: unique name suffix."""
            nodes = list(nodes)
            if not nodes:
                return
            phiH = (1.0 - phi) ** Hk
            NS = pyo.Set(initialize=nodes); m.add_component('KN_' + tag, NS)
            inter = pyo.Var(NS, m.PER, domain=pyo.NonNegativeReals); m.add_component('SOCinter_' + tag, inter)
            hi = pyo.Var(NS, m.REP, domain=pyo.Reals); m.add_component('SOChi_' + tag, hi)
            lo = pyo.Var(NS, m.REP, domain=pyo.Reals); m.add_component('SOClo_' + tag, lo)
            def dyn(mm, n, t):
                f = flow_fn(n, t)
                return soc[n, t] == (f if t % Hk == 0 else (1.0 - phi) * soc[n, t - 1] + f)
            def hir(mm, n, t):
                return hi[n, t // Hk] >= soc[n, t]
            def lor(mm, n, t):
                return lo[n, t // Hk] <= soc[n, t]
            def interdyn(mm, n, p):
                return inter[n, (p + 1) % P_k] == phiH * inter[n, p] + soc[n, WL[p] * Hk + (Hk - 1)]
            def ub(mm, n, p):
                return inter[n, p] + hi[n, WL[p]] <= cap_fn(n)
            def lb(mm, n, p):
                return phiH * inter[n, p] + lo[n, WL[p]] >= 0
            m.add_component('SocDyn_' + tag, pyo.Constraint(NS, m.h, rule=dyn))
            m.add_component('SocHi_' + tag, pyo.Constraint(NS, m.h, rule=hir))
            m.add_component('SocLo_' + tag, pyo.Constraint(NS, m.h, rule=lor))
            m.add_component('SocInter_' + tag, pyo.Constraint(NS, m.PER, rule=interdyn))
            m.add_component('SocUB_' + tag, pyo.Constraint(NS, m.PER, rule=ub))
            m.add_component('SocLB_' + tag, pyo.Constraint(NS, m.PER, rule=lb))

        # battery (self-discharge phi_s; energy capacity is a decision var)
        _kotzur_link(m.SOC,
                     lambda n, t: D['eta_ch'] * m.E_ch[n, t] - (1.0 / D['eta_dis']) * m.E_dis[n, t],
                     lambda n: m.K_sto_e[n], list(m.n), D['phi_s'], 'bat')
    else:
        # Fallback: legacy per-week-cyclic battery SOC (no inter-week linking)
        def soc_rule(m, n, t):
            t_prev = t + 167 if t % 168 == 0 else t - 1
            return (m.SOC[n, t] == (1 - D['phi_s']) * m.SOC[n, t_prev]
                    + D['eta_ch'] * m.E_ch[n, t] - (1 / D['eta_dis']) * m.E_dis[n, t])
        m.SocDyn = pyo.Constraint(m.n, m.h, rule=soc_rule)
        def soc_bound_rule(m, n, t):
            return m.SOC[n, t] <= m.K_sto_e[n]
        m.SocBound = pyo.Constraint(m.n, m.h, rule=soc_bound_rule)

    def ch_bound_rule(m, n, t):
        return m.E_ch[n, t] <= m.K_sto_p[n]
    m.ChBound = pyo.Constraint(m.n, m.h, rule=ch_bound_rule)

    def dis_bound_rule(m, n, t):
        return m.E_dis[n, t] <= m.K_sto_p[n]
    m.DisBound = pyo.Constraint(m.n, m.h, rule=dis_bound_rule)

    # ================================================================
    # PUMPED HYDRO RESERVOIR CONSTRAINTS (v12)
    # ================================================================
    rsvr_inflow = D.get('rsvr_inflow', {})

    if rsvr_nodes:
        for rn in rsvr_nodes:
            rp = rsvr_params[rn]
            for t in m.h:
                m.E_rsvr_ch[rn, t].setub(rp.get('pump_mw', rp['power_mw']))
                m.E_rsvr_dis[rn, t].setub(rp['power_mw'])
                # SOC_rsvr upper bound now via the Kotzur level bound

        if WL is not None:
            def _rsvr_flow(n, t):
                rp = rsvr_params[n]
                return (rp['eta_ch'] * m.E_rsvr_ch[n, t]
                        - (1.0 / rp['eta_dis']) * m.E_rsvr_dis[n, t]
                        + rsvr_inflow.get(n, {}).get(t, 0))
            _kotzur_link(m.SOC_rsvr, _rsvr_flow,
                         lambda n: rsvr_params[n]['energy_mwh'], rsvr_nodes, 0.0, 'rsvr')
        else:
            def rsvr_soc_rule(m, n, t):
                rp = rsvr_params[n]
                t_prev = t + 167 if t % 168 == 0 else t - 1
                inflow = rsvr_inflow.get(n, {}).get(t, 0)
                return (m.SOC_rsvr[n, t] == m.SOC_rsvr[n, t_prev]
                        + rp['eta_ch'] * m.E_rsvr_ch[n, t]
                        - (1.0 / rp['eta_dis']) * m.E_rsvr_dis[n, t] + inflow)
            m.RsvrSocDyn = pyo.Constraint(m.RSVR_NODES, m.h, rule=rsvr_soc_rule)

    # Sector coupling capacity
    def sc_cap_rule(m, n, t):
        return m.Q_sc[n, t] <= m.K_sc[n]
    m.ScCap = pyo.Constraint(m.n, m.h, rule=sc_cap_rule)

    # ---- H2 DEMAND CONSTRAINT ----
    # Generalised per-node quota (S.61): D['h2_targets'] = {node: MWh_H2}
    # (set by --h2-2040 to TYNDP DE2040 electrolyser levels x sc_eta). When
    # present, build ONE always-on per-node quota and SKIP the legacy DE/DK
    # blocks; else fall back to legacy DE (sc_h2_target) + DK.
    h2_targets = D.get('h2_targets')
    if h2_targets:
        def h2_node_rule(m, n):
            tgt = float(h2_targets.get(n, 0) or 0)
            if tgt <= 0:
                return pyo.Constraint.Skip
            return (sum(W[t] * D['sc_eta'] * m.Q_sc[n, t] for t in m.h) >= tgt)
        m.H2TargetNode = pyo.Constraint(m.n, rule=h2_node_rule)

    # ---- legacy DE primary-node H2 quota (only when no generalized targets) ----
    sc_h2_target = D.get('sc_h2_target', 0)
    primary_node = D['n'][0]  # DE
    def h2_target_rule(m, n):
        if n != primary_node:
            return pyo.Constraint.Skip
        return (sum(W[t] * D['sc_eta'] * m.Q_sc[n, t] for t in m.h)
                >= sc_h2_target)
    if not h2_targets:
        m.H2Target = pyo.Constraint(m.n, rule=h2_target_rule)
        if sc_h2_target <= 0:
            m.H2Target.deactivate()

    # ---- DK-specific H2 quota (Tier 1 #1) ----
    # Parallel to H2Target but anchored on DK P2G output. Toggled on/off
    # at solve time by LPOracle based on whether DK P2G capacity is large
    # enough to satisfy the quota (mirrors the existing DE toggle pattern
    # in directed_search.py:LPOracle.__call__). The constraint is built
    # unconditionally so the toggle has something to activate; if the
    # parameter is zero we deactivate it here at build time.
    sc_h2_target_dk = D.get('sc_h2_target_dk', 0)
    def h2_target_dk_rule(m, n):
        if n != 'DK':
            return pyo.Constraint.Skip
        return (sum(W[t] * D['sc_eta'] * m.Q_sc[n, t] for t in m.h)
                >= sc_h2_target_dk)
    if 'DK' in D['n'] and not h2_targets:
        m.H2Target_DK = pyo.Constraint(m.n, rule=h2_target_dk_rule)
        if sc_h2_target_dk <= 0:
            m.H2Target_DK.deactivate()

    # ---- DK HEAT-PUMP COUPLING CONSTRAINTS (Tier 2 #6) ---------------
    # When enable_heat=True these constraints make Q_hp[n,t] (electric MW)
    # convert to COP*Q_hp MW_th of heat, charge or supplement a thermal
    # storage SOC_heat[n,t], and meet a fixed hourly heat_demand profile.
    #
    #   COP * Q_hp[n,t] + SOC_heat[n,t-1] - SOC_heat[n,t] + Q_heat_bkp
    #     >= heat_demand[n,t]
    #
    # The thermal storage is cyclic within each representative week (same
    # convention as electric storage). Q_heat_bkp is a high-cost slack
    # representing legacy gas/biomass boilers; the LP only activates it
    # when HP capacity + storage cannot meet a peak demand hour.
    if heat_enabled:
        K_hp_dk            = float(D.get('dk_hp_mw', 2000.0))
        K_heat_storage_dk  = float(D.get('dk_heat_storage_mwh', 30000.0))
        cop_hp             = float(D.get('dk_cop', 3.0))
        heat_demand_dk     = D.get('heat_demand_dk', {})

        # HP electric capacity (fixed)
        def hp_cap_rule(m, n, t):
            return m.Q_hp[n, t] <= K_hp_dk
        m.HpCap = pyo.Constraint(m.HEAT_NODES, m.h, rule=hp_cap_rule)

        phi_s_heat = 0.0003
        if WL is not None:
            m.Heat_spill = pyo.Var(m.HEAT_NODES, m.h, domain=pyo.NonNegativeReals)
            _kotzur_link(
                m.SOC_heat,
                lambda n, t: (cop_hp * m.Q_hp[n, t] - heat_demand_dk.get(t, 0.0)
                              + m.Q_heat_bkp[n, t] - m.Heat_spill[n, t]),
                lambda n: K_heat_storage_dk, heat_nodes, phi_s_heat, 'heat')
        else:
            def heat_soc_cap_rule(m, n, t):
                return m.SOC_heat[n, t] <= K_heat_storage_dk
            m.HeatSocCap = pyo.Constraint(m.HEAT_NODES, m.h, rule=heat_soc_cap_rule)
            def heat_soc_dyn_rule(m, n, t):
                t_prev = t + 167 if t % 168 == 0 else t - 1
                return (m.SOC_heat[n, t] <= (1 - phi_s_heat) * m.SOC_heat[n, t_prev]
                        + cop_hp * m.Q_hp[n, t] - heat_demand_dk.get(t, 0.0) + m.Q_heat_bkp[n, t])
            m.HeatSocDyn = pyo.Constraint(m.HEAT_NODES, m.h, rule=heat_soc_dyn_rule)

    # ================================================================
    # DSM CONSTRAINTS
    # ================================================================
    dsm_eta = D.get('dsm_eta', 0.95)

    def dsm_up_cap_rule(m, n, t):
        return m.DSM_up[n, t] <= m.K_dsm[n]
    m.DsmUpCap = pyo.Constraint(m.n, m.h, rule=dsm_up_cap_rule)

    def dsm_down_cap_rule(m, n, t):
        return m.DSM_down[n, t] <= m.K_dsm[n]
    m.DsmDownCap = pyo.Constraint(m.n, m.h, rule=dsm_down_cap_rule)

    def dsm_daily_balance_rule(m, n, day):
        day_start = day * 24
        day_hours = [t for t in range(day_start, day_start + 24) if t in D['h_set']]
        if not day_hours:
            return pyo.Constraint.Skip
        return (sum(m.DSM_down[n, t] for t in day_hours)
                == dsm_eta * sum(m.DSM_up[n, t] for t in day_hours))
    m.DsmDailyBal = pyo.Constraint(m.n, m.DAYS, rule=dsm_daily_balance_rule)

    # ================================================================
    # EV SMART CHARGING CONSTRAINTS
    # ================================================================
    ev_avail_all = D.get('ev_avail', {})
    ev_eta_ch = D.get('ev_eta_ch', 0.96)
    ev_eta_dis = D.get('ev_eta_dis', 0.96)
    ev_ep_ratio = D.get('ev_ep_ratio', 4.0)

    has_ev = any(ev_avail_all.get(n) is not None for n in D['n'])

    # ---- EV 2040 realistic representation (S.63): fixed fleet + driving sink ----
    # When D['ev_2040'] is set, EV is a FIXED 2040 passenger fleet (K_ev power,
    # ev_E energy, both fixed by LPOracle), with a DRIVING-demand sink and
    # plug-in availability from TYNDP. SOC must stay in [minSOC*E, E]; charging
    # over the (per-rep-week-cyclic) week must cover driving. V2G allowed.
    if D.get('ev_2040'):
        ev_av = D.get('ev_avail2040', {})
        ev_dr = D.get('ev_drive', {})
        ev_E = D.get('ev_E', {})
        ev_eta2 = D.get('ev_eta', 0.94)
        minsoc = D.get('ev_minsoc', 0.20)
        ev_nodes = [n for n in D['n'] if n in ev_av]
        def ev2_ch_cap(m, n, t):
            if n not in ev_av:
                return m.EV_ch[n, t] <= 0
            return m.EV_ch[n, t] <= ev_av[n].get(t, 0.0) * m.K_ev[n]
        m.EvChCap = pyo.Constraint(m.n, m.h, rule=ev2_ch_cap)
        def ev2_dis_cap(m, n, t):
            if n not in ev_av:
                return m.EV_dis[n, t] <= 0
            return m.EV_dis[n, t] <= ev_av[n].get(t, 0.0) * m.K_ev[n]
        m.EvDisCap = pyo.Constraint(m.n, m.h, rule=ev2_dis_cap)
        # per-rep-week cyclic SOC with driving sink (EV is a daily asset; no
        # seasonal transfer needed -> cyclic within each rep week is exact)
        H2 = int(D.get('repr_hours', 168))
        def ev2_soc(m, n, t):
            if n not in ev_av:
                return pyo.Constraint.Skip
            t_prev = t + (H2 - 1) if t % H2 == 0 else t - 1
            return (m.EV_SOC[n, t] == m.EV_SOC[n, t_prev]
                    + ev_eta2 * m.EV_ch[n, t] - (1.0 / ev_eta2) * m.EV_dis[n, t]
                    - ev_dr.get(n, {}).get(t, 0.0))
        m.EvSocDyn = pyo.Constraint(m.n, m.h, rule=ev2_soc)
        def ev2_soc_ub(m, n, t):
            if n not in ev_av:
                return pyo.Constraint.Skip
            return m.EV_SOC[n, t] <= ev_E.get(n, 0.0)
        m.EvSocBound = pyo.Constraint(m.n, m.h, rule=ev2_soc_ub)
        def ev2_soc_lb(m, n, t):
            if n not in ev_av:
                return pyo.Constraint.Skip
            return m.EV_SOC[n, t] >= minsoc * ev_E.get(n, 0.0)
        m.EvSocLB = pyo.Constraint(m.n, m.h, rule=ev2_soc_lb)
    elif has_ev:
        def ev_ch_cap_rule(m, n, t):
            avail = ev_avail_all.get(n, {}).get(t, 0.0)
            return m.EV_ch[n, t] <= avail * m.K_ev[n]
        m.EvChCap = pyo.Constraint(m.n, m.h, rule=ev_ch_cap_rule)

        def ev_dis_cap_rule(m, n, t):
            avail = ev_avail_all.get(n, {}).get(t, 0.0)
            return m.EV_dis[n, t] <= avail * m.K_ev[n]
        m.EvDisCap = pyo.Constraint(m.n, m.h, rule=ev_dis_cap_rule)

        if WL is not None:
            _kotzur_link(m.EV_SOC,
                         lambda n, t: ev_eta_ch * m.EV_ch[n, t] - (1.0 / ev_eta_dis) * m.EV_dis[n, t],
                         lambda n: ev_ep_ratio * m.K_ev[n], list(m.n), 0.0, 'ev')
        else:
            def ev_soc_rule(m, n, t):
                t_prev = t + 167 if t % 168 == 0 else t - 1
                return (m.EV_SOC[n, t] == m.EV_SOC[n, t_prev]
                        + ev_eta_ch * m.EV_ch[n, t] - (1.0 / ev_eta_dis) * m.EV_dis[n, t])
            m.EvSocDyn = pyo.Constraint(m.n, m.h, rule=ev_soc_rule)
            def ev_soc_bound_rule(m, n, t):
                return m.EV_SOC[n, t] <= ev_ep_ratio * m.K_ev[n]
            m.EvSocBound = pyo.Constraint(m.n, m.h, rule=ev_soc_bound_rule)
    else:
        for n in D['n']:
            m.K_ev[n].fix(0)

    # ================================================================
    # OBJECTIVE: CAPEX (annual) + weighted OPEX
    # ================================================================
    def obj_rule(m):
        # Annualized capex + fixed O&M (DEA 2040; FOM defaults to 0 for the
        # legacy scenario). For fixed VRE capacity the FOM term is a constant
        # shift (cancels in the frontier premium) but keeps sys_cost a
        # complete annual system cost; for investable techs (gas, storage,
        # P2G) it correctly prices total fixed annual cost.
        FM = D.get('fom', {})
        inv_gen = sum((D['capex'][g] + FM.get(g, 0)) * m.K_new[n, g]
                      for n in m.n for g in m.VRE | m.FLEX)
        inv_sto = sum((D['sto_capex_p'] + D.get('sto_fom_p', 0)) * m.K_sto_p[n]
                      + (D['sto_capex_e'] + D.get('sto_fom_e', 0)) * m.K_sto_e[n] for n in m.n)
        inv_sc  = sum((D['sc_capex'] + D.get('sc_fom', 0)) * m.K_sc[n] for n in m.n)
        inv_dsm = sum(D.get('dsm_capex', 10000) * m.K_dsm[n] for n in m.n)
        inv_ev  = sum(D.get('ev_capex', 4000) * m.K_ev[n] for n in m.n)

        # NTC investment cost: annualized capex x distance
        inv_ntc = 0
        for l_name, l_data in links.items():
            capex_mw_km = l_data.get('capex_per_mw_km_yr', 0)
            dist_km = l_data.get('dist_km', 0)
            inv_ntc += capex_mw_km * dist_km * m.NTC_invest[l_name]
        inv_resid_ntc = 0
        for rn, rd in resid.items():
            inv_resid_ntc += (rd.get('capex_per_mw_km_yr', 0) * rd.get('dist_km', 0)
                              * m.NTC_resid_invest[rn])

        gen_cost = sum(W[t] * (D['opex'][g] + D['epsilon'][g]*D['tau']) * m.P[n, g, t]
                       for n in m.n for g in m.G for t in m.h)
        sto_cost = sum(W[t] * (D['sto_c_ch'] * m.E_ch[n, t] + D['sto_c_dis'] * m.E_dis[n, t])
                       for n in m.n for t in m.h)
        sc_cost  = sum(W[t] * D['sc_opex'] * m.Q_sc[n, t] for n in m.n for t in m.h)
        dsm_cost = sum(W[t] * D.get('dsm_opex', 10.0) * (m.DSM_up[n, t] + m.DSM_down[n, t])
                       for n in m.n for t in m.h)
        ev_cost  = sum(W[t] * D.get('ev_opex', 5.0) * (m.EV_ch[n, t] + m.EV_dis[n, t])
                       for n in m.n for t in m.h)
        curt_cost = sum(W[t] * D['c_curt'][v] * m.C_VRE[n, v, t]
                        for n in m.n for v in m.VRE for t in m.h)
        ls_cost  = sum(W[t] * D['v_ls'] * m.LS[n, t] for n in m.n for t in m.h)

        # Ramping cost
        ramp_cost_d = D.get('ramp_cost', None)
        if ramp_cost_d:
            ramp_total = sum(
                W[t] * ramp_cost_d.get(g, 0) * (m.Ramp_up_v[n, g, t] + m.Ramp_down_v[n, g, t])
                for n in m.n for g in m.FLEX for t in m.h
                if t % 168 != 0
            )
        else:
            ramp_total = 0

        # Reservoir operating cost (v12)
        rsvr_cost = 0
        for rn in rsvr_nodes:
            rsvr_opex = rsvr_params[rn].get('opex', 1.0)
            rsvr_cost += sum(
                W[t] * rsvr_opex * (m.E_rsvr_ch[rn, t] + m.E_rsvr_dis[rn, t])
                for t in m.h
            )

        # Residual interconnection cost
        resid_cost = 0
        for rn, rd in resid.items():
            resid_cost += sum(
                W[t] * (rd.get('price_import_series', {}).get(t, rd['price_import']) * m.F_resid_imp[rn, t]
                        - rd.get('price_export_series', {}).get(t, rd['price_export']) * m.F_resid_exp[rn, t])
                for t in m.h
            )

        # Tier 2 #6 heat-coupling cost terms.
        #
        # heat_bkp_price = 80 EUR/MWh_th. Default backup is a gas/biomass
        # boiler. EU TTF wholesale gas Q4 2024 ~30 EUR/MWh, divided by
        # 92 % boiler efficiency = 33 EUR/MWh_th. Plus ETS-1 carbon at
        # ~80 EUR/tCO2 × 0.20 tCO2/MWh_gas = 16 EUR/MWh, gives ~50
        # EUR/MWh_th raw. 80 EUR/MWh_th is set ~1.6× this to ensure the
        # LP only uses backup as a last-resort feasibility slack.
        # Source: Eurostat IRT_NRG_PC_205 gas prices Q4 2024;
        # EUA carbon price (EEX EU ETS daily, 2025).
        #
        # hp_opex_price = 2.0 EUR/MWh_e. Variable O&M for large
        # district-heating heat pumps. Source: DEA Technology Catalogue
        # 40 "Large Heat Pumps, air to water" (June 2022 update),
        # variable O&M = 1.6-2.5 EUR/MWh_e. 2.0 is the midpoint AND
        # exceeds VRE curtailment cost (0.5 EUR/MWh) by enough to break
        # the LP's tie-break degeneracy during VRE-surplus hours.
        heat_bkp_cost = 0
        hp_opex_cost = 0
        if heat_enabled:
            heat_bkp_price = float(D.get('heat_bkp_price', 80.0))
            hp_opex_price  = float(D.get('hp_opex', 2.0))
            heat_bkp_cost = sum(W[t] * heat_bkp_price * m.Q_heat_bkp[n, t]
                                for n in m.HEAT_NODES for t in m.h)
            hp_opex_cost  = sum(W[t] * hp_opex_price * m.Q_hp[n, t]
                                for n in m.HEAT_NODES for t in m.h)

        return (inv_gen + inv_sto + inv_sc + inv_dsm + inv_ev + inv_ntc + inv_resid_ntc
                + gen_cost + sto_cost + sc_cost + dsm_cost + ev_cost + curt_cost + ls_cost
                + ramp_total + resid_cost + rsvr_cost
                + heat_bkp_cost + hp_opex_cost)
    m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return m
