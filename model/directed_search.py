"""
Directed Search v12 — Least-cost VF frontier via CMA-ES / scipy.optimize
=========================================================================
Uses shared model_builder.py and data_pipeline.py.

v12 changes: 5-node topology (DE+CZ+DK+AT+NL), run-of-river,
  pumped hydro reservoir (AT), full conventional mix (lig+nuc+hc),
  5 NTC links (13D outer search), distance-based NTC costs.

v11 changes (from v10):
  - **Endogenous NTC investment**: transmission capacity is now a decision
    variable in both the LP (model_builder) and the outer optimizer.
    Each link gets one dimension in the outer search (NTC_invest in MW).
    The LP co-optimizes: flow ≤ ntc_fixed + NTC_invest, with annualized
    capex × distance in the objective.
  - **Full 4D outer search for high-VRE secondary nodes** (e.g. DK).
    Replaces the v10 scalar budget + priority-filling mechanism. DK now
    has 4 independent dimensions (sto, P2G, DSM, EV) controlled by the
    outer optimizer, same as DE. CZ remains LP-endogenous (low VRE).
  - **Expanded output recording**: per-node capacities, NTC investment,
    and per-link flows are saved to the frontier CSV.
  - v11 dimension count: DE(4) + DK(4) + NTC(2 links) = 10D for DE+CZ+DK.
  - v12 dimension count: DE(4) + DK(4) + NTC(5 links) = 13D for 5-node.
    CMA-ES popsize ~22 for 13D; eval budget 500/600.

v10 changes (from v9):
  - CMA-ES optimizer replaces Nelder-Mead as default outer search.
  - FastReserve constraint per-node activation/deactivation (B9 bug fix).
  - DK scalar budget capped at 16 GW (replaced in v11).

Usage:
  pip install pandas pyomo matplotlib numpy highspy scipy cma
  python directed_search.py
  python directed_search.py --optimizer nelder-mead  # fallback
"""

import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
try:
    from notify import notify
except Exception:
    def notify(*a, **k):
        return False
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_representative_weeks
from model_builder import build_model


# ============================================================================
# 1. LP ORACLE — N-dimensional, returns None on infeasibility
# ============================================================================

class LPOracle:
    """
    Wraps the Pyomo model as a black-box function:
      x = [DE_sto, DE_sc, DE_dsm, DE_ev, DK_sto, DK_sc, DK_dsm, DK_ev,
           NTC_DE_CZ, NTC_DE_DK]
      →  {sys_cost, vf_wind_on, vf_pv, ...}

    v11 dimension layout:
      - Primary node (DE): 4 dims (sto, P2G, DSM, EV) — always present
      - High-VRE secondary nodes (e.g. DK): 4 dims each (sto, P2G, DSM, EV)
      - Low-VRE secondary nodes (e.g. CZ): LP-endogenous (0 dims)
      - NTC investment: 1 dim per transmission link

    For DE+CZ+DK with 2 links: 4 + 4 + 2 = 10 dimensions.
    """

    # Per-node flex instrument names (in order within each node's 4D block)
    FLEX_NAMES = ['sto', 'sc', 'dsm', 'ev']

    def __init__(self, D, verbose=True):
        """Build an LPOracle. verbose=False suppresses all init-time
        print banners; spawned workers pass verbose=False to keep the
        log readable (otherwise each worker re-prints the full solver
        + NTC + VRE banner)."""
        self.D = D
        self.verbose = verbose
        self.all_nodes = D['n']
        # P2G dimensionality option (S.59.2/S.60): when True, electrolyser
        # capacity K_sc is NOT an outer-search dimension; it is left
        # LP-endogenous (floored by the H2 quota), since P2G does not raise
        # VF (grid-floor evidence) -- removes the per-node sc dims (26->21).
        # Default False = keep P2G in the outer search (legacy 26-dim).
        self.p2g_endogenous = bool(D.get('p2g_endogenous', False))
        # EV-2040 (S.63): EV is a FIXED exogenous 2040 fleet (driving load +
        # V2G flexibility), not a decision/lever -> dropped from outer search
        # and K_ev fixed at the TYNDP fleet power.
        self.ev_2040 = bool(D.get('ev_2040', False))
        # PERSISTENT solver (APPSI): build the Gurobi model ONCE and re-solve
        # incrementally (only changed bounds/constraints) -> eliminates the
        # ~12 s/solve Pyomo->Gurobi re-export that gurobi_direct paid every solve
        # (confirmed: Gurobi solve 4.4 s vs 16.4 s wall). [S.82]
        from pyomo.contrib.appsi.solvers import Gurobi as _AppsiGurobi
        from pyomo.contrib.appsi.solvers import Highs as _AppsiHighs
        from pyomo.contrib.appsi.base import TerminationCondition as _ATC
        self._ATC = _ATC
        import multiprocessing as _mpc
        _is_worker = (_mpc.current_process().name != 'MainProcess')
        self.solver = None
        self._solver_kind = None
        try:
            _g = _AppsiGurobi()
            if _g.available():
                self.solver = _g
                self._solver_kind = 'gurobi'
                if verbose:
                    print("  Solver: Gurobi (APPSI persistent)")
        except Exception:
            self.solver = None
        if self.solver is None:
            # A WORKER that cannot get Gurobi must NOT silently grind on slow
            # HiGHS (that burned a ~23 h run). Fail loudly so the pool surfaces
            # it. (Main process may still use HiGHS for sandbox / validation.)
            if _is_worker and os.environ.get('ALLOW_WORKER_HIGHS') != '1':
                raise RuntimeError(
                    "Worker could not acquire Gurobi (offline license not reaching "
                    "this spawned process?). Refusing to fall back to slow HiGHS "
                    "silently -- fix the license or set ALLOW_WORKER_HIGHS=1.")
            self.solver = _AppsiHighs()
            self._solver_kind = 'highs'
            print("  [WARN] Gurobi unavailable -> HiGHS (APPSI; MUCH slower). "
                  "Check that gurobi.lic is valid.", flush=True)
        try:
            self.solver.config.load_solution = True
        except Exception:
            pass
        # per-solve time-limit safety cap; threads only if LP_SOLVER_THREADS set
        # (default = all cores -> fast parallel barrier).
        _tl = float(os.environ.get('SOLVE_TIME_LIMIT_S', '900'))
        _thr_env = os.environ.get('LP_SOLVER_THREADS')
        try:
            if self._solver_kind == 'gurobi':
                self.solver.gurobi_options['TimeLimit'] = _tl
                self.solver.gurobi_options['QCPDual'] = 1   # match legacy dual behavior
                # optional perf knobs for the hard 12-week LPs (env-controlled,
                # off by default): GUROBI_METHOD (2=barrier-only, avoids running
                # 3 concurrent methods), GUROBI_NUMERICFOCUS, GUROBI_BARHOMOGENEOUS,
                # GUROBI_PRESOLVE.
                for _ge, _gp in (('GUROBI_METHOD', 'Method'),
                                 ('GUROBI_NUMERICFOCUS', 'NumericFocus'),
                                 ('GUROBI_BARHOMOGENEOUS', 'BarHomogeneous'),
                                 ('GUROBI_PRESOLVE', 'Presolve')):
                    _gv = os.environ.get(_ge)
                    if _gv is not None:
                        try:
                            self.solver.gurobi_options[_gp] = int(_gv)
                        except Exception:
                            pass
                if _thr_env is not None:
                    self.solver.gurobi_options['Threads'] = int(_thr_env)
            else:
                self.solver.highs_options['time_limit'] = _tl
                if _thr_env is not None:
                    self.solver.highs_options['threads'] = int(_thr_env)
        except Exception:
            pass
        # APPSI re-scans the WHOLE model for structural changes every solve.
        # We only fix/unfix vars + toggle/add/remove constraints -- never add or
        # remove vars/params, no mutable params, no named expressions, never
        # replace the objective (verified). Disable those redundant scans: the
        # 210k-component re-scan was the bulk of the per-solve "overhead". [S.82.2]
        try:
            _uc = self.solver.update_config
            _uc.check_for_new_or_removed_vars = False
            _uc.check_for_new_or_removed_params = False
            _uc.update_params = False
            _uc.update_named_expressions = False
            _uc.check_for_new_objective = False
        except Exception:
            pass
        self.model = build_model(D, D['k_bar_vre'])
        # Energy-balance constraints whose duals (= nodal prices) the VF metric
        # needs; precomputed so each solve loads only these into the dual Suffix.
        self._eb_cons = [self.model.EnergyBalance[idx]
                         for idx in self.model.EnergyBalance]
        # Reduced costs of FIXED variables = exact dZ*/dx (LP sensitivity).
        # Imported for the first-order optimizer-honesty check (S.56); if the
        # solver interface doesn't populate it, the check degrades gracefully.
        try:
            self.model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        except Exception:
            pass
        self.n_node = D['n'][0]  # primary node (DE)
        self.W = D['weights']

        node_params = D.get('node_params', {})
        links = D.get('links', {})

        # ---- Build dimension layout ----
        # Each entry: (dim_name, upper_bound, node_or_link, instrument_type)
        self.bounds = []
        self.x_keys = []
        self.dim_info = []  # metadata for each dimension

        # --- Primary node (DE): always 4D ---
        de_params = node_params.get('DE', {})
        de_flex = [
            ('sto_mw', de_params.get('k_bar_sto_p', D['k_bar_sto_p']), 'DE', 'sto'),
            ('sc_mw',  100000, 'DE', 'sc'),  # P2G: 100 GW cap
            ('dsm_mw', de_params.get('k_bar_dsm', D.get('k_bar_dsm', 50000)), 'DE', 'dsm'),
            ('ev_mw',  de_params.get('k_bar_ev', D.get('k_bar_ev', 20000)), 'DE', 'ev'),
        ]
        if self.p2g_endogenous:
            de_flex = [t for t in de_flex if t[3] != 'sc']
        if self.ev_2040:
            de_flex = [t for t in de_flex if t[3] != 'ev']
        for name, ub, node, instr in de_flex:
            self.bounds.append(ub)
            self.x_keys.append(name)
            self.dim_info.append({'node': node, 'instr': instr, 'type': 'flex'})

        # --- Secondary nodes: full 4D for high-VRE, LP-endogenous for low-VRE ---
        self.outer_search_nodes = [self.n_node]  # nodes controlled by outer search
        self.lp_endogenous_nodes = []  # nodes left free in the LP

        for n in self.all_nodes:
            if n == self.n_node:
                continue
            n_params = node_params.get(n, {})
            n_vre = n_params.get('k_bar_vre', {})
            avg_demand = sum(D['d'][n][t] for t in D['h']) / len(D['h'])
            total_vre_fixed = sum(int(v) for v in n_vre.values())  # real fixed VRE (S.39: 2/3 rule removed)
            pen_ratio = total_vre_fixed / max(avg_demand, 1)

            if pen_ratio > 1.0:
                # Any node whose VRE exceeds local demand is cannibalisation-
                # prone and its VF enters the system VF_min. Such a node MUST be
                # outer-search so its flexibility is a controllable lever; if it
                # stays LP-endogenous the LP only sizes flex for cost (not VF),
                # so the node's VF is an UNRAISABLE FLOOR that caps the whole
                # frontier (progress.md S.46: this is what pinned VF_min~0.68 via
                # CZ). Threshold lowered 2.0 -> 1.0 so CZ becomes outer-search.
                # High-VRE node: add 4 dimensions to outer search
                self.outer_search_nodes.append(n)
                n_lc = n.lower()
                node_flex = [
                    (f'sto_{n_lc}_mw', n_params.get('k_bar_sto_p', 0), n, 'sto'),
                    (f'sc_{n_lc}_mw',  n_params.get('k_bar_sc', 0), n, 'sc'),
                    (f'dsm_{n_lc}_mw', n_params.get('k_bar_dsm', 0), n, 'dsm'),
                    (f'ev_{n_lc}_mw',  n_params.get('k_bar_ev', 0), n, 'ev'),
                ]
                if self.p2g_endogenous:
                    node_flex = [t for t in node_flex if t[3] != 'sc']
                if self.ev_2040:
                    node_flex = [t for t in node_flex if t[3] != 'ev']
                for name, ub, node, instr in node_flex:
                    self.bounds.append(ub)
                    self.x_keys.append(name)
                    self.dim_info.append({'node': node, 'instr': instr, 'type': 'flex'})
                if verbose:
                    print(f"  {n}: high-VRE node (ratio={pen_ratio:.1f}×), "
                          f"full 4D outer search (sto/P2G/DSM/EV)")
            else:
                self.lp_endogenous_nodes.append(n)
                if verbose:
                    print(f"  {n}: low-VRE node (ratio={pen_ratio:.1f}×), LP-endogenous flexibility")

        # --- NTC investment dimensions: one per link ---
        self.ntc_link_names = []  # ordered link names
        for l_name, l_data in links.items():
            ntc_fixed = l_data['ntc_fixed']
            ntc_max = l_data.get('ntc_max', 50000)
            ntc_invest_max = max(0, ntc_max - ntc_fixed)
            self.bounds.append(ntc_invest_max)
            dim_name = f'ntc_{l_name.lower()}_mw'
            self.x_keys.append(dim_name)
            self.dim_info.append({'link': l_name, 'type': 'ntc'})
            self.ntc_link_names.append(l_name)
            if verbose:
                print(f"  NTC {l_name}: outer search dim, "
                      f"fixed={ntc_fixed/1000:.1f} GW, investable up to "
                      f"+{ntc_invest_max/1000:.1f} GW (max total={ntc_max/1000:.1f} GW)")

        self.n_dims = len(self.bounds)
        if verbose:
            print(f"  Total outer search dimensions: {self.n_dims}")

        # ---- Build index maps for fast lookup during __call__ ----
        # Map: (node, instrument) -> dimension index
        self.flex_dim_map = {}
        for i, info in enumerate(self.dim_info):
            if info['type'] == 'flex':
                self.flex_dim_map[(info['node'], info['instr'])] = i
        # Map: link_name -> dimension index
        self.ntc_dim_map = {}
        for i, info in enumerate(self.dim_info):
            if info['type'] == 'ntc':
                self.ntc_dim_map[info['link']] = i

        # Fix VRE at the TYNDP 2024 Distributed Energy 2040 scenario level
        # (progress.md Section 39.4). Per-node installed PV/onshore/offshore
        # are taken from node_params['k_bar_vre']; the old DE hardcode and the
        # "2/3 of caps" rule are removed -- every node is fixed at its sourced
        # scenario capacity.
        k_bar_vre_global = self.D.get('k_bar_vre', {})
        for n in self.all_nodes:
            n_vre = node_params.get(n, {}).get('k_bar_vre', k_bar_vre_global)
            for v in self.D['tech_vre']:
                self.model.K_new[n, v].fix(float(n_vre.get(v, 0)))
            if verbose:
                print(f"  {n} VRE fixed [GW]: "
                      f"PV={n_vre.get('pv',0)/1000:.1f}, "
                      f"Won={n_vre.get('wind_on',0)/1000:.1f}, "
                      f"Woff={n_vre.get('wind_off',0)/1000:.1f}")

        # Pre-compute minimum feasible P2G capacity for H₂ demand (DE)
        sc_eta = D.get('sc_eta', 0.70)
        total_weighted_hours = sum(self.W[t] for t in D['h'])
        # Per-node minimum-feasible P2G (S.61). Generalised quota dict
        # h2_targets={node:MWh_H2} (--h2-2040) takes precedence; else legacy
        # DE sc_h2_target + DK sc_h2_target_dk. min_feasible_p2g[_dk] kept as
        # back-compat aliases for DE/DK.
        self.h2_min = {}
        h2_targets = D.get('h2_targets')
        if h2_targets and total_weighted_hours > 0:
            for _n, _t in h2_targets.items():
                if _t and _t > 0:
                    self.h2_min[_n] = _t / (sc_eta * total_weighted_hours) * 1.05
        elif total_weighted_hours > 0:
            _de = D.get('sc_h2_target', 0)
            if _de > 0:
                self.h2_min[self.n_node] = _de / (sc_eta * total_weighted_hours) * 1.05
        self.min_feasible_p2g = self.h2_min.get(self.n_node, 0.0)
        if verbose and self.min_feasible_p2g > 0:
            print(f"  Min feasible P2G ({self.n_node}) for H2: {self.min_feasible_p2g:,.0f} MW")

        # Same calculation for DK (Tier 1 #1): minimum DK P2G capacity at
        # which the DK H2 quota becomes satisfiable. Below this, LPOracle
        # must deactivate H2Target_DK to avoid LP infeasibility.
        if 'DK' not in self.h2_min and total_weighted_hours > 0:
            _dk = D.get('sc_h2_target_dk', 0)
            if _dk > 0:
                self.h2_min['DK'] = _dk / (sc_eta * total_weighted_hours) * 1.05
        self.min_feasible_p2g_dk = self.h2_min.get('DK', 0.0)
        if verbose and self.min_feasible_p2g_dk > 0:
            print(f"  Min feasible DK P2G for H2: {self.min_feasible_p2g_dk:,.0f} MW")

        # P2G-endogenous: leave K_sc free for the LP at every outer-search
        # node, floored at the H2-feasibility minimum (DE/DK) so the quota is
        # always satisfiable; the LP sizes it cost-optimally (S.60).
        if self.p2g_endogenous:
            for node in self.outer_search_nodes:
                floor = self.h2_min.get(node, 0.0)
                try:
                    self.model.K_sc[node].unfix()
                    self.model.K_sc[node].setlb(floor)
                except Exception:
                    pass
            if verbose:
                print(f"  P2G ENDOGENOUS: K_sc LP-chosen at "
                      f"{self.outer_search_nodes} (floored by H2 quota); "
                      f"sc dims removed from outer search")

        # EV-2040: fix K_ev at the TYNDP fleet power for every EV node; leave
        # it fixed (exogenous fleet, not a decision) -- B_opt must NOT free it.
        if self.ev_2040:
            ev_K = D.get('ev_K', {})
            for node in self.all_nodes:
                try:
                    _k = float(ev_K.get(node, 0.0))
                    self.model.K_ev[node].setub(max(_k, self.model.K_ev[node].ub or 0))
                    self.model.K_ev[node].fix(_k)
                except Exception:
                    pass

        self.solve_count = 0
        self._first_solve_reported = False  # solve-time watchdog
        # Outer-search constraint metric: 'vf' = cannibalisation channel
        # (value factor; the paper's PRIMARY mitigation target), 'cr' = cost
        # recovery (investability). BOTH are always computed and reported.
        self.target_metric = 'vf'

    def _get_flex_val(self, x, node, instr):
        """Get the value for a (node, instrument) from the search vector x."""
        idx = self.flex_dim_map.get((node, instr))
        if idx is None:
            return None  # LP-endogenous
        return max(0.0, min(x[idx], self.bounds[idx]))

    def _get_ntc_val(self, x, link_name):
        """Get NTC investment value for a link from search vector x."""
        idx = self.ntc_dim_map.get(link_name)
        if idx is None:
            return 0.0
        return max(0.0, min(x[idx], self.bounds[idx]))

    def _solve(self, model, dual_cons=None):
        """Solve `model` via the APPSI persistent solver (incremental, no
        re-export) and populate the legacy `dual` Suffix so existing extraction
        (model.dual[con]) works unchanged. Returns True iff optimal. Energy-
        balance duals always loaded; `dual_cons` adds extras (honesty pins)."""
        try:
            res = self.solver.solve(model)
        except Exception:
            # APPSI raises if load_solution=True and no feasible solution exists
            # (infeasible, or time-limit hit with no incumbent) -> non-optimal.
            return False
        self._last_wall = getattr(res, 'wallclock_time', None)
        if res.termination_condition != self._ATC.optimal:
            return False
        if hasattr(model, 'dual'):
            _cons = (list(self._eb_cons) if dual_cons is None
                     else list(self._eb_cons) + list(dual_cons))
            try:
                _d = self.solver.get_duals(_cons)
                for _c in _cons:
                    model.dual[_c] = _d[_c]
            except Exception:
                pass
        return True

    def __call__(self, x):
        """Evaluate the LP at capacity vector x. Returns dict or None."""

        # ---- Fix flexibility capacities for all outer-search nodes ----
        for node in self.outer_search_nodes:
            sto = self._get_flex_val(x, node, 'sto')
            sc  = self._get_flex_val(x, node, 'sc')
            dsm = self._get_flex_val(x, node, 'dsm')
            ev  = self._get_flex_val(x, node, 'ev')

            # H2 quota = exogenous mandated demand, ALWAYS produced (S.46.1 /
            # T2.1). Floor P2G at the min capacity that satisfies the quota.
            # ONLY when P2G is an outer dimension; under --p2g-endogenous sc is
            # None (no dim) and K_sc is LP-chosen, floored via setlb in __init__
            # -> skip here (S.78 bugfix: max(None, float) crashed the frontier).
            if not self.p2g_endogenous:
                _flr = self.h2_min.get(node, 0.0)
                if _flr > 0 and sc is not None:
                    sc = max(sc, _flr)

            self.model.K_sto_p[node].fix(sto)
            self.model.K_sto_e[node].fix(sto * self.D.get('sto_ep_ratio', 4.0))
            if not self.p2g_endogenous:
                self.model.K_sc[node].fix(sc)   # else LP-chosen (floored via setlb)
            self.model.K_dsm[node].fix(dsm)
            if not self.ev_2040:
                self.model.K_ev[node].fix(ev)   # else fixed at exogenous fleet

        # ---- Fix NTC investment for all links ----
        for l_name in self.ntc_link_names:
            ntc_inv = self._get_ntc_val(x, l_name)
            self.model.NTC_invest[l_name].fix(ntc_inv)

        # H2 quota ALWAYS active (P2G floored above to a feasible level).
        if hasattr(self.model, 'H2Target'):
            self.model.H2Target.activate()

        # Toggle DK H2 quota constraint based on DK P2G (Tier 1 #1).
        # Mirrors the DE pattern: if DK P2G capacity is too small to
        # satisfy a non-zero DK quota, deactivating the constraint
        # prevents LP infeasibility. The quota itself is set in the data
        # dict (sc_h2_target_dk); when that is 0 the constraint is built
        # but already deactivated at build time.
        if hasattr(self.model, 'H2Target_DK') and self.min_feasible_p2g_dk > 0:
            self.model.H2Target_DK.activate()

        # Toggle fast-response reserve PER NODE -- ONLY when a node's storage
        # actually crosses the 1 MW threshold (cache last state), so we don't
        # re-touch ~5000 constraints every eval. With the persistent solver that
        # reconciliation was a big chunk of per-solve overhead. Result-identical.
        # [S.82.1]
        if hasattr(self.model, 'FastReserve'):
            if not hasattr(self, '_fr_state'):
                self._fr_state = {}
            for node in getattr(self.model, 'FastReserve_nodes', []):
                node_sto = self.model.K_sto_p[node].value or 0.0
                _want = node_sto >= 1.0
                if self._fr_state.get(node) != _want:
                    self._fr_state[node] = _want
                    for t in self.D['h']:
                        con = self.model.FastReserve[node, t]
                        if _want:
                            con.activate()
                        else:
                            con.deactivate()

        try:
            _t_solve = time.time()
            _ok = self._solve(self.model)
            _solve_dt = time.time() - _t_solve
            # ---- solve-time watchdog: surface a slow/wrong solver EARLY -------
            _warn_s = float(os.environ.get('SLOW_SOLVE_WARN_S', '30'))
            if not self._first_solve_reported:
                self._first_solve_reported = True
                print(f"  [solve timing] first eval LP {_solve_dt:.1f}s "
                      f"({self._solver_kind}; incl. ONE-TIME persistent build -- "
                      f"later solves are faster).", flush=True)
            elif not getattr(self, '_steady_reported', False):
                self._steady_reported = True
                _w = getattr(self, '_last_wall', None)
                _br = (f", solver {_w:.1f}s, my-overhead {_solve_dt-_w:.1f}s"
                       if _w is not None else "")
                print(f"  [solve timing] STEADY-STATE LP {_solve_dt:.1f}s total"
                      f"{_br} ({self._solver_kind}, persistent). Frontier runtime "
                      f"~ total x evals.", flush=True)
            elif _solve_dt > _warn_s:
                self._n_slow = getattr(self, '_n_slow', 0) + 1
                if self._n_slow <= 3 or self._n_slow % 100 == 0:
                    print(f"  [SLOW SOLVE #{self._n_slow}] LP took {_solve_dt:.0f}s "
                          f"(> {_warn_s:.0f}s, {self._solver_kind}).", flush=True)
            if not _ok:
                if self.solve_count == 0:
                    print(f"  [oracle] Non-optimal termination")
                return None
        except Exception as e:
            if self.solve_count == 0:
                print(f"  [oracle] Solver exception: {e}")
            return None

        self.solve_count += 1

        m = self.model
        D = self.D
        W = self.W

        # ---- Build result dict with all dimension values ----
        result = {'sys_cost': pyo.value(m.Obj)}

        # Store all outer-search dimension values
        for i, key in enumerate(self.x_keys):
            info = self.dim_info[i]
            if info['type'] == 'flex':
                node, instr = info['node'], info['instr']
                val = self._get_flex_val(x, node, instr)
                # Apply P2G snap for DE
                if instr == 'sc':
                    _flr = self.h2_min.get(node, 0.0)
                    if _flr > 0 and val < _flr:
                        val = _flr
                result[key] = val
            elif info['type'] == 'ntc':
                result[key] = self._get_ntc_val(x, info['link'])

        # P2G-endogenous: record LP-chosen sc per outer-search node (sc is
        # not in x_keys in this mode, so the loop above didn't capture it).
        if self.p2g_endogenous:
            for node in self.outer_search_nodes:
                suffix = '' if node == self.n_node else f'_{node.lower()}'
                result[f'sc{suffix}_mw'] = pyo.value(m.K_sc[node])
        if self.ev_2040:
            for node in self.all_nodes:
                suffix = '' if node == self.n_node else f'_{node.lower()}'
                result[f'ev{suffix}_mw'] = pyo.value(m.K_ev[node])

        # Backward-compatible DE keys
        result['sto_mw'] = result.get('sto_mw', 0)
        result['sc_mw'] = result.get('sc_mw', 0)
        result['dsm_mw'] = result.get('dsm_mw', 0)
        result['ev_mw'] = result.get('ev_mw', 0)

        # ---- Record LP-endogenous node capacities (from LP solution) ----
        for node in self.lp_endogenous_nodes:
            n_lc = node.lower()
            result[f'sto_{n_lc}_mw_lp'] = pyo.value(m.K_sto_p[node])
            result[f'sc_{n_lc}_mw_lp'] = pyo.value(m.K_sc[node])
            result[f'dsm_{n_lc}_mw_lp'] = pyo.value(m.K_dsm[node])
            result[f'ev_{n_lc}_mw_lp'] = pyo.value(m.K_ev[node])

        # ---- Record actual NTC totals (fixed + investment) ----
        for l_name in self.ntc_link_names:
            ntc_inv = self._get_ntc_val(x, l_name)
            links = D.get('links', {})
            ntc_fixed = links[l_name]['ntc_fixed']
            result[f'ntc_total_{l_name.lower()}_mw'] = ntc_fixed + ntc_inv

        # ---- Heat-pump operation diagnostics (Tier 2 #6) ----
        # Captures: total weighted HP electric load, total weighted thermal
        # storage cycling, total backup heat (slack — should ~ 0 if the
        # HP+storage capacities are sized adequately).
        if (D.get('enable_heat', False)
                and 'DK' in self.all_nodes
                and hasattr(m, 'HEAT_NODES')):
            try:
                hp_total_mwh_e = sum(W[t] * pyo.value(m.Q_hp['DK', t])
                                     for t in D['h'])
                hp_avg_mw_e   = (hp_total_mwh_e
                                 / max(sum(W[t] for t in D['h']), 1.0))
                bkp_total_mwh = sum(W[t] * pyo.value(m.Q_heat_bkp['DK', t])
                                    for t in D['h'])
                soc_max  = max(pyo.value(m.SOC_heat['DK', t]) for t in D['h'])
                soc_mean = (sum(W[t] * pyo.value(m.SOC_heat['DK', t])
                                for t in D['h'])
                            / max(sum(W[t] for t in D['h']), 1.0))
                result['hp_dk_total_twh_e']    = hp_total_mwh_e / 1e6
                result['hp_dk_avg_load_mw_e']  = hp_avg_mw_e
                result['hp_dk_backup_twh_th']  = bkp_total_mwh / 1e6
                result['hp_dk_soc_max_mwh']    = soc_max
                result['hp_dk_soc_mean_mwh']   = soc_mean
            except Exception:
                # Don't let reporting glitches kill the optimisation.
                pass

        # ---- Compute VF + cost-recovery metrics (progress.md Section 34) ----
        self._vf_cr_metrics(result)

        return result

    def _vf_cr_metrics(self, result):
        """Populate `result` with per-node VF and cost-recovery metrics.

        See progress.md Section 34. Computed from the LP duals of the
        energy balance, so this works after ANY solve (regular __call__
        or a baseline solve). Both VF (a profile-effect ratio) and CR
        (cost recovery, the absolute investability metric) are produced:

          MV   market value    = Sum_t lambda[n,t] * P[v,n,t]   [EUR/yr]
          AC   annualized cost = capex[v] * K_installed[v,n]     [EUR/yr]
          CR   cost recovery   = MV / AC                         [-]
          MM   missing money   = max(0, AC - MV)                 [EUR/yr]

        Note: lambda[n,t] already contains the representative-week weight
        W[t] (OPEX terms in the objective are W[t]-scaled), so Sum_t
        lambda*P is ALREADY the weighted annual revenue -- no extra W[t]
        factor. The identity CR = VF * base_price / LCOE holds by
        construction (LCOE = AC / weighted_generation).
        """
        m, D, W = self.model, self.D, self.W
        for node in self.all_nodes:
            lam = {t: m.dual[m.EnergyBalance[node, t]] for t in D['h']}
            total_weight = sum(W[t] for t in D['h'])
            base_price = sum(lam[t] for t in D['h']) / total_weight
            result[f'base_price_{node}'] = base_price

            for v in D['tech_vre']:
                P_vt = {t: pyo.value(m.P[node, v, t]) for t in D['h']}
                total_gen_w = sum(W[t] * P_vt[t] for t in D['h'])
                # Market value: dual-weighted revenue (lambda carries W[t])
                mv = sum(lam[t] * P_vt[t] for t in D['h'])
                # Installed VRE capacity (existing + new) and annual cost
                k_inst = pyo.value(m.K_new[node, v]) + D['k_exist_vre'].get(v, 0)
                # Total annual fixed cost = annualized capex + fixed O&M
                # (DEA 2040; S.56 audit fix -- capex-only AC overstated CR).
                ac = (D['capex'][v] + D.get('fom', {}).get(v, 0)) * k_inst
                if total_gen_w > 1e-3 and abs(base_price) > 1e-3:
                    cp = mv / total_gen_w
                    vf = cp / base_price
                else:
                    cp, vf = 0.0, 0.0
                # Cost recovery / missing money (undefined when no capacity)
                if ac > 1e-3:
                    cr = mv / ac
                    mm = max(0.0, ac - mv)
                    lcoe = ac / total_gen_w if total_gen_w > 1e-3 else 0.0
                else:
                    cr, mm, lcoe = 0.0, 0.0, 0.0
                result[f'cp_{v}_{node}'] = cp
                result[f'vf_{v}_{node}'] = vf
                result[f'mv_{v}_{node}'] = mv
                result[f'ac_{v}_{node}'] = ac
                result[f'cr_{v}_{node}'] = cr
                result[f'mm_{v}_{node}'] = mm
                result[f'lcoe_{v}_{node}'] = lcoe
                result[f'gen_{v}_{node}'] = total_gen_w

        # ---- System-level aggregates (S.56) --------------------------
        # Same (tech, node) inclusion rule as compute_vf_min: installed
        # capacity >= 1 MW. Three aggregates:
        #   vf_genw            generation-weighted system VF (companion to
        #                      the weakest-link VF_min; alternative target
        #                      via --target vfw)
        #   cannib_loss_eur    euro cannibalization loss
        #                      = sum (base_price_n - CP_vn) * gen_vn
        #   subsidy_bill_eur   CfD-style top-up to make every VRE whole
        #                      = sum missing money (mm >= 0)
        node_params = D.get('node_params', {})
        k_bar_vre_global = D.get('k_bar_vre', {})
        _wsum = _gsum = _loss = _bill = 0.0
        for node in self.all_nodes:
            n_vre = node_params.get(node, {}).get('k_bar_vre', k_bar_vre_global)
            bp_n = result.get(f'base_price_{node}', 0.0)
            for v in D['tech_vre']:
                if n_vre.get(v, 0) < 1.0:
                    continue
                g = result.get(f'gen_{v}_{node}', 0.0)
                vf = result.get(f'vf_{v}_{node}', 0.0)
                cp = result.get(f'cp_{v}_{node}', 0.0)
                if g > 1e-3 and vf > 1e-6:
                    _wsum += g * vf
                    _gsum += g
                    _loss += (bp_n - cp) * g
                _bill += result.get(f'mm_{v}_{node}', 0.0)
        result['vf_genw'] = (_wsum / _gsum) if _gsum > 0 else 0.0
        result['cannib_loss_eur'] = _loss
        result['subsidy_bill_eur'] = _bill

        # ---- Residual-interconnector boundary diagnostics (S.56) -----
        # The fixed-price residual links (DE/DK/AT/NL <-> non-modelled
        # neighbours) put an exogenous floor (price_export) and quasi-cap
        # (price_import) on nodal prices. Record, per residual node, the
        # weighted share of hours at those boundaries and the share of
        # hours the residual flow is at capacity -- the referee-facing
        # measure of how much price formation is exogenous.
        resid = D.get('resid_intercon', {})
        total_w = sum(W[t] for t in D['h'])
        for rn, rd in resid.items():
            if rn not in self.all_nodes:
                continue
            lamr = {t: m.dual[m.EnergyBalance[rn, t]] for t in D['h']}
            cap = rd['ntc'] + (pyo.value(m.NTC_resid_invest[rn])
                               if hasattr(m, 'NTC_resid_invest') else 0.0)
            pexp, pimp = rd['price_export'], rd['price_import']
            # Per-hour price = lambda / W (the dual carries the rep-week
            # weight because OPEX in the objective is W-scaled). Comparing
            # raw lambda against prices inflated ceil_share to ~1 (S.57 fix).
            _pes = rd.get('price_export_series', {}); _pis = rd.get('price_import_series', {})
            w_floor = sum(W[t] for t in D['h']
                          if W[t] > 0 and lamr[t] / W[t] <= _pes.get(t, pexp) + 0.5)
            w_ceil = sum(W[t] for t in D['h']
                         if W[t] > 0 and lamr[t] / W[t] >= _pis.get(t, pimp) - 0.5)
            w_expcap = sum(W[t] for t in D['h']
                           if pyo.value(m.F_resid_exp[rn, t]) >= cap - 1.0)
            w_impcap = sum(W[t] for t in D['h']
                           if pyo.value(m.F_resid_imp[rn, t]) >= cap - 1.0)
            result[f'resid_{rn}_floor_share'] = w_floor / total_w
            result[f'resid_{rn}_ceil_share'] = w_ceil / total_w
            result[f'resid_{rn}_expcap_share'] = w_expcap / total_w
            result[f'resid_{rn}_impcap_share'] = w_impcap / total_w

        # Backward-compatible keys for primary node (DE)
        n = self.n_node
        result['base_price'] = result[f'base_price_{n}']
        for v in D['tech_vre']:
            result[f'cp_{v}'] = result[f'cp_{v}_{n}']
            result[f'vf_{v}'] = result[f'vf_{v}_{n}']
            result[f'cr_{v}'] = result[f'cr_{v}_{n}']
            result[f'mv_{v}'] = result[f'mv_{v}_{n}']
            result[f'mm_{v}'] = result[f'mm_{v}_{n}']
        return result

    def format_capacities(self, r):
        """Extract capacity dict from a result dict (for logging/CSV)."""
        caps = {k: r[k] for k in self.x_keys if k in r}
        # Also include LP-endogenous capacities
        for node in self.lp_endogenous_nodes:
            n_lc = node.lower()
            for suffix in [f'sto_{n_lc}_mw_lp', f'sc_{n_lc}_mw_lp',
                           f'dsm_{n_lc}_mw_lp', f'ev_{n_lc}_mw_lp']:
                if suffix in r:
                    caps[suffix] = r[suffix]
        # Include NTC totals
        for l_name in self.ntc_link_names:
            key = f'ntc_total_{l_name.lower()}_mw'
            if key in r:
                caps[key] = r[key]
        return caps

    def get_bounds(self):
        """Return list of (lo, hi) tuples for scipy."""
        return [(0, b) for b in self.bounds]

    def compute_vf_min(self, r):
        """
        Compute system-wide VF_min: the worst VRE value factor across
        all nodes and technologies. This is the optimization target.

        Excludes VRE/node combos with zero installed capacity (e.g. CZ
        wind_off is zero because CZ is landlocked — its VF=0 is
        meaningless and must not drag down the system VF_min).
        """
        vfs = []
        node_params = self.D.get('node_params', {})
        k_bar_vre_global = self.D.get('k_bar_vre', {})
        for node in self.all_nodes:
            np_ = node_params.get(node, {})
            n_vre = np_.get('k_bar_vre', k_bar_vre_global)
            for v in self.D['tech_vre']:
                if n_vre.get(v, 0) < 1.0:
                    continue
                key = f'vf_{v}_{node}'
                if key in r and r[key] > 1e-6:
                    vfs.append(r[key])
        return min(vfs) if vfs else min(r.get('vf_wind_on', 0), r.get('vf_pv', 0))

    def compute_cr_min(self, r):
        """
        Compute system-wide CR_min: the worst VRE cost-recovery ratio
        (market value / annualized cost) across all nodes and
        technologies. This is the Section 34 optimization target,
        replacing VF_min.

        Excludes VRE/node combos with zero installed capacity (e.g. CZ
        wind_off is zero because CZ is landlocked -- its CR=0 is
        meaningless and must not drag down the system CR_min). The
        selection of (tech, node) pairs is identical to compute_vf_min
        so that VF_min and CR_min always refer to the same population.
        """
        crs = []
        node_params = self.D.get('node_params', {})
        k_bar_vre_global = self.D.get('k_bar_vre', {})
        for node in self.all_nodes:
            np_ = node_params.get(node, {})
            n_vre = np_.get('k_bar_vre', k_bar_vre_global)
            for v in self.D['tech_vre']:
                if n_vre.get(v, 0) < 1.0:
                    continue
                key = f'cr_{v}_{node}'
                if key in r and r[key] > 1e-6:
                    crs.append(r[key])
        return min(crs) if crs else min(r.get('cr_wind_on', 0), r.get('cr_pv', 0))

    def compute_vf_genw(self, r):
        """Generation-weighted system VF (S.56). Companion aggregate to the
        weakest-link VF_min: robust to a single outlier (node, tech) pair
        dominating the target. Computed in _vf_cr_metrics."""
        return r.get('vf_genw', 0.0)

    def compute_target_min(self, r):
        """Active outer-search constraint metric (see self.target_metric).
        'vf' -> cannibalisation channel, weakest-link (value factor min);
        'vfw' -> generation-weighted VF (outlier-robust variant);
        'cr' -> cost recovery (investability). All are always recorded."""
        tm = getattr(self, 'target_metric', 'vf')
        if tm == 'cr':
            return self.compute_cr_min(r)
        if tm == 'vfw':
            return self.compute_vf_genw(r)
        return self.compute_vf_min(r)

    # ------------------------------------------------------------------
    # Baselines (progress.md Section 36)
    # ------------------------------------------------------------------
    def make_today_vector(self):
        """Build the outer-search vector x for today's installed flexibility
        fleet -- baseline B_today (progress.md Section 36).

        NTC dimensions are 0 (today = no transmission investment). DE P2G
        is floored at the H2-feasibility minimum: today's real electrolysis
        (~0.18 GW) cannot deliver the scenario's mandated 20 TWh H2 quota,
        so B_today must carry at least the H2-feasible P2G capacity (see
        progress.md task T2.1).
        """
        k_today = self.D.get('k_today_flex', {})
        x = []
        for info in self.dim_info:
            if info['type'] == 'flex':
                node, instr = info['node'], info['instr']
                val = float(k_today.get(node, {}).get(instr, 0.0))
                if node == self.n_node and instr == 'sc':
                    val = max(val, self.min_feasible_p2g)
                x.append(val)
            else:  # ntc dimension
                x.append(0.0)
        return x

    def solve_cost_optimal(self):
        """Baseline B_opt (progress.md Section 36): the cost-optimal system.

        VRE stays fixed at the scenario level; ALL flexibility (storage,
        P2G, DSM, EV at the outer-search nodes, plus NTC investment) is
        left free for the LP to minimize cost. H2 demand and reserves are
        active. Storage energy/power ratio is locked at 4 h on the
        outer-search nodes, matching the convention the directed search
        imposes, so baseline and frontier price the same storage tech.

        Returns a result dict (sys_cost + VF/CR metrics + LP-chosen flex
        capacities) or None on non-optimal termination. After this call
        the flexibility variables are left unfixed; the next regular
        __call__ re-fixes them, so no manual state restore is needed.
        """
        m = self.model
        # Unfix outer-search flexibility so the LP optimizes it
        for node in self.outer_search_nodes:
            _flex_vars = [m.K_sto_p[node], m.K_sto_e[node], m.K_sc[node],
                          m.K_dsm[node]]
            if not self.ev_2040:
                _flex_vars.append(m.K_ev[node])   # EV fleet stays fixed
            for var in _flex_vars:
                var.unfix()
        for l_name in self.ntc_link_names:
            m.NTC_invest[l_name].unfix()
        # Lock E/P = 4 h on outer-search nodes (idempotent ConstraintList)
        if not hasattr(m, 'EPRatioOpt'):
            m.EPRatioOpt = pyo.ConstraintList()
            for node in self.outer_search_nodes:
                m.EPRatioOpt.add(m.K_sto_e[node] == self.D.get('sto_ep_ratio', 4.0) * m.K_sto_p[node])
        else:
            m.EPRatioOpt.activate()
        # H2 demand + reserves are real scenario requirements -> active
        if hasattr(m, 'H2Target'):
            m.H2Target.activate()
        if hasattr(m, 'H2Target_DK') and self.D.get('sc_h2_target_dk', 0) > 0:
            m.H2Target_DK.activate()
        if hasattr(m, 'FastReserve'):
            for node in getattr(m, 'FastReserve_nodes', []):
                for t in self.D['h']:
                    m.FastReserve[node, t].activate()
        try:
            _t_solve = time.time()
            _ok = self._solve(m)
            print(f"  [solve timing] baseline B_opt LP {time.time()-_t_solve:.1f}s "
                  f"({self._solver_kind}, persistent); ~1-5 s on Gurobi.",
                  flush=True)
            if not _ok:
                print(f"  [B_opt] Non-optimal termination")
                m.EPRatioOpt.deactivate()
                return None
        except Exception as e:
            print(f"  [B_opt] Solver exception: {e}")
            m.EPRatioOpt.deactivate()
            return None
        self.solve_count += 1

        result = {'sys_cost': pyo.value(m.Obj)}
        # LP-chosen flexibility capacities at the outer-search nodes
        for node in self.outer_search_nodes:
            suffix = '' if node == self.n_node else f'_{node.lower()}'
            result[f'sto{suffix}_mw'] = pyo.value(m.K_sto_p[node])
            result[f'sc{suffix}_mw'] = pyo.value(m.K_sc[node])
            result[f'dsm{suffix}_mw'] = pyo.value(m.K_dsm[node])
            result[f'ev{suffix}_mw'] = pyo.value(m.K_ev[node])
        # LP-endogenous node capacities
        for node in self.lp_endogenous_nodes:
            n_lc = node.lower()
            result[f'sto_{n_lc}_mw_lp'] = pyo.value(m.K_sto_p[node])
            result[f'sc_{n_lc}_mw_lp'] = pyo.value(m.K_sc[node])
            result[f'dsm_{n_lc}_mw_lp'] = pyo.value(m.K_dsm[node])
            result[f'ev_{n_lc}_mw_lp'] = pyo.value(m.K_ev[node])
        # NTC investment chosen by the LP
        for l_name in self.ntc_link_names:
            ntc_fixed = self.D['links'][l_name]['ntc_fixed']
            inv = pyo.value(m.NTC_invest[l_name])
            result[f'ntc_{l_name.lower()}_mw'] = inv
            result[f'ntc_total_{l_name.lower()}_mw'] = ntc_fixed + inv

        self._vf_cr_metrics(result)
        m.EPRatioOpt.deactivate()
        return result


# ============================================================================
# 1b. MULTIPROCESSING WORKERS — process-pool parallel LP evaluation
# ============================================================================
# Process-based parallelism sidesteps the cumulative-state corruption that
# kills threaded ``gurobipy`` after a few hundred solves. Each worker process
# has its own Python interpreter, its own gurobipy import, its own Gurobi env
# handle, and its own LPOracle — zero shared mutable state between workers.
#
# The cost is a one-time startup: each worker builds an LPOracle in the
# initializer (~30-60s per worker). This amortizes over the run.
#
# These functions MUST be module-level (not closures) because spawned worker
# processes re-import the module and lookup the function by qualified name.
# Any closure referring to outer-scope vars would be unpicklable.

_MP_WORKER_ORACLE = None


def _mp_worker_init(D):
    """ProcessPoolExecutor initializer: each worker builds its own LPOracle.

    Runs ONCE per worker process at pool startup. The D dict is pickled from
    the main process and reconstructed here. Pyomo models are not picklable,
    which is exactly why we rebuild them per-worker rather than trying to
    send oracle instances across process boundaries.
    """
    global _MP_WORKER_ORACLE
    import sys as _sys
    try:
        # verbose=False: suppress per-worker banner (solver check, NTC,
        # VRE fixes). Otherwise the main log reprints the whole banner
        # once per worker, making it hard to read.
        _MP_WORKER_ORACLE = LPOracle(D, verbose=False)
    except BaseException as e:
        # Worker init failures are silent by default in ProcessPoolExecutor;
        # force the traceback to stderr so we see it in the parent's log.
        import traceback as _tb
        print(f"[mp worker init] FAILED to build LPOracle: "
              f"{type(e).__name__}: {e}", file=_sys.stderr, flush=True)
        _tb.print_exc()
        raise

    # Register explicit cleanup on worker shutdown. Without this, gurobipy
    # emits "environment still referenced so free is deferred" because the
    # module-level _MP_WORKER_ORACLE global keeps a Pyomo model alive past
    # atexit. Dropping the reference before exit lets Gurobi reclaim the
    # env cleanly and suppresses the warning. Purely cosmetic — the OS
    # reclaims memory either way when the process dies.
    import atexit as _atexit
    import gc as _gc

    def _worker_cleanup():
        global _MP_WORKER_ORACLE
        _MP_WORKER_ORACLE = None
        _gc.collect()

    _atexit.register(_worker_cleanup)


def _mp_worker_solve(x):
    """ProcessPoolExecutor task: solve one LP with the worker's oracle.

    Input: x (list[float]) — portfolio candidate in full-dim space.
    Output: result dict (picklable, plain floats) or None on infeasibility.

    Cache management and snap-zone pre-filtering stay on the MAIN process.
    This function does only the LP solve, which keeps the worker logic
    trivially stateless and makes crash diagnosis easier.
    """
    global _MP_WORKER_ORACLE
    import sys as _sys
    try:
        return _MP_WORKER_ORACLE(list(x))
    except BaseException as e:
        import traceback as _tb, os as _os
        print(f"[mp worker {_os.getpid()}] solve EXCEPTION: "
              f"{type(e).__name__}: {e}", file=_sys.stderr, flush=True)
        _tb.print_exc()
        raise


# ============================================================================
# 2. DIRECTED SEARCH — targets min(VF_wind, VF_pv)
# ============================================================================

# ---- CMA-ES dimensionality helpers ----------------------------------------
# Two failure modes of naive CMA-ES usage on a warm-started, bounded search:
#
# (A) Saturation collapse. When the monotone frontier clamps
#     ``lower_bounds[i]`` close to ``upper_bounds[i]`` (e.g. DSM at 7999.3 MW of
#     an 8000 MW cap), the dimension's range drops below CMA-ES's internal
#     resolution. The ``BoundaryHandlerBase._inverse_i`` method then fails its
#     strict ``lb <= y <= ub`` assertion during ``CMAEvolutionStrategy``
#     construction with "argument of inverse must be within the given bounds".
#
# (B) Inverted-margin clip. The idiom
#     ``np.clip(x, lb + 1.0, ub - 1.0)`` silently inverts its arguments when
#     ``ub - lb < 2``. NumPy evaluates ``min(max(x, lb+1), ub-1)`` which, for
#     an inverted interval, returns ``ub - 1 < lb`` — below the true lower
#     bound. CMA-ES then rejects the initial mean.
#
# ``_split_active_frozen_dims`` addresses (A) by removing near-saturated
# dimensions from the search (fixing them at their midpoint). This also cures
# the ``sigma0`` mismatch: one scalar step size applied to a 1-MW-range dim
# alongside a 200-GW dim wastes the entire search budget. ``_interior_x0``
# addresses (B) as a defensive guard for any dimension that survives the
# saturation filter but still has a narrow interior.

SATURATION_TOL_MW = 10.0  # Dimensions with range < this are frozen at midpoint.


def _split_active_frozen_dims(lb, ub, tol=SATURATION_TOL_MW):
    """
    Partition dimensions by whether CMA-ES can meaningfully search them.

    Returns:
      active_idx   list[int]  : indices where ``ub[i] - lb[i] >= tol``
      frozen_idx   list[int]  : indices where the range is below ``tol``
      frozen_vals  dict[int,float] : mapping ``i -> 0.5*(lb[i]+ub[i])`` for
                                     frozen dims (used to reconstruct x)
    """
    d = len(lb)
    active_idx = [i for i in range(d) if ub[i] - lb[i] >= tol]
    frozen_idx = [i for i in range(d) if ub[i] - lb[i] < tol]
    frozen_vals = {i: 0.5 * (lb[i] + ub[i]) for i in frozen_idx}
    return active_idx, frozen_idx, frozen_vals


def _interior_x0(x, lb, ub, margin=1.0):
    """
    Project ``x`` into the open interior ``(lb+margin, ub-margin)``.

    For ranges narrower than ``2*margin`` the open interior is empty and the
    naive ``np.clip`` pattern inverts its arguments. In that case we snap to
    the midpoint — CMA-ES then starts exactly in-bounds and the frozen-dim
    filter would usually have removed this dimension anyway.
    """
    if ub - lb < 2.0 * margin:
        return 0.5 * (lb + ub)
    return float(np.clip(x, lb + margin, ub - margin))


def _print_opt_status(meta, indent='    '):
    """
    Compact one-line report of an optimizer's termination state.

    Prints a prominent warning if the evaluation budget was exhausted —
    that signals the returned result may be suboptimal and a higher
    ``max_evals`` would likely improve it. Also flags when the result came
    from the closest-match fallback (VF target not achieved).

    Expects ``meta`` = ``result['_opt_meta']`` from find_min_cost_portfolio
    or find_min_cost_monotone.
    """
    if not meta:
        return
    budget = meta.get('budget', 0)
    n_evals = meta.get('n_evals', 0)
    pct = int(100 * n_evals / budget) if budget else 0
    exhausted = meta.get('budget_exhausted', False)
    feasible = meta.get('feasible', False)
    fallback = meta.get('fallback_used', False)
    stops = meta.get('stop_reasons', [])
    d_nom = meta.get('d_nominal', 0)
    d_act = meta.get('d_active', d_nom)
    d_frz = meta.get('d_frozen', 0)
    opt_name = meta.get('optimizer', 'opt').upper()

    dim_note = f" [{d_act}/{d_nom} dims active]" if d_frz > 0 else ""
    n_workers_meta = meta.get('n_workers', 1)
    worker_note = f" [{n_workers_meta} workers]" if n_workers_meta > 1 else ""
    # Primary line: budget usage and termination reason.
    if exhausted:
        print(f"{indent}** {opt_name}: {n_evals}/{budget} evals "
              f"[BUDGET EXHAUSTED]{dim_note}{worker_note} — raise max_evals to improve")
    else:
        stop_str = ", ".join(str(s) for s in stops) if stops else "converged"
        print(f"{indent}{opt_name}: {n_evals}/{budget} evals ({pct}%) — "
              f"stopped: {stop_str}{dim_note}{worker_note}")
    # Waste-reduction telemetry: cache hits and P2G snap-zone skips.
    ch = meta.get('cache_hits', 0)
    sk = meta.get('snap_skips', 0)
    if ch or sk:
        print(f"{indent}   waste-reduction: cache_hits={ch}, "
              f"p2g_snap_skips={sk} (LP solves avoided: {ch})")
    # Secondary line: feasibility / fallback status.
    if fallback:
        print(f"{indent}** CLOSEST-MATCH FALLBACK: VF target not achieved; "
              f"returned nearest-VF point (informational only)")
    elif not feasible:
        print(f"{indent}** INFEASIBLE: no point satisfied VF >= target "
              f"(result={{}}; caller should keep previous state)")


def find_min_cost_portfolio(oracle, vf_target, x0, bounds=None, max_evals=120,
                            optimizer='cma', n_workers=1, oracle_pool=None,
                            parallel_mode='process', D=None, cma_seed=42,
                            accept_target=None):
    """
    Find the cheapest portfolio where system-wide VF_min >= vf_target.
    Targets the worst-performing VRE technology across ALL nodes.

    Parameters:
      optimizer: 'cma' (CMA-ES, default) or 'nelder-mead' (scipy fallback).
      n_workers: number of parallel LP workers (1 = serial, default).
      oracle_pool: list of LPOracle instances for thread-mode parallel eval;
                   must have len >= n_workers when using thread mode. Ignored
                   in process mode (each worker builds its own oracle).
      parallel_mode: 'process' (default) or 'thread'. Process mode spawns
                   separate Python processes via ProcessPoolExecutor; thread
                   mode uses ThreadPoolExecutor. Thread mode is faster to
                   start but unreliable over long runs due to Gurobi C-state
                   corruption.
      D: data dict. Required when parallel_mode == 'process' so worker
         processes can rebuild an LPOracle in their own interpreter.
    """
    if bounds is None:
        bounds = oracle.get_bounds()

    PENALTY = 1e12
    # accept_target (S.56): the penalty drives the search toward vf_target
    # (which the caller may INFLATE by --target-slack to offset the known
    # ~0.02 quadratic-penalty undershoot), while feasibility/acceptance is
    # judged against the TRUE target. Default: identical (old behaviour).
    accept = vf_target if accept_target is None else accept_target
    best_result = {'sys_cost': np.inf}
    eval_log = []
    # Track whether any feasible (VF_min >= vf_target) point was encountered
    # — used below to distinguish a genuine feasible best from the
    # closest-match fallback.
    feasible_seen = False

    # --- Eval caching and P2G snap-zone pre-filter ------------------------
    # Two waste reducers that act on the *input* side of the oracle:
    #
    # (1) P2G snap-zone pre-filter. For the primary node (DE), the oracle
    #     internally clamps sc < min_feasible_p2g to 0 (H₂ target makes the
    #     interval (0, min_sc) infeasible). CMA-ES doesn't know this — it
    #     samples values in the snap zone and the LP returns a result
    #     indistinguishable from sc=0. Pre-snapping here canonicalizes the
    #     input so cache lookups find those duplicates and so the objective
    #     surface's plateau is explicit.
    #
    # (2) Per-call eval cache. CMA-ES basins converge toward repeated
    #     sampling near the same point; rounding to EVAL_CACHE_RES (10 MW)
    #     makes those duplicates hit the cache and skip the LP solve.
    #     Cache is local to this function call (new one per VF target),
    #     so no cross-target contamination.
    de_sc_idx = oracle.flex_dim_map.get((oracle.n_node, 'sc'))
    min_sc = getattr(oracle, 'min_feasible_p2g', 0.0)
    EVAL_CACHE_RES = 10.0  # MW
    eval_cache = {}
    cache_hits = [0]
    snap_skips = [0]

    def _snap_x(x_list):
        """Apply P2G snap-zone pre-filter to a candidate point."""
        x2 = list(x_list)
        if de_sc_idx is not None and x2[de_sc_idx] < min_sc:
            x2[de_sc_idx] = min_sc   # floor to H2-feasible min (always-on H2)
            snap_skips[0] += 1
        return x2

    def _cache_key(x_list):
        return tuple(round(v / EVAL_CACHE_RES) for v in x_list)

    def _eval_one(x, ora):
        """Clip -> snap -> cache-check -> LP solve. Returns
        (penalized, r, vf_min, violation, x_snapped)."""
        x_clipped = [np.clip(x[i], bounds[i][0], bounds[i][1]) for i in range(len(x))]
        x_snapped = _snap_x(x_clipped)
        key = _cache_key(x_snapped)
        cached = eval_cache.get(key)
        if cached is not None:
            cache_hits[0] += 1
            return cached
        r = ora(x_snapped)
        if r is None:
            out = (1e15, None, None, 1.0, x_snapped)
        else:
            vf_min = ora.compute_target_min(r)  # active target metric (VF or CR)
            cost = r['sys_cost']
            violation = max(0.0, vf_target - vf_min)
            penalized = cost + PENALTY * violation**2
            out = (penalized, r, vf_min, violation, x_snapped)
        eval_cache[key] = out
        return out

    def _process(out):
        """Reduce an eval result into best_result / eval_log. Returns penalized."""
        nonlocal best_result, feasible_seen
        penalized, r, vf_min, violation, x_snapped = out
        if r is None:
            eval_log.append({'x': list(x_snapped), 'vf_min': -1.0,
                             'vf_w': 0.0, 'vf_pv': 0.0, 'cost': 1e15})
            return penalized
        eval_log.append({'x': list(x_snapped), 'vf_min': vf_min,
                         'vf_w': r.get('vf_wind_on', 0),
                         'vf_pv': r.get('vf_pv', 0),
                         'cost': r['sys_cost']})
        if vf_min is not None and vf_min >= accept - 1e-4:
            feasible_seen = True
            if r['sys_cost'] < best_result.get('sys_cost', np.inf):
                best_result = r.copy()
        return penalized

    def objective(x):
        # Serial evaluation path (Nelder-Mead, CMA-ES n_workers==1).
        return _process(_eval_one(x, oracle))

    d = len(x0)

    # Optimizer-state metadata captured below and stapled onto best_result.
    # See _opt_meta documentation near the return statement.
    opt_meta = {
        'optimizer': optimizer, 'budget': max_evals,
        'n_evals': 0, 'budget_exhausted': False,
        'stop_reasons': [], 'feasible': False, 'fallback_used': False,
        'd_nominal': d, 'd_active': d, 'd_frozen': 0,
    }

    if optimizer == 'cma':
        import cma
        lb = [bounds[i][0] for i in range(d)]
        ub = [bounds[i][1] for i in range(d)]

        # Structural dimensionality reduction: remove saturated dimensions
        # (ub - lb < SATURATION_TOL_MW) from the CMA-ES search, fixing them
        # at midpoint. In the global frontier lb[i]=0 by construction so this
        # filter is typically a no-op; we apply the same scaffolding as in
        # find_min_cost_monotone for symmetry and to future-proof the code
        # against callers that pass custom bounds.
        active_idx, frozen_idx, frozen_vals = _split_active_frozen_dims(lb, ub)
        d_active = len(active_idx)
        opt_meta['d_active'] = d_active
        opt_meta['d_frozen'] = len(frozen_idx)

        if d_active == 0:
            # Degenerate: nothing to search. Evaluate the frozen point once.
            x_full = [frozen_vals.get(i, 0.0) for i in range(d)]
            objective(x_full)
            opt_meta['stop_reasons'] = ['all_dims_frozen']
        else:
            # Reduced-dim objective: active search vars y_k map back to
            # x_full[active_idx[k]] = y_k; frozen dims stay at midpoint.
            def objective_reduced(y):
                x_full = [0.0] * d
                for k, i in enumerate(active_idx):
                    x_full[i] = y[k]
                for i, v in frozen_vals.items():
                    x_full[i] = v
                return objective(x_full)

            lb_a = [lb[i] for i in active_idx]
            ub_a = [ub[i] for i in active_idx]

            # Initial step size (sigma0): 25% of the mean ACTIVE-dim range.
            # Using only active dims avoids the pathology where a frozen dim
            # with range 0 drags sigma0 toward zero across the search.
            ranges_a = [ub_a[k] - lb_a[k] for k in range(d_active)]
            sigma0 = 0.25 * np.mean(ranges_a)
            # Popsize scales with the EFFECTIVE dimensionality, not nominal d.
            popsize = 2 * (4 + int(3 * np.log(max(d_active, 1))))

            # CMA-ES options:
            #   - bounds: hard box constraints (CMA-ES handles these natively)
            #   - maxfevals: evaluation budget
            #   - popsize: 2x default, doubled for the non-smooth penalty surface
            #   - tolx, tolfun: generous — prefer full budget over premature
            #     convergence to local minima
            #   - verbose: -9 suppresses CMA-ES's internal logging
            #   - seed: fixed for reproducibility across warm-started targets
            opts = cma.CMAOptions()
            opts.set('bounds', [lb_a, ub_a])
            opts.set('maxfevals', max_evals)
            opts.set('popsize', popsize)
            opts.set('tolx', 100)        # stop if step < 100 MW (converged)
            opts.set('tolfun', 1e5)      # stop if cost spread < 100k EUR
            # tolflatfitness: number of consecutive generations with flat
            # best-fitness before CMA-ES stops. Default = 1 fires too
            # eagerly on the quadratic-penalty plateau (see progress.md
            # section 33). Raise to 10 generations: gives the search
            # 10*popsize ~= 200 extra evals on a flat basin before
            # quitting, which is cheap insurance against premature stop.
            opts.set('tolflatfitness', 10)
            opts.set('verbose', -9)      # silence CMA-ES output
            opts.set('seed', int(cma_seed))

            # Warm-start mean for active dims. _interior_x0 guards against
            # inverted-margin clipping on narrow-but-not-frozen ranges.
            y0 = [_interior_x0(x0[i], lb[i], ub[i]) for i in active_idx]

            es = cma.CMAEvolutionStrategy(y0, sigma0, opts)

            # Decide between serial and parallel evaluation paths.
            # Parallel path requires n_workers > 1 AND mode-specific inputs:
            #   - THREAD mode: oracle_pool with >= n_workers entries (one
            #     oracle per thread; Queue-based checkout keeps it safe).
            #   - PROCESS mode: D (the data dict) so each spawned worker
            #     can build its own LPOracle at startup.
            # Process mode is the default (see --parallel-mode CLI flag);
            # thread mode is retained for short tests but unreliable for
            # long runs due to shared Gurobi C-state corruption.
            use_parallel = n_workers > 1
            if parallel_mode == 'thread':
                use_parallel = use_parallel and (
                    oracle_pool is not None
                    and len(oracle_pool) >= n_workers
                )
            elif parallel_mode == 'process':
                use_parallel = use_parallel and (D is not None)
            else:
                use_parallel = False

            if use_parallel and parallel_mode == 'thread':
                from concurrent.futures import ThreadPoolExecutor
                from queue import Queue as _Queue

                pool_q = _Queue()
                for o in oracle_pool[:n_workers]:
                    pool_q.put(o)

                def _parallel_eval(y):
                    """Thread-pool worker: reconstruct x_full, check out an
                    oracle from the shared queue, run LP, return eval tuple.
                    Returns the raw output of _eval_one (no side effects);
                    the main thread applies _process to keep best_result /
                    eval_log updates single-threaded (race-free).

                    Exceptions inside a thread pool worker are only raised
                    on the main thread when the future is consumed, and a
                    C-level crash (e.g. Gurobi/Pyomo thread-unsafety) bypasses
                    Python's exception machinery entirely. We print a trace
                    from the worker so if the process crashes we at least see
                    what was running."""
                    import traceback as _tb
                    x_full = [0.0] * d
                    for k, i in enumerate(active_idx):
                        x_full[i] = y[k]
                    for i, v in frozen_vals.items():
                        x_full[i] = v
                    ora = pool_q.get()
                    try:
                        return _eval_one(x_full, ora)
                    except BaseException as _e:
                        print(f"  [parallel worker] EXCEPTION: {type(_e).__name__}: {_e}",
                              flush=True)
                        _tb.print_exc()
                        raise
                    finally:
                        pool_q.put(ora)

                # Manual ask/tell loop. CMA-ES's internal loop (es.optimize)
                # serializes evaluations; by driving ask/tell ourselves we
                # dispatch the whole generation to the thread pool at once.
                # Gurobi releases the GIL during optimize(), so threaded LP
                # solves achieve real parallelism.
                executor = ThreadPoolExecutor(max_workers=n_workers)
                try:
                    # Manual ask/tell: dispatches popsize candidates per
                    # generation to the thread pool. We DO NOT truncate
                    # solutions — cma's tell() requires exactly popsize
                    # fitnesses per call, so the last generation may push
                    # countevals slightly above max_evals (by up to
                    # popsize - 1). This is bounded and intentional.
                    _gen_n = 0
                    while not es.stop():
                        solutions = es.ask()
                        _gen_n += 1
                        print(f"  [gen {_gen_n}] dispatching {len(solutions)} "
                              f"evals to {n_workers} workers "
                              f"(countevals={es.countevals})", flush=True)
                        try:
                            outs = list(executor.map(_parallel_eval, solutions))
                        except BaseException as _e:
                            import traceback as _tb
                            print(f"  [main] executor.map raised: "
                                  f"{type(_e).__name__}: {_e}", flush=True)
                            _tb.print_exc()
                            raise
                        fitnesses = [_process(out) for out in outs]
                        es.tell(solutions, fitnesses)
                        if es.countevals >= max_evals:
                            break
                finally:
                    executor.shutdown(wait=True)
            elif use_parallel and parallel_mode == 'process':
                # PROCESS-POOL path: each worker is a separate Python process
                # with its own LPOracle (built once in _mp_worker_init).
                # Cache lookup + snap stay on the main process so there's no
                # shared mutable state between workers — the only inter-
                # process traffic is one list[float] in and one dict of
                # floats out per dispatched eval.
                from concurrent.futures import ProcessPoolExecutor, as_completed

                print(f"  [process mode] Spawning {n_workers} worker "
                      f"processes (each builds its own LPOracle, "
                      f"~30-60 s × {n_workers} one-time)...", flush=True)
                executor = ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_mp_worker_init,
                    initargs=(D,),
                )
                try:
                    _gen_n = 0
                    while not es.stop():
                        solutions = es.ask()
                        _gen_n += 1

                        # Reconstruct full-dim candidates, clip to bounds,
                        # and snap the P2G snap-zone. All pure Python on
                        # the main process.
                        xs_snapped = []
                        for y in solutions:
                            x_full = [0.0] * d
                            for k, i in enumerate(active_idx):
                                x_full[i] = y[k]
                            for i, v in frozen_vals.items():
                                x_full[i] = v
                            x_clipped = [
                                np.clip(x_full[i], bounds[i][0], bounds[i][1])
                                for i in range(d)
                            ]
                            xs_snapped.append(_snap_x(x_clipped))

                        # Check the local cache. Cache hits bypass the pool
                        # entirely; only misses are dispatched to workers.
                        outs = [None] * len(solutions)
                        submit_idx = []
                        for i, x in enumerate(xs_snapped):
                            hit = eval_cache.get(_cache_key(x))
                            if hit is not None:
                                cache_hits[0] += 1
                                outs[i] = hit
                            else:
                                submit_idx.append(i)

                        print(f"  [gen {_gen_n}] dispatching "
                              f"{len(submit_idx)}/{len(solutions)} evals to "
                              f"{n_workers} worker processes "
                              f"(countevals={es.countevals}, "
                              f"cache_hits={cache_hits[0]})", flush=True)

                        # Submit cache misses and collect via as_completed
                        # so slow LPs don't block fast ones.
                        if submit_idx:
                            futs = {
                                executor.submit(
                                    _mp_worker_solve, list(xs_snapped[i])
                                ): i
                                for i in submit_idx
                            }
                            for fut in as_completed(futs):
                                i = futs[fut]
                                try:
                                    r = fut.result()
                                except BaseException as _e:
                                    import traceback as _tb
                                    print(f"  [main] worker future raised "
                                          f"for index {i}: "
                                          f"{type(_e).__name__}: {_e}",
                                          flush=True)
                                    _tb.print_exc()
                                    r = None

                                # Build the standard eval tuple on the main
                                # process (matches _eval_one's contract).
                                x = xs_snapped[i]
                                if r is None:
                                    out = (1e15, None, None, 1.0, x)
                                else:
                                    vf_min_r = oracle.compute_target_min(r)  # active target metric
                                    cost = r['sys_cost']
                                    violation = max(0.0, vf_target - vf_min_r)
                                    penalized = cost + PENALTY * violation**2
                                    out = (penalized, r, vf_min_r, violation, x)
                                    # Attribute the worker's LP solve back to
                                    # the main oracle's counter so the final
                                    # "TOTAL LP SOLVES" report reflects total
                                    # work done (worker oracles have their own
                                    # counters that never reach the main log).
                                    oracle.solve_count += 1
                                eval_cache[_cache_key(x)] = out
                                outs[i] = out

                        # _process is race-free here — main process only.
                        fitnesses = [_process(out) for out in outs]
                        es.tell(solutions, fitnesses)
                        if es.countevals >= max_evals:
                            break
                finally:
                    executor.shutdown(wait=True)
            else:
                # Serial path: unchanged from the original structure.
                es.optimize(objective_reduced)

            # Capture CMA-ES termination state. ``es.stop()`` returns a dict
            # of triggered conditions — ``'maxfevals'`` means budget exhausted.
            # ``es.countevals`` counts f-evaluations including rejected ones.
            stop = dict(es.stop())
            opt_meta['n_evals'] = int(getattr(es, 'countevals', len(eval_log)))
            opt_meta['stop_reasons'] = list(stop.keys())
            opt_meta['budget_exhausted'] = 'maxfevals' in stop
            opt_meta['cache_hits'] = cache_hits[0]
            opt_meta['snap_skips'] = snap_skips[0]
            opt_meta['n_workers'] = n_workers if use_parallel else 1

    else:
        # Fallback: Nelder-Mead (original v9 implementation)
        simplex = np.zeros((d + 1, d))
        simplex[0] = x0
        for i in range(d):
            vertex = np.array(x0, dtype=float)
            step = 0.25 * (bounds[i][1] - bounds[i][0])
            vertex[i] = min(x0[i] + step, bounds[i][1])
            simplex[i + 1] = vertex

        nm_result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxfev': max_evals, 'xatol': 500, 'fatol': 1e6,
                     'adaptive': True, 'initial_simplex': simplex})
        # scipy's OptimizeResult: ``nit`` = iterations, ``nfev`` = f-evals,
        # ``success`` = True on convergence, ``message`` describes exit.
        opt_meta['n_evals'] = int(getattr(nm_result, 'nfev', len(eval_log)))
        # Nelder-Mead exhausts budget when success=False and the message
        # mentions 'Maximum number of function evaluations'.
        nm_msg = str(getattr(nm_result, 'message', ''))
        opt_meta['budget_exhausted'] = (
            not getattr(nm_result, 'success', True)
            and 'function evaluations' in nm_msg.lower()
        )
        opt_meta['stop_reasons'] = [nm_msg] if nm_msg else []

    # Record feasibility of the best_result BEFORE any closest-match fallback
    # so callers can distinguish "we met the target" from "we returned the
    # closest we found for informational purposes."
    opt_meta['feasible'] = feasible_seen and best_result.get('sys_cost', np.inf) < np.inf

    # Closest-match fallback: if no feasible point was found, return the
    # evaluation whose VF_min came closest to vf_target. Useful for the
    # global frontier as an informational "best effort" record — callers
    # should check opt_meta['feasible'] to tell these apart.
    # Filter out infeasible-LP entries (vf_min == -1.0 sentinel) so the
    # fallback can't pick an LP-infeasible point as "closest".
    if best_result.get('sys_cost', np.inf) == np.inf and eval_log:
        feasible_lp = [e for e in eval_log if e['vf_min'] >= 0]
        if feasible_lp:
            best_eval = min(feasible_lp, key=lambda e: abs(accept - e['vf_min']))
            r = oracle(best_eval['x'])
            if r is not None:
                best_result = r
                opt_meta['fallback_used'] = True

    # _opt_meta schema (documented here because callers will rely on it):
    #   optimizer         : 'cma' or 'nelder-mead'
    #   budget            : int, maxfevals passed in
    #   n_evals           : int, actual f-evaluations performed
    #   budget_exhausted  : bool, True if the optimizer hit maxfevals
    #   stop_reasons      : list[str], CMA-ES es.stop() keys / scipy msg
    #   feasible          : bool, True iff a point with VF_min >= vf_target
    #                       (within 1e-4 tol) was found AND returned
    #   fallback_used     : bool, True iff we returned a closest-match
    #                       (not feasible) result instead of infeasibility
    #   d_nominal         : int, nominal problem dimensionality
    #   d_active, d_frozen: dims searched by CMA-ES vs fixed at midpoint
    if isinstance(best_result, dict):
        best_result['_opt_meta'] = opt_meta

    return best_result, eval_log


def find_min_feasible_p2g(oracle, lo=1000, hi=100000, tol=500):
    """
    Binary search for the minimum P2G capacity that satisfies H₂ demand
    (with zero storage/DSM). Needed for a meaningful baseline.
    """
    print("  Finding minimum feasible P2G capacity (binary search)...")
    idx_sc = oracle.flex_dim_map.get((oracle.n_node, 'sc'))
    if idx_sc is None:
        print("  WARNING: No P2G dimension found for primary node — skipping.")
        return 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        x = [0.0] * oracle.n_dims
        x[idx_sc] = mid
        r = oracle(x)
        if r is not None:
            hi = mid
        else:
            lo = mid
    min_sc = hi
    print(f"  Minimum feasible P2G: ~{min_sc:,.0f} MW")
    return min_sc


def _extras_sha():
    try:
        import hashlib
        _d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tyndp 2040')
        _h = hashlib.sha256()
        for _f in ('tyndp_extras.json', 'tyndp_borders_dsr.json'):
            _p = os.path.join(_d, _f)
            if os.path.exists(_p):
                _h.update(open(_p, 'rb').read())
        return _h.hexdigest()[:16]
    except Exception:
        return ''
_EXTRAS_SHA = _extras_sha()


def _write_checkpoint(frontier_rows, path):
    """Atomically persist the frontier-so-far (S.56 crash resilience).
    Called after the baseline row and after EVERY completed VF target, so
    a killed run can be restarted with --resume without losing targets."""
    if not path:
        return
    try:
        tmp = path + '.tmp'
        pd.DataFrame(frontier_rows).to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  [checkpoint] WARNING: could not write {path}: {e}")


def _cost_subgradient(oracle, x0):
    """Exact dZ*/dx of the inner-LP value function at fixed capacities x0
    (S.60). Z*(x) is convex in x; its subgradient is the duals of equality
    PIN constraints K==x0 (solver-independent). The storage pin couples
    power+energy (E/P=4h). Returns (dz dict by outer-dim index, ref cost);
    restores the model's fixed state. dz empty on failure."""
    m = oracle.model
    r0 = oracle(x0)
    if r0 is None:
        return {}, np.inf
    cost_ref = r0['sys_cost']
    dz = {}
    try:
        if hasattr(m, 'HonestyPins'):
            m.del_component('HonestyPins')
            if hasattr(m, 'HonestyPins_index'):
                m.del_component('HonestyPins_index')
        m.HonestyPins = pyo.ConstraintList()
        pin_of = {}
        for i, info in enumerate(oracle.dim_info):
            if info['type'] == 'flex':
                node = info['node']
                v = oracle._get_flex_val(x0, node, info['instr'])
                if info['instr'] == 'sto':
                    m.K_sto_p[node].unfix(); m.K_sto_e[node].unfix()
                    pin_of[i] = m.HonestyPins.add(m.K_sto_p[node] == v)
                    m.HonestyPins.add(m.K_sto_e[node] == oracle.D.get('sto_ep_ratio', 4.0) * m.K_sto_p[node])
                elif info['instr'] == 'sc':
                    if node == oracle.n_node:
                        v = max(v, getattr(oracle, 'min_feasible_p2g', 0.0))
                    elif node == 'DK':
                        v = max(v, getattr(oracle, 'min_feasible_p2g_dk', 0.0))
                    m.K_sc[node].unfix()
                    pin_of[i] = m.HonestyPins.add(m.K_sc[node] == v)
                elif info['instr'] == 'dsm':
                    m.K_dsm[node].unfix()
                    pin_of[i] = m.HonestyPins.add(m.K_dsm[node] == v)
                elif info['instr'] == 'ev':
                    m.K_ev[node].unfix()
                    pin_of[i] = m.HonestyPins.add(m.K_ev[node] == v)
            else:
                l = info['link']
                v = oracle._get_ntc_val(x0, l)
                m.NTC_invest[l].unfix()
                pin_of[i] = m.HonestyPins.add(m.NTC_invest[l] == v)
        if not oracle._solve(m, dual_cons=list(pin_of.values())):
            raise RuntimeError('pin solve non-optimal')
        for i, con in pin_of.items():
            dz[i] = float(m.dual[con])
    except Exception as e:
        print(f"    [subgrad] pin-dual unavailable ({type(e).__name__}: {e})")
        dz = {}
    finally:
        try:
            m.del_component('HonestyPins')
            if hasattr(m, 'HonestyPins_index'):
                m.del_component('HonestyPins_index')
        except Exception:
            pass
        try:
            oracle(x0)
        except Exception:
            pass
    return dz, cost_ref


def _gradient_descent_refine(oracle, best, accept_target, max_iters=8,
                             max_lp=150, lower_bounds=None, verbose=True):
    """Convexity-aware local refinement (S.60). From a feasible incumbent,
    iteratively reduce cost while holding VF_min >= accept_target, using the
    EXACT cost subgradient dZ*/dx (pin duals). Two move types per iteration:
      (a) multi-dim accelerator: one step OPPOSITE the gradient over all
          dims with |dz|>0 (line-searched scale) -- fast when several dims
          can move together away from the VF boundary;
      (b) coordinate descent over the dims ranked by |dz|: shrink over-built
          dims (dz>0) / grow under-built dims (dz<0) ONE at a time -- this
          recovers VF-NEUTRAL over-building (e.g. DSM/EV/P2G slack) that the
          coupled step (a) rejects because it also perturbs VF-active dims.
    Every trial is a real LP; only strictly cost-reducing, VF-feasible moves
    are accepted. Exploits the convexity of the inner-LP value function;
    respects monotone lower_bounds; bounded by max_lp LP solves."""
    meta = {'iters': 0, 'lp_solves': 0, 'gain_bn': 0.0}
    if best is None or best.get('sys_cost', np.inf) == np.inf:
        return best, meta
    if oracle.compute_target_min(best) < accept_target - 1e-4:
        return best, meta
    x = [float(best.get(k, 0.0)) for k in oracle.x_keys]
    cost = best['sys_cost']; cur = best
    lb = lower_bounds if lower_bounds is not None else [0.0] * oracle.n_dims

    def _floor(i):
        f = float(lb[i]); info = oracle.dim_info[i]
        if info['type'] == 'flex' and info['instr'] == 'sc':
            if info['node'] == oracle.n_node:
                f = max(f, getattr(oracle, 'min_feasible_p2g', 0.0))
            elif info['node'] == 'DK':
                f = max(f, getattr(oracle, 'min_feasible_p2g_dk', 0.0))
        return f

    for it in range(max_iters):
        if meta['lp_solves'] >= max_lp:
            break
        dz, _cref = _cost_subgradient(oracle, x)
        meta['lp_solves'] += 1
        if not dz:
            break
        dirs = [(i, g) for i, g in dz.items()
                if abs(g) > 1.0 and oracle.bounds[i] > 1.0]
        if not dirs:
            break
        progressed = False

        # (a) multi-dim accelerator
        for scale in (0.12, 0.06, 0.03):
            if meta['lp_solves'] >= max_lp:
                break
            xp = list(x)
            for i, g in dirs:
                step = scale * oracle.bounds[i]
                if g > 0:
                    xp[i] = max(_floor(i), x[i] - step)
                else:
                    xp[i] = min(oracle.bounds[i], x[i] + step)
            rp = oracle(xp); meta['lp_solves'] += 1
            if rp is None:
                continue
            if (oracle.compute_target_min(rp) >= accept_target - 1e-4
                    and rp['sys_cost'] < cost - 1e6):
                x = xp; cost = rp['sys_cost']; cur = rp.copy()
                progressed = True
                break

        # (b) coordinate descent over dz-ranked dims (recovers VF-neutral slack)
        for i, g in sorted(dirs, key=lambda t: -abs(t[1])):
            if meta['lp_solves'] >= max_lp:
                break
            moved = False
            for scale in (0.20, 0.08, 0.03, 0.01):
                step = scale * oracle.bounds[i]
                xi = (max(_floor(i), x[i] - step) if g > 0
                      else min(oracle.bounds[i], x[i] + step))
                if abs(xi - x[i]) < 1.0:
                    continue
                xp = list(x); xp[i] = xi
                rp = oracle(xp); meta['lp_solves'] += 1
                if rp is None:
                    continue
                if (oracle.compute_target_min(rp) >= accept_target - 1e-4
                        and rp['sys_cost'] < cost - 1e6):
                    x = xp; cost = rp['sys_cost']; cur = rp.copy()
                    progressed = True; moved = True
                    break
            # if a coordinate move worked, the next dim sees the updated x
        meta['iters'] = it + 1
        if not progressed:
            break

    meta['gain_bn'] = (best['sys_cost'] - cost) / 1e9
    if verbose and meta['gain_bn'] > 1e-4:
        print(f"    [refine] convex descent: -{meta['gain_bn']:.3f} bn over "
              f"{meta['iters']} iter(s), {meta['lp_solves']} LP solves")
    return cur, meta

def _honesty_check(oracle, best, accept_target, max_probes=6,
                   lower_bounds=None):
    """First-order optimizer-honesty check at a candidate frontier point
    (S.56). Z*(x) is CONVEX in the fixed capacities x (LP value function
    w.r.t. variable fixings), and the reduced costs of the fixed variables
    are an exact subgradient dZ*/dx -- free information the heuristic
    search never used. At a genuine least-cost frontier point there is no
    feasible direction that lowers cost while keeping the target metric
    above accept_target.

    Procedure: re-solve at x_best (refreshes duals/rc), read the exact
    per-dimension cost gradient, rank candidate descent moves by expected
    gain |dZ/dx| * step, and VERIFY up to max_probes of them with real LP
    solves (cost must fall AND target metric must hold). A verified
    improvement replaces the point (free cost reduction) and is recorded
    in the CSV -- nonzero values are direct evidence of non-convergence,
    ~zero values are publishable evidence of (local first-order)
    optimality.

    Returns (result dict -- possibly an improved point, meta dict)."""
    meta = {'rc_available': False, 'n_descent': 0, 'n_probes': 0,
            'improved_bn': 0.0}
    x0 = [float(best.get(k, 0.0)) for k in oracle.x_keys]
    dz, cost_ref = _cost_subgradient(oracle, x0)   # exact dZ*/dx (S.60)
    if not dz:
        return best, meta
    meta['rc_available'] = True
    if best.get('sys_cost', np.inf) < cost_ref:
        cost_ref = best['sys_cost']

    # Candidate descent moves (move OPPOSITE the cost gradient, within box;
    # respect the DE/DK P2G floors -- below them the oracle snaps anyway)
    cands = []
    for i, g in dz.items():
        ub = oracle.bounds[i]
        if ub <= 1.0:
            continue
        step = max(500.0, 0.02 * ub)
        info = oracle.dim_info[i]
        floor = float(lower_bounds[i]) if lower_bounds is not None else 0.0
        if info['type'] == 'flex' and info['instr'] == 'sc':
            if info['node'] == oracle.n_node:
                floor = max(floor, getattr(oracle, 'min_feasible_p2g', 0.0))
            elif info['node'] == 'DK':
                floor = max(floor, getattr(oracle, 'min_feasible_p2g_dk', 0.0))
        if g > 1.0 and x0[i] - step >= floor:          # cheaper to SHRINK
            cands.append((g * step, i, -step))
        elif g < -1.0 and x0[i] + step <= ub:          # cheaper to GROW
            cands.append((-g * step, i, +step))
    meta['n_descent'] = len(cands)
    cands.sort(reverse=True)

    best_imp = None
    for gain_est, i, dstep in cands[:max_probes]:
        xp = list(x0)
        xp[i] = min(max(xp[i] + dstep, 0.0), oracle.bounds[i])
        rp = oracle(xp)
        meta['n_probes'] += 1
        if rp is None:
            continue
        if (oracle.compute_target_min(rp) >= accept_target - 1e-4
                and rp['sys_cost'] < cost_ref - 1e5):
            if best_imp is None or rp['sys_cost'] < best_imp['sys_cost']:
                best_imp = rp.copy()
                meta['improved_dim'] = oracle.x_keys[i]
    if best_imp is not None:
        meta['improved_bn'] = (cost_ref - best_imp['sys_cost']) / 1e9
        print(f"    [honesty] IMPROVEMENT found: -{meta['improved_bn']:.3f} bn "
              f"via {meta.get('improved_dim')} -> point was NOT converged; "
              f"taking the improved point")
        return best_imp, meta
    print(f"    [honesty] first-order check passed "
          f"({meta['n_descent']} descent candidates, "
          f"{meta['n_probes']} verified, no feasible cost reduction)")
    return best, meta


def _next_vf_target(history, base, baseline_cost_bn=None, step0=0.03,
                    step_fine=0.025, step_coarse=0.045, improve_eps=0.005,
                    noise_bn=0.4, noise_frac=0.004, max_targets=22, vf_cap=1.0):
    """Dynamic, scenario-agnostic VF-target generator (None to stop). history
    items: {'target','achieved','premium'} in attempt order. Climb from
    base+step0; COARSE step in the flat bottom (premium below the noise floor),
    FINE step on the elbow (premium above it) so resolution concentrates where
    the curve bends. Stop after 2 CONSECUTIVE targets fail to raise achieved
    VF_min by > improve_eps (the proven anti-false-ceiling rule); ceiling = max
    achieved VF, already reported by best-of-restarts on overshoot. Pure /
    unit-tested -- reused unchanged across all scenarios and vf/vfw targets."""
    # S.103: optional finer grid (env GRID_STEP_MULT, e.g. 0.5); unset = 1 = unchanged
    _gm = float(os.environ.get('GRID_STEP_MULT', '1') or 1)
    step0, step_fine, step_coarse = step0 * _gm, step_fine * _gm, step_coarse * _gm
    noise = max(noise_bn, noise_frac * baseline_cost_bn) if baseline_cost_bn else noise_bn
    if not history:
        f = round(base + step0, 4)
        return f if f < vf_cap else None
    if len(history) >= max_targets:
        return None
    run_best = base
    improved = []
    for h in history:
        improved.append(h['achieved'] > run_best + improve_eps)
        run_best = max(run_best, h['achieved'])
    trailing_stalls = 0
    for im in reversed(improved):
        if im:
            break
        trailing_stalls += 1
    if trailing_stalls >= 2:
        return None
    highest = max(h['target'] for h in history)
    imp = [h for h, i in zip(history, improved) if i]
    last_prem = imp[-1]['premium'] if imp else 0.0
    nstep = step_coarse if last_prem < noise else step_fine
    nxt = round(highest + nstep, 4)
    if nxt >= vf_cap:
        nxt = round(vf_cap - 1e-4, 4)
    if round(nxt, 4) in {round(h['target'], 4) for h in history}:
        return None
    return nxt


def trace_pareto_frontier(D, vf_targets=None, optimizer='cma',
                          n_workers=1, oracle_pool=None,
                          parallel_mode='process',
                          max_evals_override=None,
                          cma_seed=42, target_metric='vf', n_restarts=1,
                          target_slack=0.0, polish_evals=0, honesty_evals=0,
                          refine_iters=0,
                          checkpoint_path=None, resume=False):
    """
    Sweeps VF targets from low to high, warm-starting each solve
    from the previous solution. Targets min(VF_wind, VF_pv).

    VF targets widened to accommodate PV (which had VF~0.60 at baseline
    in v4). The optimizer now needs to bring PV up, not just wind.

    Parameters:
      optimizer: 'cma' (CMA-ES, default) or 'nelder-mead' (scipy fallback).
      n_workers: int, parallel LP workers for CMA-ES (default 1).
      oracle_pool: list[LPOracle] for parallel eval, THREAD mode only.
                   Ignored in PROCESS mode (workers build their own oracles).
      parallel_mode: 'process' (default, reliable) or 'thread' (faster
                   startup, but fails after a few hundred solves due to
                   shared Gurobi C-state corruption).
      max_evals_override: int or None. If set, uses this budget for EVERY
                         target (overrides the default 1200/1800 split).
                         Used by the single-target diagnostic mode.
    """
    _auto_grid = (vf_targets is None and target_metric in ('vf', 'vfw'))
    if vf_targets is None and target_metric == 'cr':
        vf_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]

    # Multi-start threshold: above this VF, we run from multiple starting
    # points to avoid Nelder-Mead collapsing the storage dimension.
    MULTISTART_VF = 0.88

    oracle = LPOracle(D)
    oracle.target_metric = target_metric

    # Build oracle pool for parallel eval — THREAD MODE ONLY.
    # Each oracle is an independent Pyomo model — N oracles = N models
    # built (cost ~30-60s per model for 5-node 1008h instances).
    # In PROCESS mode each worker builds its own oracle in _mp_worker_init,
    # so the main process needs only the single ``oracle`` above for
    # baseline / compute_vf_min / format_capacities work.
    if n_workers > 1 and parallel_mode == 'thread' and oracle_pool is None:
        print(f"\n  [thread mode] Building oracle pool: {n_workers-1} "
              f"additional LPOracle instance(s)...")
        extras = []
        for _i in range(n_workers - 1):
            extras.append(LPOracle(D))
        oracle_pool = [oracle] + extras
        print(f"  Oracle pool ready ({len(oracle_pool)} instances).")

    # ---- COST-OPTIMAL BASELINE B_opt (progress.md Section 36) ----
    # The global frontier's reference is the cost-minimizing system: VRE
    # fixed, ALL flexibility free for the LP. premium(theta) = cost(theta)
    # - cost_opt is then >= 0 and monotone. This replaces the old zero-
    # flexibility strawman baseline, which manufactured a spurious
    # "free lunch zone" by starting from a non-cost-optimal system.
    baseline = oracle.solve_cost_optimal()

    if baseline is None:
        print("  ERROR: Could not compute cost-optimal baseline B_opt!")
        print("  (This shouldn't happen — check model constraints.)")
        return pd.DataFrame(), 0.0

    # natural_vf_min holds the CR_min of B_opt -- the optimization target
    # metric (Section 34). The name is kept for minimal diff; it is CR.
    natural_vf_min = oracle.compute_target_min(baseline)
    natural_vf_display = oracle.compute_vf_min(baseline)

    if _auto_grid:
        base = natural_vf_min
        # DYNAMIC grid (scenario-agnostic): only the first hop is fixed here; the
        # sweep below EXTENDS active_targets on the fly via _next_vf_target, which
        # adapts the step to the premium curve and stops at the ceiling (2-stall).
        _first = round(base + 0.03 * float(os.environ.get('GRID_STEP_MULT', '1') or 1), 4)
        vf_targets = [_first] if _first < 1.0 else []
        print(f"  Dynamic VF grid (natural VF_min={base:.3f}): adaptive climb "
              f"from {_first}; ceiling emerges via the 2-stall stop.")

    print(f"\n  COST-OPTIMAL BASELINE B_opt (VRE fixed, flexibility LP-optimal):")
    print(f"    System cost:  {baseline['sys_cost']/1e9:.3f} bn EUR")
    for n in oracle.all_nodes:
        vf_strs = ', '.join(
            f'CR_{v}={baseline.get(f"cr_{v}_{n}", 0):.3f}(VF={baseline.get(f"vf_{v}_{n}", 0):.3f})'
            for v in D['tech_vre'])
        print(f"    {n}: {vf_strs}")
    print(f"    Optimization target = {oracle.target_metric.upper()}_min = {natural_vf_min:.4f}")
    print(f"    (CR_min={oracle.compute_cr_min(baseline):.4f}, VF_min={oracle.compute_vf_min(baseline):.4f}; both always reported)")
    baseline_cost = baseline['sys_cost']
    # CfD-equivalent subsidy bill at B_opt (S.56 welfare layer): the sum of
    # missing money across all VRE. Frontier rows report the AVOIDED subsidy
    # vs this reference and the net benefit under a marginal-cost-of-public-
    # funds (MCF) of 0 and 0.3.
    baseline_subsidy = baseline.get('subsidy_bill_eur', 0.0)
    # Warm-start vector for the first target = B_opt's LP-chosen capacities.
    x_baseline = [baseline.get(k, 0.0) for k in oracle.x_keys]

    # Filter targets: skip any theta at/below the cost-optimal CR_min
    active_targets = [t for t in vf_targets if t > natural_vf_min + 0.005]
    if len(active_targets) < len(vf_targets):
        skipped = len(vf_targets) - len(active_targets)
        print(f"\n  Skipping {skipped} theta targets at/below cost-optimal CR_min={natural_vf_min:.3f}")

    # Helper to build a frontier row from a result dict
    def _frontier_row(r, vf_target_val, premium):
        caps = oracle.format_capacities(r)
        row = {
            'vf_target': vf_target_val,          # target threshold (in target_metric units)
            'target_metric': oracle.target_metric,
            'target_min': oracle.compute_target_min(r),  # ACTIVE constraint metric value
            'cr_min': oracle.compute_cr_min(r),  # cost-recovery min (always reported)
            'vf_min': oracle.compute_vf_min(r),  # value-factor min (always reported)
            'cr_wind_on': r.get('cr_wind_on', 0),
            'cr_wind_off': r.get('cr_wind_off', 0),
            'cr_pv': r.get('cr_pv', 0),
            'vf_wind_on': r.get('vf_wind_on', 0),
            'vf_wind_off': r.get('vf_wind_off', 0),
            'vf_pv': r.get('vf_pv', 0),
            **caps,
            'sys_cost': r['sys_cost'],
            'premium_bn': premium,
        }
        # Add per-node VF / CR / missing-money for multi-node
        for n in oracle.all_nodes:
            for v in D['tech_vre']:
                for pref in ('vf_', 'cr_', 'mm_'):
                    key = f'{pref}{v}_{n}'
                    if key in r:
                        row[key] = r[key]
        # Heat-pump metrics if reported by LPOracle (Tier 2 #6)
        for hp_key in ('hp_dk_total_twh_e', 'hp_dk_avg_load_mw_e',
                       'hp_dk_backup_twh_th', 'hp_dk_soc_max_mwh',
                       'hp_dk_soc_mean_mwh'):
            if hp_key in r:
                row[hp_key] = r[hp_key]
        # CR = VF * (base_price / LCOE) decomposition (progress.md 34.3):
        # price-level factor base/LCOE per node/tech, so the VF->CR
        # relationship is reconstructable from the frontier CSV.
        for nn in oracle.all_nodes:
            bp = r.get(f'base_price_{nn}', r.get('base_price', 0.0))
            for vv in D['tech_vre']:
                lc = r.get(f'lcoe_{vv}_{nn}', 0.0)
                row[f'plf_{vv}_{nn}'] = (bp / lc) if lc > 1e-9 else 0.0
            row[f'base_price_{nn}'] = bp          # S.103: nodal average price [EUR/MWh]
        # ---- S.56 system aggregates + welfare (flexibility-vs-subsidy) ----
        row['vf_genw'] = r.get('vf_genw')
        row['cannib_loss_bn'] = r.get('cannib_loss_eur', 0.0) / 1e9
        row['subsidy_bill_bn'] = r.get('subsidy_bill_eur', 0.0) / 1e9
        row['avoided_subsidy_bn'] = (baseline_subsidy
                                     - r.get('subsidy_bill_eur', 0.0)) / 1e9
        # Net social benefit of the flexibility path vs paying CfD top-ups:
        # (1 + MCF) * avoided_subsidy - premium. MCF = marginal cost of
        # public funds (0 = pure transfer view; 0.3 = standard deadweight).
        row['net_benefit_mcf0_bn'] = row['avoided_subsidy_bn'] - premium
        row['net_benefit_mcf30_bn'] = 1.3 * row['avoided_subsidy_bn'] - premium
        # Residual-interconnector boundary diagnostics (price exogeneity)
        for kk, vv in r.items():
            if isinstance(kk, str) and kk.startswith('resid_'):
                row[kk] = vv
        # First-order honesty check results, if run
        hm = r.get('_honesty_meta')
        if hm:
            row['honesty_rc_available'] = hm.get('rc_available')
            row['honesty_n_descent'] = hm.get('n_descent')
            row['honesty_n_probes'] = hm.get('n_probes')
            row['honesty_improved_bn'] = hm.get('improved_bn')
        rfm = r.get('_refine_meta')
        if rfm:
            row['refine_iters'] = rfm.get('iters')
            row['refine_lp_solves'] = rfm.get('lp_solves')
            row['refine_gain_bn'] = rfm.get('gain_bn')
        # multi-restart optimality-gap proxy, if attached by the caller
        rm = r.get('_restart_meta')
        if rm:
            row['n_restarts'] = rm.get('n_restarts')
            row['n_feasible_restarts'] = rm.get('n_feasible')
            row['restart_gap_pct'] = rm.get('gap_pct')
            row['cost_min_bn'] = (rm['cost_min'] / 1e9) if rm.get('cost_min') else None
            row['cost_max_bn'] = (rm['cost_max'] / 1e9) if rm.get('cost_max') else None
            if 'polish_gain_bn' in rm:
                row['polish_gain_bn'] = rm.get('polish_gain_bn')
        return row

    # Include baseline as the first frontier point
    frontier = [_frontier_row(baseline, round(natural_vf_min, 3), 0.0)]
    history = []   # dynamic-grid attempts (auto-grid mode only)

    x_current = np.array(x_baseline, dtype=float)

    # Best-so-far tracking: ensures the frontier is monotonically
    # non-decreasing in VF_min. If target N+1 finds a worse VF_min than
    # target N, we keep the previous best as the frontier point and
    # warm-start. This prevents non-monotone artifacts from CMA-ES
    # landing in different basins at adjacent targets.
    best_so_far_vf = natural_vf_min
    best_so_far_result = baseline.copy()
    _ceiling_stall = 0            # consecutive targets with no VF_min improvement
    _prev_achieved = natural_vf_min
    best_so_far_x = np.array(x_baseline, dtype=float)

    # ---- RESUME from a per-target checkpoint (S.56) -------------------
    # The checkpoint CSV holds the baseline row + every completed target.
    # On --resume we (a) verify the stored baseline matches the freshly
    # solved one (model unchanged), (b) reload completed rows, (c) skip
    # their targets, (d) rebuild the warm-start state with ONE extra LP
    # solve at the last stored optimum.
    if checkpoint_path and not (resume and os.path.exists(checkpoint_path)):
        # fresh start: old restart caches of this run id must never be reused (S.105)
        import glob as _glob
        for _old in _glob.glob(checkpoint_path.replace('checkpoint_frontier_', 'restart_cache_').replace('.csv', '') + '_t*.pkl'):
            try:
                os.remove(_old)
            except OSError:
                pass
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            _df_ck = pd.read_csv(checkpoint_path)
        except Exception as _e:
            print(f"  RESUME: could not read checkpoint ({_e}) -- starting fresh.")
            _df_ck = None
        if _df_ck is not None and len(_df_ck) >= 1:
            _ck_base = float(_df_ck.iloc[0]['sys_cost'])
            _drift = abs(_ck_base - baseline_cost) / max(abs(baseline_cost), 1.0)
            if _drift > 0.005:
                raise RuntimeError(
                    f"RESUME ABORTED: checkpoint baseline cost "
                    f"{_ck_base/1e9:.3f} bn differs from the fresh baseline "
                    f"{baseline_cost/1e9:.3f} bn by {100*_drift:.2f}% -- the "
                    f"model/scenario has changed since the checkpoint was "
                    f"written. Delete {checkpoint_path} to start over.")
            if _drift > 1e-9:
                print(f"  RESUME: baseline drift {100*_drift:.4f}% (tolerated)")
            if len(_df_ck) > 1:
                frontier = _df_ck.to_dict('records')
                _tmins = [float(rr.get('target_min', 0.0)) for rr in frontier[1:]]
                _last_target = max(float(rr.get('vf_target', 0.0))
                                   for rr in frontier[1:])
                best_so_far_vf = max([natural_vf_min] + _tmins)
                _prev_achieved = best_so_far_vf
                _ceiling_stall = 0
                _prev = natural_vf_min
                for _tv in _tmins:
                    _ceiling_stall = _ceiling_stall + 1 if _tv <= _prev + 0.005 else 0
                    _prev = max(_prev, _tv)
                if _auto_grid:
                    # dynamic grid: rebuild history -> next target from generator
                    history = [{'target': float(rr.get('vf_target', 0.0)),
                                'achieved': float(rr.get('target_min',
                                            rr.get('vf_min', 0.0)) or 0.0),
                                'premium': float(rr.get('premium_bn', 0.0) or 0.0)}
                               for rr in frontier[1:]]
                    _nxt = _next_vf_target(history, natural_vf_min,
                                           baseline_cost_bn=baseline_cost / 1e9)
                    active_targets = ([_nxt] if (_nxt is not None
                                      and _nxt > _last_target + 1e-9) else [])
                    _nxt_msg = (f"next dynamic target = "
                                f"{active_targets[0] if active_targets else 'none (ceiling)'}")
                else:
                    _n_before = len(active_targets)
                    active_targets = [t for t in active_targets
                                      if t > _last_target + 1e-9]
                    _nxt_msg = (f"skipping {_n_before - len(active_targets)} done; "
                                f"{len(active_targets)} remain")
                _last = frontier[-1]
                x_current = np.array(
                    [float(_last.get(k, 0.0) or 0.0) for k in oracle.x_keys])
                best_so_far_x = x_current.copy()
                print(f"  RESUME: loaded {len(frontier)-1} completed target(s) "
                      f"up to {_last_target:.3f}; {_nxt_msg}. Rebuilding "
                      f"warm-start state (1 LP solve)...")
                _r_ws = oracle(list(x_current))
                if _r_ws is not None:
                    best_so_far_result = _r_ws
    _write_checkpoint(frontier, checkpoint_path)

    for vf_star in active_targets:
        print(f"\n  --- Target {oracle.target_metric.upper()}_min >= {vf_star:.3f} ---")
        t0 = time.time()

        if optimizer == 'cma':
            # CMA-ES with MULTI-RESTART (referee-defensible heuristic honesty,
            # progress.md T1.3). Run n_restarts independent searches from
            # different seeds, keep the cheapest TARGET-FEASIBLE result, and
            # record the cost spread (cost_max-cost_min)/cost_min as an
            # empirical optimality-gap proxy. n_restarts=1 == prior behaviour.
            d = oracle.n_dims
            # Inflated INTERNAL target (S.56): the quadratic penalty makes
            # CMA undershoot each target by ~0.02, biasing every frontier
            # point left. Aim target_slack ABOVE the true target; acceptance
            # stays at the true target (accept_target below).
            vf_internal = vf_star + max(0.0, target_slack)
            if max_evals_override is not None:
                max_ev = max_evals_override
            else:
                max_ev = 1200 if vf_star < 0.90 else 1800
            restart_costs = []
            best = {'sys_cost': np.inf}       # cheapest target-feasible restart
            best_any = {'sys_cost': np.inf}   # cheapest of any restart (ceiling fallback)
            evals = 0
            # Warm-start SEEDING (progress.md S.44): B_opt builds ~0 storage, so
            # a pure warm-start strands CMA in the low-storage basin in high
            # dimension and yields FALSE ceilings. Seed alternate restarts from
            # HIGH-storage points so the search reaches the storage-heavy region
            # that high VF requires. Restart 0 = warm-start; others = high-sto.
            def _seed_all(x_base, frac):
                # Seed ALL flexibility dims high (storage, P2G, DSM, EV, NTC),
                # not just storage: high VF needs the full flex set (S.54), and
                # storage-only seeds let CMA drift back to a low-flex basin ->
                # false ~0.68 ceiling at 26-D.
                xs = list(x_base)
                for _i in range(len(xs)):
                    xs[_i] = max(xs[_i], frac * oracle.bounds[_i])
                return xs
            _seeds = [list(x_current), _seed_all(x_current, 0.60),
                      _seed_all(x_current, 0.40), _seed_all(x_current, 0.80)]
            for _si in range(max(1, n_restarts)):
                x0_si = _seeds[_si % len(_seeds)]
                # S.105 crash protection: each finished restart is cached on
                # disk; after a crash + --resume, finished restarts of the
                # interrupted target are reloaded instead of recomputed.
                _rc = None
                if checkpoint_path:
                    _rc = (checkpoint_path.replace('checkpoint_frontier_', 'restart_cache_')
                           .replace('.csv', '') + f'_t{vf_star:.4f}_r{_si}.pkl')
                b = None; log = []
                _fp = (round(float(baseline_cost), 0), os.environ.get('TYNDP_FIXES', ''),
                       _EXTRAS_SHA, round(float(vf_star), 4), int(_si), int(max_ev),
                       round(float(vf_internal), 4))
                if _rc and resume and os.path.exists(_rc):
                    try:
                        import pickle as _pk
                        with open(_rc, 'rb') as _fh:
                            _obj = _pk.load(_fh)
                        if isinstance(_obj, dict) and _obj.get('fp') == _fp:
                            b = _obj['b']
                            print(f"    [restart cache] restart {_si+1} of target {vf_star:.4f} reloaded")
                        else:
                            print(f"    [restart cache] stale cache for restart {_si+1} ignored")
                    except Exception as _e:
                        print(f"    [restart cache] could not reload ({_e}); recomputing")
                        b = None
                if b is None:
                    b, log = find_min_cost_portfolio(
                        oracle, vf_internal, x0=x0_si, max_evals=max_ev,
                        optimizer='cma',
                        n_workers=n_workers, oracle_pool=oracle_pool,
                        parallel_mode=parallel_mode, D=D,
                        cma_seed=cma_seed + _si,
                        accept_target=vf_star,
                    )
                    if _rc:
                        try:
                            import pickle as _pk
                            _tmp = _rc + '.tmp'
                            with open(_tmp, 'wb') as _fh:
                                _pk.dump({'fp': _fp, 'b': b}, _fh)
                            os.replace(_tmp, _rc)
                        except Exception as _e:
                            print(f"    [restart cache] not written ({_e})")
                evals += len(log)
                if b and b.get('sys_cost', np.inf) < np.inf:
                    # Prefer the cheapest TARGET-FEASIBLE restart for `best`.
                    # Only fall back to an infeasible closest-match (best_any)
                    # if NO restart met the target (= true ceiling). Otherwise
                    # a cheaper-but-infeasible restart could be recorded as the
                    # frontier point and warm-start the next target (bug C4).
                    if b['sys_cost'] < best_any.get('sys_cost', np.inf):
                        best_any = b
                    if oracle.compute_target_min(b) >= vf_star - 1e-3:
                        restart_costs.append(b['sys_cost'])
                        if b['sys_cost'] < best.get('sys_cost', np.inf):
                            best = b
            if best.get('sys_cost', np.inf) == np.inf:
                best = best_any   # no feasible restart -> ceiling marker
            if restart_costs:
                cmin, cmax = min(restart_costs), max(restart_costs)
                gap_pct = 100.0 * (cmax - cmin) / cmin if cmin > 0 else 0.0
            else:
                cmin = cmax = None
                gap_pct = float('nan')
            if isinstance(best, dict) and best.get('sys_cost', np.inf) < np.inf:
                best['_restart_meta'] = {
                    'n_restarts': max(1, n_restarts),
                    'n_feasible': len(restart_costs),
                    'gap_pct': gap_pct,
                    'cost_min': cmin, 'cost_max': cmax,
                }
            if n_restarts > 1:
                print(f"    Multi-restart: {len(restart_costs)}/{n_restarts} "
                      f"feasible; empirical optimality gap = {gap_pct:.2f}% "
                      f"(cost spread across restarts)")

            # ---- Saturation-freeze polish (S.56) ----------------------
            # In every production run several dims sit AT their caps
            # (DSM/EV, some NTC). Freeze dims within 2% of their upper
            # bound at the incumbent value and re-run a short CMA on the
            # remaining low-dimensional subspace -- CMA is reliable in low
            # dim (S.39), so this recovers cost the 26-D search left on
            # the table.
            if (polish_evals > 0
                    and best.get('sys_cost', np.inf) < np.inf
                    and oracle.compute_target_min(best) >= vf_star - 1e-3):
                _xb = [float(best.get(k, 0.0)) for k in oracle.x_keys]
                _pb, _nfroz = [], 0
                for _i, _xv in enumerate(_xb):
                    _ubi = oracle.bounds[_i]
                    if _ubi > 0 and _xv >= 0.98 * _ubi:
                        _pb.append((_xv, _xv)); _nfroz += 1
                    else:
                        _pb.append((0.0, _ubi))
                if 0 < _nfroz < len(_xb):
                    print(f"    Polish: freezing {_nfroz} saturated dim(s), "
                          f"re-optimizing the remaining "
                          f"{len(_xb) - _nfroz} (budget {polish_evals})")
                    _b2, _log2 = find_min_cost_portfolio(
                        oracle, vf_internal, x0=_xb, bounds=_pb,
                        max_evals=polish_evals, optimizer='cma',
                        n_workers=n_workers, oracle_pool=oracle_pool,
                        parallel_mode=parallel_mode, D=D,
                        cma_seed=cma_seed + 977,
                        accept_target=vf_star,
                    )
                    evals += len(_log2)
                    if (_b2 and _b2.get('sys_cost', np.inf)
                            < best['sys_cost'] - 1e5
                            and oracle.compute_target_min(_b2)
                            >= vf_star - 1e-3):
                        _gain = (best['sys_cost'] - _b2['sys_cost']) / 1e9
                        print(f"    Polish gain: -{_gain:.3f} bn EUR")
                        _b2['_restart_meta'] = {
                            **(best.get('_restart_meta') or {}),
                            'polish_gain_bn': _gain}
                        _b2['_opt_meta'] = best.get('_opt_meta', {})
                        best = _b2
                    else:
                        if isinstance(best.get('_restart_meta'), dict):
                            best['_restart_meta']['polish_gain_bn'] = 0.0

            # ---- Convexity-aware gradient-descent refinement (S.60) ----
            if (refine_iters > 0
                    and best.get('sys_cost', np.inf) < np.inf
                    and oracle.compute_target_min(best) >= vf_star - 1e-3):
                _rm_keep = best.get('_restart_meta')
                _om_keep = best.get('_opt_meta')
                _bestr, _rmeta = _gradient_descent_refine(
                    oracle, best, vf_star, max_iters=refine_iters)
                if _bestr is not best:
                    if _rm_keep is not None:
                        _bestr['_restart_meta'] = _rm_keep
                    if _om_keep is not None:
                        _bestr['_opt_meta'] = _om_keep
                    best = _bestr
                best['_refine_meta'] = _rmeta

            # ---- First-order honesty CHECK (S.56), now a convergence cert -
            if (honesty_evals > 0
                    and best.get('sys_cost', np.inf) < np.inf
                    and oracle.compute_target_min(best) >= vf_star - 1e-3):
                _rm_keep = best.get('_restart_meta')
                _om_keep = best.get('_opt_meta')
                _best2, _hmeta = _honesty_check(
                    oracle, best, vf_star, max_probes=honesty_evals)
                if _best2 is not best:
                    if _rm_keep is not None:
                        _best2['_restart_meta'] = _rm_keep
                    if _om_keep is not None:
                        _best2['_opt_meta'] = _om_keep
                    best = _best2
                best['_honesty_meta'] = _hmeta
        elif vf_star >= MULTISTART_VF:
            # Nelder-Mead multi-start for high VF targets
            evals_per_start = 80 if vf_star < 0.93 else 120
            def _seed(frac=0.5):
                """Generate a seed at `frac` of each dimension's upper bound."""
                return np.array([oracle.bounds[j] * frac for j in range(oracle.n_dims)])
            starts = [
                ('warm', x_current),
                ('low',  _seed(0.3)),
                ('high', _seed(0.6)),
                ('mid',  _seed(0.5)),
            ]
            best_overall = {'sys_cost': np.inf}
            best_label = '(none)'
            total_evals = 0
            for label, x0_start in starts:
                b, log = find_min_cost_portfolio(
                    oracle, vf_star, x0=x0_start, max_evals=evals_per_start,
                    optimizer='nelder-mead'
                )
                total_evals += len(log)
                if b and b.get('sys_cost', np.inf) < np.inf:
                    vf_b = oracle.compute_target_min(b)  # active target metric
                    caps_b = oracle.format_capacities(b)
                    cap_str_b = ", ".join(f"{k}={v:,.0f}" for k, v in caps_b.items())
                    print(f"      {label}: VF_min={vf_b:.4f}, {cap_str_b}, "
                          f"cost={b['sys_cost']/1e9:.3f}bn")
                    if vf_b >= vf_star - 1e-4 and b['sys_cost'] < best_overall.get('sys_cost', np.inf):
                        best_overall = b
                        best_label = label
                else:
                    print(f"      {label}: INFEASIBLE or no solution")
            best = best_overall
            evals = total_evals
            if best.get('sys_cost', np.inf) < np.inf:
                print(f"    (best from multi-start: '{best_label}')")
        else:
            best, log = find_min_cost_portfolio(
                oracle, vf_star, x0=x_current, max_evals=120,
                optimizer='nelder-mead'
            )
            evals = len(log)

        dt = time.time() - t0

        if best and best.get('sys_cost', np.inf) < np.inf:
            premium = (best['sys_cost'] - baseline_cost) / 1e9
            vf_min = oracle.compute_target_min(best)  # active target metric
            caps = oracle.format_capacities(best)
            cap_str = ", ".join(f"{k}={v:,.0f}" for k, v in caps.items())
            print(f"    Found: {cap_str}")
            for nd in oracle.all_nodes:
                vf_strs = ', '.join(f'VF_{v}={best.get(f"vf_{v}_{nd}", 0):.4f}' for v in D['tech_vre'])
                print(f"    {nd}: {vf_strs}")
            print(f"    Target {oracle.target_metric.upper()}_min={vf_min:.4f}, Premium={premium:.3f} bn EUR (vs B_opt)")
            print(f"    LP solves: {evals}, Time: {dt:.1f}s")
            _rm = best.get('_restart_meta', {}) or {}
            try:
                import queue_progress as _qp
                _pre, _suf = _qp.on_target(dt)
            except Exception:
                _pre, _suf = '', ''
            _feas = (f", {_rm.get('n_feasible')}/{_rm.get('n_restarts')} restarts feasible"
                     if _rm.get('n_restarts') else '')
            notify((_pre + '\n' if _pre else '')
                   + f"{oracle.target_metric}-target {vf_star:.3f}: "
                   f"{oracle.target_metric.upper()}_min={vf_min:.3f}, "
                   f"premium={premium:.2f} bn, "
                   f"gap={_rm.get('gap_pct', float('nan')):.1f}%{_feas}, {dt/3600:.1f} h"
                   + ('\n' + _suf if _suf else ''))
            # Optimizer termination status — flags budget exhaustion and
            # whether the VF target was actually achieved vs closest-match.
            _print_opt_status(best.get('_opt_meta', {}))

            # Best-so-far monotonicity enforcement:
            # If this target found a LOWER VF_min than the previous best,
            # the optimizer landed in a worse basin. We record the current
            # result (it's the best for THIS target) but keep the previous
            # best as warm-start so the next target doesn't inherit the
            # regression.  For the frontier, we take whichever has higher
            # VF_min; if tied, whichever is cheaper.
            if vf_min > best_so_far_vf - 1e-4:
                # New result is at least as good — normal case
                frontier.append(_frontier_row(best, vf_star, premium))
                _write_checkpoint(frontier, checkpoint_path)
                x_current = np.array([best[k] for k in oracle.x_keys], dtype=float)
                best_so_far_vf = vf_min
                best_so_far_result = best.copy()
                best_so_far_x = x_current.copy()
            else:
                # Regression: this target achieved LESS than previous best.
                # Record the previous best at this VF target (it dominates),
                # and keep warm-starting from the previous best.
                prev_premium = (best_so_far_result['sys_cost'] - baseline_cost) / 1e9
                print(f"    ** Monotonicity fix: CR_min={vf_min:.4f} < best_so_far={best_so_far_vf:.4f}")
                print(f"       Keeping previous best (VF_min={best_so_far_vf:.4f}, "
                      f"premium={prev_premium:.3f} bn EUR)")
                frontier.append(_frontier_row(best_so_far_result, vf_star, prev_premium))
                _write_checkpoint(frontier, checkpoint_path)
                x_current = best_so_far_x.copy()

            # ---- DYNAMIC GRID (auto-grid only): record attempt + extend sweep -
            if _auto_grid:
                history.append({'target': vf_star, 'achieved': vf_min,
                                'premium': premium})
                _nxt = _next_vf_target(history, natural_vf_min,
                                       baseline_cost_bn=baseline_cost / 1e9)
                if _nxt is not None and round(_nxt, 4) not in [round(t, 4)
                                                               for t in active_targets]:
                    active_targets.append(_nxt)

            # Early-stop at the GENUINE ceiling = VF_min stops CLIMBING across
            # targets. A single-target undershoot is NOT a ceiling: the quadratic
            # penalty makes CMA undershoot each target by ~0.02, so the old
            # `vf_min < vf_star - 0.015` rule quit at a FALSE ceiling (S.54).
            # Stop only after VF_min fails to improve for 2 consecutive targets.
            if vf_min <= _prev_achieved + 0.005:
                _ceiling_stall += 1
            else:
                _ceiling_stall = 0
            _prev_achieved = max(_prev_achieved, vf_min)
            if _ceiling_stall >= 2:
                print(f"    ** Ceiling: {oracle.target_metric.upper()}_min stopped "
                      f"climbing (stalled at {_prev_achieved:.3f} for 2 targets) "
                      f"-> stopping sweep.")
                break
        else:
            print(f"    FAILED — VF*={vf_star:.3f} above achievable maximum "
                  f"(LP solves: {evals}, {dt:.1f}s)")
            if _auto_grid:
                # record a stall so the generator counts it toward the ceiling
                # (2-stall stop) rather than hard-breaking on a single miss.
                _lp = (best_so_far_result.get('sys_cost', baseline_cost)
                       - baseline_cost) / 1e9
                history.append({'target': vf_star, 'achieved': best_so_far_vf,
                                'premium': _lp})
                _nxt = _next_vf_target(history, natural_vf_min,
                                       baseline_cost_bn=baseline_cost / 1e9)
                if _nxt is not None and round(_nxt, 4) not in [round(t, 4)
                                                               for t in active_targets]:
                    active_targets.append(_nxt)
            else:
                break

    print(f"\n  TOTAL LP SOLVES: {oracle.solve_count}")

    df = pd.DataFrame(frontier)

    # ---- DEDUPLICATE: drop consecutive rows with identical cr_min ----
    # When the monotonicity fix repeats the best-so-far for multiple
    # targets, the frontier has duplicate rows. Keep only the first
    # occurrence (lowest theta) to avoid zero-width marginal cost bins.
    if len(df) >= 2:
        keep = [True]  # always keep baseline
        for i in range(1, len(df)):
            if abs(df['target_min'].iloc[i] - df['target_min'].iloc[i-1]) < 1e-4:
                keep.append(False)
            else:
                keep.append(True)
        n_dupes = keep.count(False)
        if n_dupes > 0:
            print(f"  Deduplication: removed {n_dupes} repeated frontier points")
        df = df[keep].reset_index(drop=True)

    # ---- MARGINAL COST: finite differences dPremium/d(target metric) ----
    # bn EUR per unit of the ACTIVE target metric (VF or CR).
    if len(df) >= 2:
        mc = [np.nan]  # first point has no predecessor
        for i in range(1, len(df)):
            dcr = df['target_min'].iloc[i] - df['target_min'].iloc[i-1]
            dp  = df['premium_bn'].iloc[i] - df['premium_bn'].iloc[i-1]
            if abs(dcr) > 1e-6:
                mc.append(dp / dcr)  # bn EUR per unit CR
            else:
                mc.append(np.nan)
        df['marginal_cost_bn_per_vf'] = mc
        mlabel = oracle.target_metric.upper()
        print(f"\n  MARGINAL COSTS (dPremium/d{mlabel}):")
        for i in range(1, len(df)):
            if not np.isnan(df['marginal_cost_bn_per_vf'].iloc[i]):
                print(f"    {mlabel} {df['target_min'].iloc[i-1]:.3f} -> {df['target_min'].iloc[i]:.3f}: "
                      f"{df['marginal_cost_bn_per_vf'].iloc[i]/100:+.2f} bn EUR per 0.01 {mlabel} "
                      f"= {df['marginal_cost_bn_per_vf'].iloc[i]:+.2f} bn EUR per 1.0 {mlabel}")
    else:
        df['marginal_cost_bn_per_vf'] = np.nan

    return df, baseline_cost


# ============================================================================
# 2b. MONOTONE (SEQUENTIAL) FRONTIER — capacities can only increase
# ============================================================================

def find_min_cost_monotone(oracle, vf_target, x0, lower_bounds, bounds=None,
                           max_evals=120, optimizer='cma',
                           n_workers=1, oracle_pool=None,
                           parallel_mode='process', D=None,
                           accept_target=None):
    """
    Like find_min_cost_portfolio but with lower bounds on each dimension.
    Capacities cannot decrease below lower_bounds (path dependency).

    Parameters:
      n_workers: number of parallel LP workers (1 = serial).
      oracle_pool: list of LPOracle instances for parallel eval (thread mode).
      parallel_mode: 'process' (default, ProcessPoolExecutor) or 'thread'
                     (ThreadPoolExecutor; legacy, Gurobi-unsafe for long runs).
      D: data dict (required for parallel_mode='process'); each spawned
         worker uses it to build its own LPOracle via _mp_worker_init.
    """
    if bounds is None:
        bounds = oracle.get_bounds()

    PENALTY = 1e12
    # S.57: same slack mechanics as the global frontier -- the penalty
    # drives toward vf_target (possibly inflated), acceptance uses the
    # TRUE target.
    accept = vf_target if accept_target is None else accept_target
    best_result = {'sys_cost': np.inf}
    feasible_seen = False
    eval_counter = {'n': 0}  # dict to avoid nonlocal-in-closure gymnastics

    # --- Waste reducers (see find_min_cost_portfolio for rationale) -------
    # Monotone-specific wrinkle for P2G snap: snap only applies if the
    # current lower_bounds for DE sc is BELOW min_sc. Once the monotone
    # climb has placed P2G above min_sc, lower_bounds[de_sc_idx] >= min_sc
    # already forbids the snap zone, so the pre-filter is redundant.
    de_sc_idx = oracle.flex_dim_map.get((oracle.n_node, 'sc'))
    min_sc = getattr(oracle, 'min_feasible_p2g', 0.0)
    snap_applies = (
        de_sc_idx is not None
        and lower_bounds[de_sc_idx] < min_sc
    )
    EVAL_CACHE_RES = 10.0
    eval_cache = {}
    cache_hits = [0]
    snap_skips = [0]

    def _snap_x(x_list):
        x2 = list(x_list)
        if snap_applies and x2[de_sc_idx] < min_sc:
            x2[de_sc_idx] = min_sc   # floor to H2-feasible min (always-on H2)
            snap_skips[0] += 1
        return x2

    def _cache_key(x_list):
        return tuple(round(v / EVAL_CACHE_RES) for v in x_list)

    def _eval_one(x, ora):
        x_clipped = [np.clip(x[i], lower_bounds[i], bounds[i][1]) for i in range(len(x))]
        x_snapped = _snap_x(x_clipped)
        key = _cache_key(x_snapped)
        cached = eval_cache.get(key)
        if cached is not None:
            cache_hits[0] += 1
            return cached
        r = ora(x_snapped)
        eval_counter['n'] += 1
        if r is None:
            out = (1e15, None, None, 1.0, x_snapped)
        else:
            vf_min = ora.compute_target_min(r)  # active target metric (VF or CR)
            cost = r['sys_cost']
            violation = max(0.0, vf_target - vf_min)
            penalized = cost + PENALTY * violation**2
            out = (penalized, r, vf_min, violation, x_snapped)
        eval_cache[key] = out
        return out

    def _process(out):
        nonlocal best_result, feasible_seen
        penalized, r, vf_min, violation, x_snapped = out
        if r is None:
            return penalized
        if vf_min is not None and vf_min >= accept - 1e-4:
            feasible_seen = True
            if r['sys_cost'] < best_result.get('sys_cost', np.inf):
                best_result = r.copy()
        return penalized

    def objective(x):
        return _process(_eval_one(x, oracle))

    # Initial point: respect lower bounds
    x0_safe = [max(x0[i], lower_bounds[i]) for i in range(len(x0))]
    d = len(x0_safe)

    opt_meta = {
        'optimizer': optimizer, 'budget': max_evals,
        'n_evals': 0, 'budget_exhausted': False,
        'stop_reasons': [], 'feasible': False, 'fallback_used': False,
        'd_nominal': d, 'd_active': d, 'd_frozen': 0,
    }

    if optimizer == 'cma':
        import cma
        # Monotone bounds: lower_bounds (not 0) as the floor.
        lb = [lower_bounds[i] for i in range(d)]
        ub = [bounds[i][1] for i in range(d)]

        # Structural dimensionality reduction. As the monotone frontier climbs
        # toward high VF targets, capacities saturate against their caps and
        # the effective range on those dimensions collapses. Two symptoms:
        #  (1) At VF*=0.80 with DE DSM ~7999.x of 8000 cap, CMA-ES raises
        #      "argument of inverse must be within the given bounds" during
        #      init — the inverted-margin clip places x0 below lb.
        #  (2) A single scalar sigma0 across 1-MW and 200-GW dims wastes the
        #      eval budget; CMA-ES oversteps the saturated dims and understeps
        #      the large ones.
        # Fix: freeze saturated dims at their midpoint and run CMA-ES only on
        # the still-active subspace. This also makes the monotone frontier's
        # marginal-cost curve cleaner — past the saturation point, the dim is
        # no longer a "decision variable" in any meaningful sense.
        active_idx, frozen_idx, frozen_vals = _split_active_frozen_dims(lb, ub)
        d_active = len(active_idx)
        opt_meta['d_active'] = d_active
        opt_meta['d_frozen'] = len(frozen_idx)

        if d_active == 0:
            # All dims saturated — the monotone path has effectively
            # determined the portfolio. Evaluate and return.
            x_full = [frozen_vals.get(i, x0_safe[i]) for i in range(d)]
            objective(x_full)
            opt_meta['stop_reasons'] = ['all_dims_frozen']
        else:
            # Saturation state is reported by _print_opt_status via _opt_meta
            # (d_active/d_nominal) after the optimize call completes.

            def objective_reduced(y):
                x_full = list(x0_safe)  # start with warm-started values
                for k, i in enumerate(active_idx):
                    x_full[i] = y[k]
                for i, v in frozen_vals.items():
                    x_full[i] = v
                return objective(x_full)

            lb_a = [lb[i] for i in active_idx]
            ub_a = [ub[i] for i in active_idx]
            ranges_a = [ub_a[k] - lb_a[k] for k in range(d_active)]
            sigma0 = 0.25 * np.mean(ranges_a) if np.mean(ranges_a) > 0 else 1000.0
            popsize = 2 * (4 + int(3 * np.log(max(d_active, 1))))

            opts = cma.CMAOptions()
            opts.set('bounds', [lb_a, ub_a])
            opts.set('maxfevals', max_evals)
            opts.set('popsize', popsize)
            opts.set('tolx', 100)
            opts.set('tolfun', 1e5)
            opts.set('verbose', -9)
            opts.set('seed', 42)

            # _interior_x0 is the minimal safeguard retained from the
            # original np.clip(x, lb+1, ub-1) pattern — it keeps x0 strictly
            # inside bounds when the interior is narrower than 2*margin.
            y0 = [_interior_x0(x0_safe[i], lb[i], ub[i]) for i in active_idx]

            es = cma.CMAEvolutionStrategy(y0, sigma0, opts)

            # Parallel path mirrors find_min_cost_portfolio.
            # THREAD mode: oracle_pool required.
            # PROCESS mode: D (data dict) required — workers build their
            #   own LPOracle via _mp_worker_init. Default for long runs;
            #   thread mode is retained for short tests but unreliable for
            #   long runs due to shared Gurobi C-state corruption.
            use_parallel = n_workers > 1
            if parallel_mode == 'thread':
                use_parallel = use_parallel and (
                    oracle_pool is not None
                    and len(oracle_pool) >= n_workers
                )
            elif parallel_mode == 'process':
                use_parallel = use_parallel and (D is not None)
            else:
                use_parallel = False

            if use_parallel and parallel_mode == 'thread':
                from concurrent.futures import ThreadPoolExecutor
                from queue import Queue as _Queue

                pool_q = _Queue()
                for o in oracle_pool[:n_workers]:
                    pool_q.put(o)

                def _parallel_eval(y):
                    import traceback as _tb
                    x_full = list(x0_safe)
                    for k, i in enumerate(active_idx):
                        x_full[i] = y[k]
                    for i, v in frozen_vals.items():
                        x_full[i] = v
                    ora = pool_q.get()
                    try:
                        return _eval_one(x_full, ora)
                    except BaseException as _e:
                        print(f"  [mono parallel worker] EXCEPTION: "
                              f"{type(_e).__name__}: {_e}", flush=True)
                        _tb.print_exc()
                        raise
                    finally:
                        pool_q.put(ora)

                executor = ThreadPoolExecutor(max_workers=n_workers)
                try:
                    # See find_min_cost_portfolio for parallel-loop rationale.
                    _gen_n = 0
                    while not es.stop():
                        solutions = es.ask()
                        _gen_n += 1
                        print(f"  [mono gen {_gen_n}] dispatching "
                              f"{len(solutions)} evals to {n_workers} "
                              f"workers (countevals={es.countevals})",
                              flush=True)
                        try:
                            outs = list(executor.map(_parallel_eval, solutions))
                        except BaseException as _e:
                            import traceback as _tb
                            print(f"  [mono main] executor.map raised: "
                                  f"{type(_e).__name__}: {_e}", flush=True)
                            _tb.print_exc()
                            raise
                        fitnesses = [_process(out) for out in outs]
                        es.tell(solutions, fitnesses)
                        if es.countevals >= max_evals:
                            break
                finally:
                    executor.shutdown(wait=True)
            elif use_parallel and parallel_mode == 'process':
                # PROCESS-POOL path: each worker is a separate Python process
                # with its own LPOracle (built once in _mp_worker_init).
                # Cache lookup + snap + clipping stay on the main process,
                # so there's no shared mutable state between workers.
                from concurrent.futures import ProcessPoolExecutor, as_completed

                print(f"  [mono process mode] Spawning {n_workers} worker "
                      f"processes (each builds its own LPOracle, "
                      f"~30-60 s × {n_workers} one-time)...", flush=True)
                executor = ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_mp_worker_init,
                    initargs=(D,),
                )
                try:
                    _gen_n = 0
                    while not es.stop():
                        solutions = es.ask()
                        _gen_n += 1

                        # Reconstruct full-dim candidates, clip to
                        # [lower_bounds, ub], snap the P2G snap-zone.
                        # All pure Python on the main process.
                        xs_snapped = []
                        for y in solutions:
                            x_full = list(x0_safe)
                            for k, i in enumerate(active_idx):
                                x_full[i] = y[k]
                            for i, v in frozen_vals.items():
                                x_full[i] = v
                            x_clipped = [
                                np.clip(x_full[i], lower_bounds[i], bounds[i][1])
                                for i in range(d)
                            ]
                            xs_snapped.append(_snap_x(x_clipped))

                        # Check cache. Hits bypass the pool; only misses
                        # are dispatched to worker processes.
                        outs = [None] * len(solutions)
                        submit_idx = []
                        for i, x in enumerate(xs_snapped):
                            hit = eval_cache.get(_cache_key(x))
                            if hit is not None:
                                cache_hits[0] += 1
                                outs[i] = hit
                            else:
                                submit_idx.append(i)

                        print(f"  [mono gen {_gen_n}] dispatching "
                              f"{len(submit_idx)}/{len(solutions)} evals to "
                              f"{n_workers} worker processes "
                              f"(countevals={es.countevals}, "
                              f"cache_hits={cache_hits[0]})", flush=True)

                        if submit_idx:
                            futs = {
                                executor.submit(
                                    _mp_worker_solve, list(xs_snapped[i])
                                ): i
                                for i in submit_idx
                            }
                            for fut in as_completed(futs):
                                i = futs[fut]
                                try:
                                    r = fut.result()
                                except BaseException as _e:
                                    import traceback as _tb
                                    print(f"  [mono main] worker future raised "
                                          f"for index {i}: "
                                          f"{type(_e).__name__}: {_e}",
                                          flush=True)
                                    _tb.print_exc()
                                    r = None

                                # Build the standard eval tuple on the main
                                # process (matches _eval_one's contract).
                                x = xs_snapped[i]
                                if r is None:
                                    out = (1e15, None, None, 1.0, x)
                                else:
                                    vf_min_r = oracle.compute_target_min(r)  # active target metric
                                    cost = r['sys_cost']
                                    violation = max(0.0, vf_target - vf_min_r)
                                    penalized = cost + PENALTY * violation**2
                                    out = (penalized, r, vf_min_r, violation, x)
                                    # Attribute worker solves to main oracle
                                    # so the monotone-frontier "TOTAL LP
                                    # SOLVES" report is accurate.
                                    oracle.solve_count += 1
                                eval_cache[_cache_key(x)] = out
                                outs[i] = out

                        # _process is race-free here — main process only.
                        fitnesses = [_process(out) for out in outs]
                        es.tell(solutions, fitnesses)
                        if es.countevals >= max_evals:
                            break
                finally:
                    executor.shutdown(wait=True)
            else:
                es.optimize(objective_reduced)

            stop = dict(es.stop())
            opt_meta['n_evals'] = int(getattr(es, 'countevals', eval_counter['n']))
            opt_meta['stop_reasons'] = list(stop.keys())
            opt_meta['budget_exhausted'] = 'maxfevals' in stop
            opt_meta['cache_hits'] = cache_hits[0]
            opt_meta['snap_skips'] = snap_skips[0]
            opt_meta['n_workers'] = n_workers if use_parallel else 1
    else:
        simplex = np.zeros((d + 1, d))
        simplex[0] = x0_safe
        for i in range(d):
            vertex = np.array(x0_safe, dtype=float)
            step = 0.25 * (bounds[i][1] - lower_bounds[i])
            vertex[i] = min(x0_safe[i] + step, bounds[i][1])
            simplex[i + 1] = vertex

        nm_result = minimize(
            objective, x0_safe, method='Nelder-Mead',
            options={'maxfev': max_evals, 'xatol': 500, 'fatol': 1e6,
                     'adaptive': True, 'initial_simplex': simplex})
        opt_meta['n_evals'] = int(getattr(nm_result, 'nfev', eval_counter['n']))
        nm_msg = str(getattr(nm_result, 'message', ''))
        opt_meta['budget_exhausted'] = (
            not getattr(nm_result, 'success', True)
            and 'function evaluations' in nm_msg.lower()
        )
        opt_meta['stop_reasons'] = [nm_msg] if nm_msg else []

    opt_meta['feasible'] = feasible_seen and best_result.get('sys_cost', np.inf) < np.inf

    # NOTE: find_min_cost_monotone intentionally has NO closest-match fallback.
    # Using an infeasible (VF < vf_target) point as the new lower_bounds would
    # permanently corrupt the monotone climb — every subsequent target would
    # inherit a capacity stack that never actually achieved any VF target.
    # If feasible=False the caller should keep the previous lower_bounds and
    # skip this target. See trace_monotone_frontier for the handling.

    if isinstance(best_result, dict):
        best_result['_opt_meta'] = opt_meta

    return best_result


def trace_monotone_frontier(D, vf_targets=None, optimizer='cma',
                            n_workers=1, oracle_pool=None,
                            parallel_mode='process',
                            max_evals_override=None, target_metric='vf',
                            target_slack=0.0, honesty_evals=0,
                            checkpoint_path=None, resume=False):
    """
    Traces the least-cost VF frontier with MONOTONICITY constraint:
    at each VF target, capacities can only INCREASE relative to the
    previous point. This quantifies the cost of path dependency —
    how much more expensive it is to build the frontier incrementally
    vs. globally re-optimizing at each point.

    Returns (df_monotone, baseline_cost).

    Parameters:
      n_workers: parallel LP workers (default 1).
      oracle_pool: list[LPOracle] for parallel eval (thread mode). If None
                   and n_workers > 1 and parallel_mode='thread', a pool is
                   built here.
      parallel_mode: 'process' (default) or 'thread'. Process mode spawns
                   separate worker processes (safe for long runs); thread
                   mode shares a single Gurobi environment (not safe for
                   hundreds+ of concurrent optimize() calls).
      max_evals_override: if set, uses this budget for every target.
    """
    _auto_grid_m = (vf_targets is None and target_metric in ('vf', 'vfw'))
    if vf_targets is None and target_metric == 'cr':
        vf_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]

    oracle = LPOracle(D)
    oracle.target_metric = target_metric

    # Oracle pool is only meaningful in thread mode. Process mode builds
    # its own oracles inside each spawned worker via _mp_worker_init.
    if n_workers > 1 and parallel_mode == 'thread' and oracle_pool is None:
        print(f"\n  [thread mode] Building oracle pool (monotone): "
              f"{n_workers-1} extra LPOracle instance(s)...")
        extras = [LPOracle(D) for _ in range(n_workers - 1)]
        oracle_pool = [oracle] + extras

    # ---- BASELINE B_today (progress.md Section 36) ----
    # The transition (monotone) frontier's reference is today's installed
    # flexibility fleet (k_today_flex). Capacity can only be ADDED from
    # here; premium is measured against cost_today. Lower bounds therefore
    # start at today's fleet rather than at zero.
    x_today = oracle.make_today_vector()
    baseline = oracle(x_today)
    if baseline is None:
        print("  ERROR: Could not compute today's-fleet baseline B_today!")
        return pd.DataFrame(), 0.0

    # natural_vf_min holds CR_min (Section 34); name kept for minimal diff.
    natural_vf_min = oracle.compute_target_min(baseline)
    natural_vf_display = oracle.compute_vf_min(baseline)
    baseline_cost = baseline['sys_cost']
    baseline_subsidy = baseline.get('subsidy_bill_eur', 0.0)

    if _auto_grid_m:
        _b = natural_vf_min
        _ladder = [round(_b + 0.04, 3), 0.50, 0.60, 0.70, 0.75, 0.80,
                   0.85, 0.90, 0.95, 1.00, 1.05]
        vf_targets = []
        for _t in sorted(set(_ladder)):
            if _t > _b + 0.005 and (not vf_targets or _t - vf_targets[-1] >= 0.03):
                vf_targets.append(_t)
        print(f"  Adaptive MONOTONE grid (natural {target_metric}_min="
              f"{_b:.3f}): {vf_targets}")

    print(f"\n  MONOTONE (TRANSITION) FRONTIER — BASELINE B_today (today's fleet):")
    print(f"    System CR_min: {natural_vf_min:.4f} (VF_min: {natural_vf_display:.4f}), "
          f"Cost: {baseline_cost/1e9:.3f} bn EUR")

    active_targets = [t for t in vf_targets if t > natural_vf_min + 0.005]

    # Build baseline row
    caps = oracle.format_capacities(baseline)
    frontier_row_0 = {
        'vf_target': round(natural_vf_min, 3),
        'cr_min': natural_vf_min,
        'vf_min': natural_vf_display,
        'vf_wind_on': baseline.get('vf_wind_on', 0), 'vf_wind_off': baseline.get('vf_wind_off', 0), 'vf_pv': baseline.get('vf_pv', 0),
        'cr_wind_on': baseline.get('cr_wind_on', 0), 'cr_wind_off': baseline.get('cr_wind_off', 0), 'cr_pv': baseline.get('cr_pv', 0),
        **caps,
        'sys_cost': baseline_cost, 'premium_bn': 0.0,
    }
    for n in oracle.all_nodes:
        for v in D['tech_vre']:
            for pref in ('vf_', 'cr_', 'mm_'):
                key = f'{pref}{v}_{n}'
                if key in baseline:
                    frontier_row_0[key] = baseline[key]
    frontier = [frontier_row_0]

    # Lower bounds start at today's installed fleet (the transition floor).
    lower_bounds = list(x_today)
    x_current = np.array(x_today, dtype=float)

    # ---- RESUME from checkpoint (S.57; mirrors the global frontier) ----
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            _df_ck = pd.read_csv(checkpoint_path)
        except Exception as _e:
            print(f"  RESUME(monotone): cannot read checkpoint ({_e}) -- fresh start.")
            _df_ck = None
        if _df_ck is not None and len(_df_ck) >= 1:
            _ck_base = float(_df_ck.iloc[0]['sys_cost'])
            _drift = abs(_ck_base - baseline_cost) / max(abs(baseline_cost), 1.0)
            if _drift > 0.005:
                raise RuntimeError(
                    f"RESUME(monotone) ABORTED: checkpoint baseline "
                    f"{_ck_base/1e9:.3f} bn vs fresh {baseline_cost/1e9:.3f} bn "
                    f"({100*_drift:.2f}% drift) -- model changed; delete "
                    f"{checkpoint_path} to start over.")
            if len(_df_ck) > 1:
                frontier = _df_ck.to_dict('records')
                _last = frontier[-1]
                _last_target = max(float(rr.get('vf_target', 0.0))
                                   for rr in frontier[1:])
                x_current = np.array(
                    [float(_last.get(k, 0.0) or 0.0) for k in oracle.x_keys])
                lower_bounds = [max(lower_bounds[i], float(x_current[i]))
                                for i in range(oracle.n_dims)]
                _n0 = len(active_targets)
                active_targets = [t for t in active_targets
                                  if t > _last_target + 1e-9]
                print(f"  RESUME(monotone): {len(frontier)-1} target(s) loaded; "
                      f"skipping {_n0 - len(active_targets)}; "
                      f"{len(active_targets)} remain.")
    _write_checkpoint(frontier, checkpoint_path)

    for vf_star in active_targets:
        print(f"\n  --- MONOTONE Target theta = {vf_star:.2f} (cost-recovery CR) ---")
        lb_str = ", ".join(f"{oracle.x_keys[i]}>={lower_bounds[i]:,.0f}" for i in range(oracle.n_dims))
        print(f"      Lower bounds: {lb_str}")
        t0 = time.time()

        # Monotone budget: raised from 200 -> 600 after the 2026-04-22 run
        # showed the v12 monotone frontier regularly failing the VF target
        # near the ceiling (0.87+) due to budget exhaustion. 600 evals gives
        # CMA-ES ~25 generations at popsize 24, usually enough for a warm-
        # started climb where active_dims is often << 17.
        if max_evals_override is not None:
            max_ev = max_evals_override
        else:
            max_ev = 600 if optimizer == 'cma' else 120
        best = find_min_cost_monotone(
            oracle, vf_star + max(0.0, target_slack), x0=x_current,
            lower_bounds=lower_bounds, max_evals=max_ev,
            optimizer=optimizer,
            n_workers=n_workers, oracle_pool=oracle_pool,
            parallel_mode=parallel_mode, D=D,
            accept_target=vf_star,
        )

        # First-order honesty check, lower-bound aware (S.57)
        if (honesty_evals > 0 and best
                and best.get('sys_cost', np.inf) < np.inf
                and oracle.compute_target_min(best) >= vf_star - 1e-3):
            _om = best.get('_opt_meta')
            _b2, _hm = _honesty_check(oracle, best, vf_star,
                                      max_probes=honesty_evals,
                                      lower_bounds=lower_bounds)
            if _b2 is not best and _om is not None:
                _b2['_opt_meta'] = _om
            best = _b2
            best['_honesty_meta'] = _hm

        dt = time.time() - t0

        if best and best.get('sys_cost', np.inf) < np.inf:
            premium = (best['sys_cost'] - baseline_cost) / 1e9
            vf_min = oracle.compute_target_min(best)  # active target metric
            caps = oracle.format_capacities(best)
            cap_str = ", ".join(f"{k}={v:,.0f}" for k, v in caps.items())
            print(f"    Found: {cap_str}")
            print(f"    CR_min={vf_min:.4f}, Premium={premium:.3f} bn EUR vs B_today ({dt:.1f}s)")
            _print_opt_status(best.get('_opt_meta', {}))

            row = {
                'vf_target': vf_star,
                'target_metric': oracle.target_metric,
                'target_min': vf_min,   # loop var holds the ACTIVE target metric
                'cr_min': oracle.compute_cr_min(best),
                'vf_min': oracle.compute_vf_min(best),
                'vf_wind_on': best.get('vf_wind_on', 0), 'vf_wind_off': best.get('vf_wind_off', 0), 'vf_pv': best.get('vf_pv', 0),
                'cr_wind_on': best.get('cr_wind_on', 0), 'cr_wind_off': best.get('cr_wind_off', 0), 'cr_pv': best.get('cr_pv', 0),
                **caps,
                'sys_cost': best['sys_cost'], 'premium_bn': premium,
            }
            for nd in oracle.all_nodes:
                for v in D['tech_vre']:
                    for pref in ('vf_', 'cr_', 'mm_'):
                        key = f'{pref}{v}_{nd}'
                        if key in best:
                            row[key] = best[key]
            # Expose optimizer status on the frontier row for post-hoc
            # analysis (e.g. filtering budget-exhausted targets in the CSV).
            meta = best.get('_opt_meta', {})
            row['opt_budget_exhausted'] = bool(meta.get('budget_exhausted', False))
            row['opt_n_evals'] = int(meta.get('n_evals', 0))
            row['opt_budget'] = int(meta.get('budget', 0))
            row['opt_stop_reason'] = ",".join(str(s) for s in meta.get('stop_reasons', []))
            # S.57: aggregates + welfare layer (vs B_today) + diagnostics
            row['vf_genw'] = best.get('vf_genw')
            row['cannib_loss_bn'] = best.get('cannib_loss_eur', 0.0) / 1e9
            row['subsidy_bill_bn'] = best.get('subsidy_bill_eur', 0.0) / 1e9
            row['avoided_subsidy_bn'] = (baseline_subsidy
                                         - best.get('subsidy_bill_eur', 0.0)) / 1e9
            row['net_benefit_mcf0_bn'] = row['avoided_subsidy_bn'] - premium
            row['net_benefit_mcf30_bn'] = 1.3 * row['avoided_subsidy_bn'] - premium
            for _kk, _vv in best.items():
                if isinstance(_kk, str) and _kk.startswith('resid_'):
                    row[_kk] = _vv
            _hm = best.get('_honesty_meta')
            if _hm:
                row['honesty_n_descent'] = _hm.get('n_descent')
                row['honesty_n_probes'] = _hm.get('n_probes')
                row['honesty_improved_bn'] = _hm.get('improved_bn')
            frontier.append(row)
            _write_checkpoint(frontier, checkpoint_path)

            # CRITICAL: update lower bounds — capacities can only increase.
            # We only reach this branch when the VF target was ACHIEVED
            # (best_result is populated from a feasible point). The monotone
            # floor therefore advances to a VF-feasible capacity stack, as
            # required for the path-dependency interpretation.
            new_x = np.array([best[k] for k in oracle.x_keys], dtype=float)
            lower_bounds = [max(lower_bounds[i], new_x[i]) for i in range(oracle.n_dims)]
            x_current = new_x
        else:
            # VF target not achieved within budget. We deliberately do NOT
            # advance lower_bounds here — feeding an infeasible capacity
            # stack as the floor for later targets would corrupt the entire
            # monotone climb. The target is simply skipped; subsequent
            # targets warm-start from the previous feasible point.
            print(f"    FAILED — VF*={vf_star} not achievable with monotone "
                  f"constraints ({dt:.1f}s)")
            meta = best.get('_opt_meta', {}) if isinstance(best, dict) else {}
            _print_opt_status(meta)
            if meta.get('budget_exhausted'):
                print(f"    HINT: budget was exhausted — try a higher "
                      f"max_evals for this target (current={meta.get('budget', '?')}).")

    print(f"\n  MONOTONE FRONTIER — TOTAL LP SOLVES: {oracle.solve_count}")

    df = pd.DataFrame(frontier)

    # ---- MARGINAL COST (dPremium/dCR, Section 34) ----
    if len(df) >= 2:
        mc = [np.nan]
        for i in range(1, len(df)):
            dcr = df['cr_min'].iloc[i] - df['cr_min'].iloc[i-1]
            dp  = df['premium_bn'].iloc[i] - df['premium_bn'].iloc[i-1]
            mc.append(dp / dcr if abs(dcr) > 1e-6 else np.nan)
        df['marginal_cost_bn_per_vf'] = mc
    else:
        df['marginal_cost_bn_per_vf'] = np.nan

    return df, baseline_cost


# ============================================================================
# 3. PLOTTING
# ============================================================================

def plot_frontier(df, baseline_cost, outdir, df_monotone=None, tag=''):
    os.makedirs(outdir, exist_ok=True)
    sfx = f'_{tag}' if tag else ''
    df.to_csv(os.path.join(outdir, f'directed_search_frontier{sfx}.csv'), index=False)
    if df_monotone is not None and not df_monotone.empty:
        df_monotone.to_csv(os.path.join(outdir, f'monotone_frontier{sfx}.csv'), index=False)

    n_panels = 4 if (df_monotone is not None and not df_monotone.empty) else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4.5 * n_panels))
    _m = (df['target_metric'].iloc[0] if 'target_metric' in df.columns and len(df) else 'cr')
    _mp = 'vf' if _m == 'vf' else 'cr'
    _ML = _mp.upper()
    fig.suptitle(f'Least-cost Investability Frontier (v12: 5-node, endogenous NTC; {_ML} target)', fontsize=14)

    # ---- Panel 1: Frontier (premium vs the ACTIVE target metric) ----
    axes[0].plot(df[f'{_mp}_min'], df['premium_bn'], 'ko-', linewidth=2, markersize=8,
                 label='Global optimum (premium vs B_opt)')
    if df_monotone is not None and not df_monotone.empty:
        axes[0].plot(df_monotone[f'{_mp}_min'], df_monotone['premium_bn'], 's--',
                     color='crimson', linewidth=2, markersize=7,
                     label='Transition / monotone (premium vs B_today)')
    axes[0].plot(df[f'{_mp}_wind_on'], df['premium_bn'], 'b^--', alpha=0.4, markersize=5, label=f'{_ML}_wind_on')
    if f'{_mp}_wind_off' in df.columns:
        axes[0].plot(df[f'{_mp}_wind_off'], df['premium_bn'], 'gs--', alpha=0.4, markersize=5, label=f'{_ML}_wind_off')
    axes[0].plot(df[f'{_mp}_pv'], df['premium_bn'], 'rv--', alpha=0.4, markersize=5, label=f'{_ML}_pv')
    axes[0].axvline(x=1.0, color='gray', linestyle=':', alpha=0.7,
                    label=('CR=1 (full cost recovery)' if _mp == 'cr' else 'VF=1 (earns market avg)'))
    axes[0].set_ylabel('System Premium [Billion EUR/yr]')
    axes[0].set_xlabel('Cost-recovery ratio CR (= MV / annualized cost)' if _mp == 'cr'
                       else 'Value factor VF (= capture price / base price)')
    axes[0].set_title(f'Cost of VRE de-cannibalisation (premium vs {_ML} target)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # ---- Panel 2: Marginal cost (dPremium/dVF) ----
    if 'marginal_cost_bn_per_vf' in df.columns:
        mc_valid = df[df['marginal_cost_bn_per_vf'].notna() & (df.index > 0)]
        if len(mc_valid) > 0:
            # Plot as step-like bar chart between CR points
            vf_mid = [(df['cr_min'].iloc[i-1] + df['cr_min'].iloc[i])/2
                      for i in range(1, len(df)) if not np.isnan(df['marginal_cost_bn_per_vf'].iloc[i])]
            mc_vals = [df['marginal_cost_bn_per_vf'].iloc[i]
                       for i in range(1, len(df)) if not np.isnan(df['marginal_cost_bn_per_vf'].iloc[i])]
            colors_mc = ['green' if v < 0 else 'firebrick' for v in mc_vals]
            axes[1].bar(vf_mid, mc_vals, width=0.015, color=colors_mc, alpha=0.8,
                        edgecolor='black', linewidth=0.5)
            axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('Marginal Cost [Billion EUR per unit CR]')
            axes[1].set_xlabel('Cost-recovery ratio CR')
            axes[1].set_title('Marginal Cost of Investability (dPremium/dCR)')
            axes[1].grid(True, alpha=0.3)

            if df_monotone is not None and 'marginal_cost_bn_per_vf' in df_monotone.columns:
                mc_mono = df_monotone[df_monotone['marginal_cost_bn_per_vf'].notna() & (df_monotone.index > 0)]
                if len(mc_mono) > 0:
                    vf_mid_m = [(df_monotone['cr_min'].iloc[i-1] + df_monotone['cr_min'].iloc[i])/2
                                for i in range(1, len(df_monotone))
                                if not np.isnan(df_monotone['marginal_cost_bn_per_vf'].iloc[i])]
                    mc_vals_m = [df_monotone['marginal_cost_bn_per_vf'].iloc[i]
                                 for i in range(1, len(df_monotone))
                                 if not np.isnan(df_monotone['marginal_cost_bn_per_vf'].iloc[i])]
                    axes[1].plot(vf_mid_m, mc_vals_m, 's-', color='crimson', markersize=5,
                                 label='Monotone (path-dep.)', alpha=0.8)
                    axes[1].legend(fontsize=8)

    # ---- Panel 3: Portfolio composition (DE + DK + NTC) ----
    n = len(df)
    x_pos = np.arange(n)

    # Collect all capacity columns dynamically
    # DE instruments
    de_cols = [('sto_mw', 'DE Storage', 'orange'),
               ('sc_mw', 'DE P2G', 'green'),
               ('dsm_mw', 'DE DSM', 'purple'),
               ('ev_mw', 'DE EV', 'dodgerblue')]
    # DK instruments (if present as outer-search)
    dk_cols = [('sto_dk_mw', 'DK Storage', 'sandybrown'),
               ('sc_dk_mw', 'DK P2G', 'limegreen'),
               ('dsm_dk_mw', 'DK DSM', 'mediumpurple'),
               ('ev_dk_mw', 'DK EV', 'lightskyblue')]
    # NTC investment for all links (v12: 5 links)
    ntc_cols = [('ntc_de_cz_mw', 'NTC DE-CZ', 'gray'),
                ('ntc_de_dk_mw', 'NTC DE-DK', 'dimgray'),
                ('ntc_de_at_mw', 'NTC DE-AT', 'silver'),
                ('ntc_de_nl_mw', 'NTC DE-NL', 'lightslategray'),
                ('ntc_cz_at_mw', 'NTC CZ-AT', 'darkgray')]

    all_bar_cols = [(c, l, clr) for c, l, clr in de_cols + dk_cols + ntc_cols if c in df.columns]
    n_bars = len(all_bar_cols)
    width = min(0.80 / max(n_bars, 1), 0.15)

    for j, (col, label, color) in enumerate(all_bar_cols):
        offset = (j - (n_bars - 1) / 2) * width
        vals = df[col].fillna(0) / 1000
        axes[2].bar(x_pos + offset, vals, width, label=f'{label} [GW]', color=color)

    axes[2].set_ylabel('Installed Capacity [GW]')
    axes[2].set_xlabel('theta (cost-recovery CR) target')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f"{v:.2f}" for v in df['vf_target']], rotation=45)
    axes[2].set_title('Optimal Mitigation Portfolio at Each VF Level')
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.3, axis='y')

    # ---- Panel 4: Monotone portfolio composition (realistic — no decommissioning) ----
    if n_panels == 4:
        dfm = df_monotone
        n_m = len(dfm)
        x_pos_m = np.arange(n_m)

        all_bar_cols_m = [(c, l, clr) for c, l, clr in de_cols + dk_cols + ntc_cols if c in dfm.columns]
        n_bars_m = len(all_bar_cols_m)
        width_m = min(0.80 / max(n_bars_m, 1), 0.15)

        for j, (col, label, color) in enumerate(all_bar_cols_m):
            offset = (j - (n_bars_m - 1) / 2) * width_m
            vals = dfm[col].fillna(0) / 1000
            axes[3].bar(x_pos_m + offset, vals, width_m, label=f'{label} [GW]', color=color)

        axes[3].set_ylabel('Installed Capacity [GW]')
        axes[3].set_xlabel('theta (cost-recovery CR) target')
        axes[3].set_xticks(x_pos_m)
        axes[3].set_xticklabels([f"{v:.2f}" for v in dfm['vf_target']], rotation=45)
        axes[3].set_title('Monotone Portfolio (realistic — no decommissioning)')
        axes[3].legend(fontsize=7, ncol=2)
        axes[3].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = os.path.join(outdir, f'supply_curve_directed{sfx}.png')
    plt.savefig(fname, dpi=150)
    print(f"\n  Saved: {fname}")
    plt.close('all')


# ============================================================================
# 4. POST-RUN SCALAR VALIDATION
# ============================================================================

def validate_frontier_quality(D, df_frontier, n_perturb=5, target_metric='vf'):
    """
    Post-run validation: perturb each dimension at the highest-VF frontier
    point and check if any perturbation improves cost or VF_min.

    Returns a dict with validation results.
    """
    if df_frontier.empty:
        print("  [validation] No frontier data to validate.")
        return {}

    oracle = LPOracle(D)
    oracle.target_metric = target_metric

    # Use the highest-VF frontier point as reference
    ref_row = df_frontier.iloc[-1]
    ref_x = []
    for k in oracle.x_keys:
        ref_x.append(ref_row.get(k, 0.0))
    ref_result = oracle(ref_x)
    if ref_result is None:
        print("  [validation] Reference point infeasible — skipping.")
        return {}

    ref_cost = ref_result['sys_cost']
    ref_vf = oracle.compute_target_min(ref_result)  # validate on active target
    print(f"\n  [validation] Reference point: CR_min={ref_vf:.4f}, cost={ref_cost/1e9:.3f} bn EUR")

    # Perturb each dimension independently
    best_improvement = 0.0
    for dim_idx in range(oracle.n_dims):
        dim_name = oracle.x_keys[dim_idx]
        dim_max = oracle.bounds[dim_idx]
        if dim_max < 1.0:
            continue

        test_vals = np.linspace(0, dim_max, n_perturb + 2)[1:-1]
        for tv in test_vals:
            x_test = list(ref_x)
            x_test[dim_idx] = tv
            r = oracle(x_test)
            if r is None:
                continue
            alt_vf = oracle.compute_target_min(r)  # validate on active target
            if alt_vf >= ref_vf - 0.005 and r['sys_cost'] < ref_cost:
                improvement = (ref_cost - r['sys_cost']) / ref_cost * 100
                if improvement > best_improvement:
                    best_improvement = improvement
                    print(f"    {dim_name}={tv:,.0f}: cost -{improvement:.2f}%, CR_min={alt_vf:.4f}")

    status = 'PASS' if best_improvement < 1.0 else 'WARN'
    print(f"\n  [validation] Max cost improvement found: {best_improvement:.2f}% -> {status}")
    print(f"  [validation] Total validation LP solves: {oracle.solve_count}")
    return {'status': status, 'max_improvement_pct': best_improvement}


# ============================================================================
# 4b. SENSITIVITY: multi-restart wrapper + consolidated results writer
# ============================================================================
# These helpers exist purely to make Tier 1 / Tier 2 sensitivity runs
# self-contained. The multi-restart wrapper runs the diagnostic-mode VF
# target with N seeds (42, 43, ...) and picks the lowest *penalised*
# fitness — matching what CMA-ES itself was trying to minimise. Section
# 33 of progress.md justifies the pick metric: the basin landscape is
# multi-modal under quadratic penalty, so different seeds explore
# different basins, and the best penalised fitness is the principled
# "best of N starts".


def _build_run_tag(args):
    """Compose a short, filesystem-safe label encoding the active
    sensitivity flags. Used as the default --run-tag suffix."""
    parts = []
    parts.append(f"h2dk{args.dk_h2_twh:g}")
    if abs(args.dk_nordic_mult - 1.0) > 1e-6:
        parts.append(f"nor{args.dk_nordic_mult:g}")
    if abs(args.ntc_cost_mult - 1.0) > 1e-6:
        parts.append(f"ntc{args.ntc_cost_mult:g}")
    if args.enable_heat:
        parts.append("heat")
    if args.n_restarts > 1:
        parts.append(f"r{args.n_restarts}")
    parts.append(f"s{args.cma_seed}")
    return "_".join(parts)


def _run_id(args, nodes):
    """Filesystem-safe identifier UNIQUE to a run's distinguishing settings
    (S.70.5), so different years/weather/sensitivities never collide on the
    checkpoint or output filenames. --run-tag overrides the auto suffix."""
    base = f"{len(nodes)}node_{args.target}"
    if getattr(args, 'run_tag', None):
        return f"{base}_{args.run_tag}"
    sc = args.scenario.replace('tyndp_', '').replace('_', '')
    cy = f"cy{args.climate_year}" if args.climate_year is not None else "cydieter"
    flags = ''.join(c for c, on in [('d', args.demand_2040), ('h', args.h2_2040),
                                     ('e', args.ev_2040),
                                     ('p', args.p2g_endogenous)] if on) or 'base'
    parts = [base, sc, cy, flags, f"w{args.n_weeks}"]
    if getattr(args, 'resid_export_price', None) is not None:
        parts.append(f"rxp{args.resid_export_price:g}")
    if getattr(args, 'no_resid', False):
        parts.append("noresid")
    if getattr(args, 'ev_cap', None) is not None:
        parts.append(f"evcap{int(args.ev_cap/1000)}")
    # S.103: TYNDP_FIXES runs get their own files (never resume old checkpoints)
    _tfx = os.environ.get('TYNDP_FIXES', '').strip().lower()
    if _tfx:
        parts.append('tf' + ('all' if _tfx == 'all' else _tfx.replace(',', '-')))
    return "_".join(parts)


def _diagnostic_multi_restart(D, vf_target, max_evals, n_workers,
                              parallel_mode, base_seed=42, n_restarts=1):
    """Run the diagnostic VF target across n_restarts seeds. For each
    seed, captures the resulting frontier row, achieved VF_min, premium,
    penalised fitness (= sys_cost + 1e12*max(0, vf_target-vf_min)^2),
    plus the optimiser metadata. Returns (best_df, all_restart_records,
    baseline_cost) where best_df is the single-seed DataFrame from the
    seed with the LOWEST penalised fitness — same metric CMA-ES
    minimises, so the choice is defensible.
    """
    PENALTY = 1e12
    all_records = []
    best_df = None
    best_baseline_cost = None
    best_pen = float('inf')

    for r in range(n_restarts):
        seed = int(base_seed) + r
        if n_restarts > 1:
            print(f"\n  ===== Restart {r+1}/{n_restarts} (seed={seed}) =====")
        df_r, baseline_cost = trace_pareto_frontier(
            D, vf_targets=[vf_target], optimizer='cma',
            n_workers=n_workers, parallel_mode=parallel_mode,
            max_evals_override=max_evals, cma_seed=seed,
        )
        if df_r.empty or len(df_r) < 2:
            print(f"  [restart {r+1}] No frontier point (baseline only) — skipping")
            all_records.append({
                'restart': r+1, 'seed': seed, 'vf_min': None,
                'premium_bn': None, 'penalized': None, 'evals': 0,
                'feasible': False, 'fallback_used': False, 'note': 'empty',
            })
            continue
        # Use the LAST row (the one matching vf_target, not the baseline).
        last = df_r.iloc[-1]
        # restart selection on the metric the target is set in (VF_min); this
        # read cr_min before (legacy of the CR target), so the restart with the
        # lowest cost was chosen whenever every cr_min exceeded the VF target
        vf_min  = float(last['vf_min']) if 'vf_min' in last.index else float(last['cr_min'])
        premium = float(last['premium_bn'])
        sys_cost = float(last['sys_cost'])
        violation = max(0.0, float(vf_target) - vf_min)
        penalized = sys_cost + PENALTY * violation**2
        rec = {
            'restart': r+1, 'seed': seed, 'vf_min': vf_min,
            'premium_bn': premium, 'sys_cost': sys_cost,
            'penalized': penalized, 'violation': violation,
        }
        # Capture the full per-restart portfolio so basin-divergence
        # (storage-heavy vs P2G-heavy DK, etc.) can be inspected on the
        # 'restart_portfolios' sheet without re-running anything.
        cap_keys = [c for c in last.index
                    if any(c.endswith(suf) for suf in
                           ['_mw', '_mw_lp']) or c.startswith('ntc_total_')]
        for k in cap_keys:
            try:
                rec[k] = float(last[k])
            except (ValueError, TypeError):
                pass
        # Per-(node, tech) VFs as well (so we see WHERE the seed bound).
        for col in last.index:
            if col.startswith('vf_') and col not in ('vf_min', 'vf_target'):
                try:
                    rec[col] = float(last[col])
                except (ValueError, TypeError):
                    pass
        all_records.append(rec)
        print(f"  [restart {r+1}] VF_min={vf_min:.4f}, premium={premium:.3f} bn, "
              f"penalised={penalized:.4e}")
        if penalized < best_pen:
            best_pen = penalized
            best_df = df_r
            best_baseline_cost = baseline_cost

    if best_df is None:
        # Fallback: all restarts failed — return whatever the last seed
        # produced so the run at least writes something.
        return df_r, all_records, baseline_cost
    return best_df, all_records, best_baseline_cost


def write_sensitivity_xlsx(out_path, args, D, df_frontier, baseline_cost,
                           restart_records=None, eval_log_tail=None,
                           wall_time_s=None):
    """Write the consolidated multi-sheet Excel result for ONE
    sensitivity run. Sheets:

      meta           One-row summary of flags, seed, runtime, achieved.
      frontier       VF target, achieved VF_min, premium, sys_cost,
                     evals_used, status flags, vf_<tech>_<node> (wide).
      portfolios     VF target, all capacity x_keys, NTC totals,
                     LP-endogenous capacities for low-VRE nodes.
      vfs            VF target, vf_<tech>_<node> per pair (long).
      ntc            VF target, ntc_invest_<link>, ntc_total_<link>.
      heat           VF target, HP electric load, storage, backup
                     (only if enable_heat=True; sheet skipped otherwise).
      binding        VF target, binding (node,tech) pair, runner-up.
      restarts       seed -> VF_min, premium, penalised, status.
      eval_log_tail  Last ~1000 evals (x vector, vf_min, cost) for
                     post-hoc landscape inspection.
    """
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- META --------------------------------------------------------
    meta_rows = [
        ('run_tag',        getattr(args, 'run_tag', None) or _build_run_tag(args)),
        ('nodes',          ','.join(D['n'])),
        ('n_repr_weeks',   D['n_repr_weeks']),
        ('dk_h2_twh',      args.dk_h2_twh),
        ('dk_nordic_mult', args.dk_nordic_mult),
        ('ntc_cost_mult',  args.ntc_cost_mult),
        ('enable_heat',    args.enable_heat),
        ('dk_heat_twh',    args.dk_heat_twh if args.enable_heat else None),
        ('dk_hp_mw',       args.dk_hp_mw if args.enable_heat else None),
        ('dk_heat_storage_mwh', args.dk_heat_storage_mwh if args.enable_heat else None),
        ('dk_cop',         args.dk_cop if args.enable_heat else None),
        ('cma_seed',       args.cma_seed),
        ('n_restarts',     args.n_restarts),
        ('n_workers',      args.n_workers),
        ('parallel_mode',  args.parallel_mode),
        ('diagnostic_vf',  args.diagnostic_vf),
        ('diagnostic_evals', args.diagnostic_evals),
        ('baseline_cost_bn', baseline_cost / 1e9 if baseline_cost else None),
        ('wall_time_s',    wall_time_s),
        ('wall_time_h',    wall_time_s / 3600.0 if wall_time_s else None),
    ]
    meta_df = pd.DataFrame(meta_rows, columns=['key', 'value'])

    # ---- FRONTIER (the headline table) -------------------------------
    # Pull the columns most useful for paper tables.
    headline_cols = ['vf_target', 'vf_min', 'premium_bn', 'sys_cost',
                     'vf_pv', 'vf_wind_on', 'vf_wind_off']
    fr_df = df_frontier.copy()
    for c in headline_cols:
        if c not in fr_df.columns:
            fr_df[c] = np.nan
    front_out = fr_df[[c for c in headline_cols if c in fr_df.columns]].copy()

    # ---- PORTFOLIOS (wide capacity table) ----------------------------
    cap_cols = [c for c in fr_df.columns
                if any(c.endswith(suf) for suf in
                       ['_mw', '_mw_lp']) or c.startswith('ntc_total_')]
    portfolios_df = fr_df[['vf_target', 'vf_min'] + cap_cols].copy() if cap_cols else fr_df[['vf_target', 'vf_min']].copy()

    # ---- VFs (long format: one row per (target, node, tech)) ---------
    vfs_long = []
    for _, row in fr_df.iterrows():
        vf_t = row.get('vf_target')
        for n in D['n']:
            for v in D['tech_vre']:
                key = f'vf_{v}_{n}'
                if key in row and not pd.isna(row[key]) and row[key] > 1e-9:
                    vfs_long.append({
                        'vf_target': vf_t,
                        'node': n,
                        'tech': v,
                        'vf': float(row[key]),
                    })
    vfs_df = pd.DataFrame(vfs_long)

    # ---- NTC table ---------------------------------------------------
    ntc_rows = []
    links = D.get('links', {})
    for _, row in fr_df.iterrows():
        rec = {'vf_target': row.get('vf_target'),
               'vf_min':    row.get('vf_min')}
        for l in links:
            l_lc = l.lower()
            inv_key   = f'ntc_{l_lc}_mw'
            total_key = f'ntc_total_{l_lc}_mw'
            rec[f'fixed_{l_lc}'] = links[l]['ntc_fixed']
            rec[f'invest_{l_lc}'] = float(row.get(inv_key, 0.0)) if inv_key in row else None
            rec[f'total_{l_lc}'] = float(row.get(total_key, links[l]['ntc_fixed'])) if total_key in row else None
        ntc_rows.append(rec)
    ntc_df = pd.DataFrame(ntc_rows)

    # ---- HEAT table --------------------------------------------------
    heat_df = None
    if args.enable_heat:
        heat_cols = ['hp_dk_total_twh_e', 'hp_dk_avg_load_mw_e',
                     'hp_dk_backup_twh_th', 'hp_dk_soc_max_mwh',
                     'hp_dk_soc_mean_mwh']
        # These are reported by LPOracle.__call__ when heat is enabled
        # but only persisted in `df_frontier` if `_frontier_row` plumbed
        # them through — we extract directly from the last LP solve in
        # an upstream call instead of relying on the frontier dataframe.
        heat_cols_present = [c for c in heat_cols if c in fr_df.columns]
        if heat_cols_present:
            heat_df = fr_df[['vf_target', 'vf_min'] + heat_cols_present].copy()
        else:
            heat_df = pd.DataFrame({'note': ['heat enabled but per-target '
                                             'HP metrics not in frontier '
                                             'dataframe; check '
                                             '_frontier_row']})

    # ---- BINDING constraint per target (which (node,tech) is min VF) -
    binding_rows = []
    for _, row in fr_df.iterrows():
        vf_t = row.get('vf_target')
        pair_vfs = []
        for n in D['n']:
            for v in D['tech_vre']:
                key = f'vf_{v}_{n}'
                if key in row and not pd.isna(row[key]) and row[key] > 1e-9:
                    pair_vfs.append((row[key], n, v))
        pair_vfs.sort()
        rec = {'vf_target': vf_t, 'vf_min_observed': row.get('vf_min')}
        if len(pair_vfs) >= 1:
            rec['binding_node'] = pair_vfs[0][1]
            rec['binding_tech'] = pair_vfs[0][2]
            rec['binding_vf']   = pair_vfs[0][0]
        if len(pair_vfs) >= 2:
            rec['runner_up_node'] = pair_vfs[1][1]
            rec['runner_up_tech'] = pair_vfs[1][2]
            rec['runner_up_vf']   = pair_vfs[1][0]
            rec['gap_to_runner_up'] = pair_vfs[1][0] - pair_vfs[0][0]
        binding_rows.append(rec)
    binding_df = pd.DataFrame(binding_rows)

    # ---- RESTARTS table ----------------------------------------------
    restarts_df = pd.DataFrame(restart_records) if restart_records else None

    # ---- EVAL LOG tail -----------------------------------------------
    eval_log_df = None
    if eval_log_tail:
        eval_log_df = pd.DataFrame(eval_log_tail)

    with pd.ExcelWriter(out_path, engine='openpyxl') as xw:
        meta_df.to_excel(xw, sheet_name='meta', index=False)
        front_out.to_excel(xw, sheet_name='frontier', index=False)
        portfolios_df.to_excel(xw, sheet_name='portfolios', index=False)
        if not vfs_df.empty:
            vfs_df.to_excel(xw, sheet_name='vfs', index=False)
        ntc_df.to_excel(xw, sheet_name='ntc', index=False)
        if heat_df is not None:
            heat_df.to_excel(xw, sheet_name='heat', index=False)
        binding_df.to_excel(xw, sheet_name='binding', index=False)
        if restarts_df is not None:
            restarts_df.to_excel(xw, sheet_name='restarts', index=False)
        if eval_log_df is not None:
            eval_log_df.to_excel(xw, sheet_name='eval_log_tail', index=False)
    print(f"\n  Sensitivity result written: {out_path}")


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Directed Search v12 — Least-cost VF frontier')
    parser.add_argument('--ev-cap', type=float, default=None,
                        help='Override EV capacity cap [MW]. Default: use data_pipeline value.')
    parser.add_argument('--no-monotone', action='store_true',
                        help='Skip the monotone (path-dependent) frontier to save time.')
    parser.add_argument('--single-node', action='store_true',
                        help='Run single-node (DE only) mode.')
    parser.add_argument('--nodes', type=str, default=None,
                        help='Comma-separated node list, e.g. "DE,CZ,DK,AT,NL". Default: DE,CZ,DK,AT,NL.')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip post-run scalar validation.')
    parser.add_argument('--fast', action='store_true',
                        help='Use slim VF target grid (5 points) for quick test runs.')
    parser.add_argument('--optimizer', type=str, default='cma',
                        choices=['cma', 'nelder-mead'],
                        help='Outer optimizer: "cma" (CMA-ES, default) or "nelder-mead" (scipy).')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Parallel LP workers for CMA-ES generation evaluation '
                             '(1 = serial; 2-4 typical on multi-core machines). '
                             'Each worker builds its own LPOracle at startup '
                             '(~30-60 s per worker, one-time cost).')
    parser.add_argument('--parallel-mode', type=str, default='process',
                        choices=['process', 'thread'],
                        help='Parallelism strategy (only if --n-workers > 1). '
                             '"process" (default) = separate Python processes, '
                             'each with its own Gurobi env. Reliable over long '
                             'runs. "thread" = shared process, threads sharing '
                             'Gurobi state; faster startup but fails after a '
                             'few hundred solves due to Gurobi C-state '
                             'corruption. Use "thread" only for short tests.')
    parser.add_argument('--diagnostic-vf', type=float, default=None,
                        help='Single-target diagnostic mode: run only ONE VF target '
                             'at a high eval budget to verify the optimizer can '
                             'actually reach the ceiling. Skips the full sweep, '
                             'skips monotone, skips validation unless --validate. '
                             'Example: --diagnostic-vf 0.87')
    parser.add_argument('--diagnostic-evals', type=int, default=3000,
                        help='Eval budget for --diagnostic-vf (default 3000).')
    # ---------- Sensitivity flags (Tier 1 / Tier 2) ----------
    parser.add_argument('--dk-h2-twh', type=float, default=5.0,
                        help='DK-specific H2 quota in TWh/yr (Tier 1 #1). '
                             'Default 5.0 (= moderate Danish PtX policy by '
                             '2030). Set 0 to disable. Test 0/5/10.')
    parser.add_argument('--dk-nordic-mult', type=float, default=1.0,
                        help='Multiplier on the 4800 MW DK->Nordic '
                             'residual NTC (Tier 1 #3). Default 1.0. '
                             'Test 1.5 and 2.0.')
    parser.add_argument('--ntc-cost-mult', type=float, default=1.0,
                        help='Multiplier on annualised NTC capex '
                             '(Tier 2 #8). Default 1.0. Test 0.5 and 2.0.')
    parser.add_argument('--enable-heat', action='store_true',
                        help='Enable DK heat-pump coupling (Tier 2 #6). '
                             'Adds 2 GW_e fixed HP, 30 GWh_th storage, '
                             '15 TWh_th annual heat demand to DK.')
    parser.add_argument('--dk-heat-twh', type=float, default=15.0,
                        help='Annual DK heat-pump heat demand TWh_th '
                             '(default 15). Only active if --enable-heat.')
    parser.add_argument('--dk-hp-mw', type=float, default=2000.0,
                        help='DK heat-pump electric capacity MW_e '
                             '(default 2000). Only active if --enable-heat.')
    parser.add_argument('--dk-heat-storage-mwh', type=float, default=30000.0,
                        help='DK thermal storage MWh_th '
                             '(default 30000). Only active if --enable-heat.')
    parser.add_argument('--dk-cop', type=float, default=3.0,
                        help='DK HP coefficient of performance (default 3.0).')
    parser.add_argument('--n-restarts', type=int, default=1,
                        help='Multi-restart count for diagnostic mode. '
                             'Each restart uses seed = base + restart_index. '
                             'Best penalised-fitness result is reported. '
                             'Default 1 (no multi-restart). Recommended 3 for '
                             'sensitivity runs (~3x compute, mitigates '
                             'basin divergence; see progress.md sec 33).')
    parser.add_argument('--scenario', type=str, default='tyndp_de_2040',
                        choices=['tyndp_de_2040', 'de2035', 'de2040', 'de2050',
                                 'nt2030', 'legacy'],
                        help="Parameter bundle / year (scenarios.py): "
                             "'tyndp_de_2040' (default) or 'legacy' (pre-TYNDP, "
                             "to reproduce old results).")
    parser.add_argument('--max-evals', type=int, default=None,
                        help='Override the per-target CMA-ES evaluation budget '
                             '(default dimension-tuned 1200/1800). Use a small '
                             'value (e.g. 300) for fast low-dimension '
                             '(single-node) runs (progress.md T3.6).')
    parser.add_argument('--cma-seed', type=int, default=42,
                        help='Base CMA-ES seed (default 42). With '
                             '--n-restarts > 1, seeds are 42, 43, 44, ...')
    parser.add_argument('--target', type=str, default='vf',
                        choices=['vf', 'vfw', 'cr'],
                        help="Outer-search constraint metric. 'vf' (default) "
                             "targets the cannibalisation channel (value "
                             "factor) -- the paper's primary frontier; 'cr' "
                             "targets cost recovery (investability). CR and VF "
                             "are reported in BOTH cases (decomposition "
                             "CR = VF * base_price/LCOE).")
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Optional short label for the consolidated '
                             'output file name. Default builds a tag from '
                             'the active sensitivity flags.')
    # ---- S.56 additions -------------------------------------------------
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted frontier run from the '
                             'per-target checkpoint CSV (written after every '
                             'completed VF target). Verifies the baseline '
                             'still matches before reusing rows.')
    parser.add_argument('--target-slack', type=float, default=0.01,
                        help='Inflate the INTERNAL optimization target by '
                             'this amount to offset the quadratic-penalty '
                             'undershoot (~0.02); acceptance uses the true '
                             'target. 0 disables (old behaviour). Default 0.01.')
    parser.add_argument('--polish-evals', type=int, default=300,
                        help='Budget for the saturation-freeze polish: after '
                             'the multi-restart, freeze dims at >=98%% of '
                             'their bound and re-run CMA on the remaining '
                             'low-dim subspace. 0 disables. Default 300.')
    parser.add_argument('--honesty-evals', type=int, default=6,
                        help='Max verification LP solves for the first-order '
                             '(reduced-cost subgradient) honesty check at '
                             'each frontier point. 0 disables. Default 6.')
    parser.add_argument('--refine-iters', type=int, default=8,
                        help='Max iterations of the S.60 convexity-aware '
                             'gradient-descent refinement at each frontier '
                             'point (exact-subgradient multi-dim descent; the '
                             'real steep-region cost reduction before the '
                             'honesty certificate). 0 disables. Default 8.')
    parser.add_argument('--climate-year', type=int, default=None,
                        help='PECD v3.1 climate year (1982-2019) for VRE '
                             'capacity factors (multi-weather-year robustness, '
                             'Phase B). Default None = DIETER time series. '
                             'Requires PECD parquets in tyndp 2040/PECD/.')
    parser.add_argument('--pecd-dir', type=str, default=None,
                        help='PECD parquet directory (default: '
                             '<model>/tyndp 2040/PECD).')
    parser.add_argument('--n-weeks', type=int, default=6,
                        help='Number of k-medoids representative weeks '
                             '(default 6). Use 8/12 for the week-count '
                             'robustness sweep (S.56; Kotzur linking works '
                             'for any count).')
    parser.add_argument('--resid-export-price', type=float, default=None,
                        help='Override price_export [EUR/MWh] on ALL residual '
                             'interconnectors (sensitivity; default: keep '
                             'data_pipeline values).')
    parser.add_argument('--resid-import-price', type=float, default=None,
                        help='Override price_import [EUR/MWh] on ALL residual '
                             'interconnectors (sensitivity).')
    parser.add_argument('--no-resid', action='store_true',
                        help='Remove ALL residual interconnectors (isolated-'
                             'system bound for the residual-price '
                             'sensitivity bracket).')
    parser.add_argument('--h2-2040', action='store_true',
                        help='Set per-node H2 quota to TYNDP 2024 DE2040 '
                             'electrolyser electricity x sc_eta (DE 106.7, '
                             'CZ 3.6, DK 60.0, AT 25.5, NL 42.7 TWh H2). '
                             'Generalises the H2 constraint to all 5 nodes '
                             '(S.61). Default off (legacy DE 20 / DK via '
                             '--dk-h2-twh). NB DK is a TYNDP H2-export hub.')
    parser.add_argument('--demand-2040', action='store_true',
                        help='Scale each node base demand to its TYNDP 2024 '
                             'Distributed Energy 2040 final-demand total (excl. '
                             'electrolysis; incl. EV + heat + industrial '
                             'electrification). Fixes the S.61 under-count '
                             '(model had only ~today demand vs 2040 VRE -> '
                             'overstated cannibalization). Default off.')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Solve only the cost-optimal B_opt (no frontier) '
                             'and print VF_min/VF_genw/CR_min per node, then '
                             'exit. Pre-flight / week-count validation.')
    parser.add_argument('--ev-2040', action='store_true',
                        help='Realistic EV: fixed TYNDP 2040 passenger fleet '
                             '(driving-demand sink + plug-in availability + '
                             'V2G), K_ev fixed at fleet power, EV dropped from '
                             'the outer search, EV capex=0 (consumer-owned). '
                             'Requires --demand-2040 (base auto-excludes EV '
                             'charging to avoid double count). Default off.')
    parser.add_argument('--p2g-endogenous', action='store_true',
                        help='Drop electrolyser (P2G) from the outer search '
                             'and let the LP size it (floored by the H2 quota). '
                             'P2G does not raise VF (grid-floor evidence, '
                             'S.59.2), so this removes 5 non-helping dims '
                             '(26->21) and speeds/stabilises CMA. Default off '
                             '(P2G stays an outer dimension).')
    args = parser.parse_args()

    wall_start = time.time()

    if args.single_node:
        nodes = ['DE']
    elif args.nodes:
        nodes = [n.strip() for n in args.nodes.split(',')]
    else:
        nodes = ['DE', 'CZ', 'DK', 'AT', 'NL']  # default: 5-node v12
    mode_str = f"{len(nodes)}-node ({'+'.join(nodes)})"
    opt_label = 'CMA-ES' if args.optimizer == 'cma' else 'Nelder-Mead'
    print("="*60)
    print(f"  DIRECTED SEARCH v11 ({opt_label}, {mode_str})")
    print("="*60)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ts_path = os.path.join(root, 'time_series.xlsx')
    di_path = os.path.join(root, 'data_input.xlsx')

    n_steps = 5 if not args.no_validate else 4
    print(f"\n[1/{n_steps}] Loading data ({args.n_weeks} representative weeks, nodes={nodes})...")
    _pecd_dir = args.pecd_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tyndp 2040', 'PECD')
    D = load_representative_weeks(
        ts_path, di_path, n_weeks=args.n_weeks, nodes=nodes,
        climate_year=args.climate_year,
        pecd_dir=(_pecd_dir if args.climate_year is not None else None),
        dk_h2_twh=args.dk_h2_twh,
        dk_nordic_mult=args.dk_nordic_mult,
        ntc_cost_mult=args.ntc_cost_mult,
        enable_heat=args.enable_heat,
        dk_heat_twh=args.dk_heat_twh,
        dk_hp_mw=args.dk_hp_mw,
        dk_heat_storage_mwh=args.dk_heat_storage_mwh,
        dk_cop=args.dk_cop,
    )

    from scenarios import apply_scenario
    apply_scenario(D, args.scenario)
    # ---- S.86: deployment-potential overrides for sensitivity runs ----
    # Env CAP_OVERRIDES="NL.k_bar_sto_p=80000;NL.k_bar_sto_e=320000"
    # Applied to node_params BEFORE dims/LPOracle -> outer bounds + LP caps move.
    _ovr = os.environ.get('CAP_OVERRIDES', '').strip()
    if _ovr:
        for _tok in _ovr.split(';'):
            _tok = _tok.strip()
            if not _tok:
                continue
            _lhs, _val = _tok.split('=')
            _nd, _key = _lhs.strip().split('.')
            _np_ = D.get('node_params', {}).get(_nd)
            if _np_ is None:
                raise SystemExit(f"[CAP_OVERRIDES] unknown node '{_nd}'")
            _old = _np_.get(_key, '<absent>')
            _np_[_key] = float(_val)
            print(f"  ** CAP OVERRIDE: {_nd}.{_key} {_old} -> {float(_val):,.0f} **")
    # ---- S.96: storage-duration sensitivity (supervisor: ceiling assumptions) ----
    # Env STO_EP_RATIO=8 -> energy/power ratio 8 h instead of the default 4 h on
    # every node (fixed portfolios, B_opt, honesty pins). Energy caps are scaled
    # to ratio * power cap so the power potential stays the binding bound.
    # Unset/empty -> 4 h, bit-identical to all production runs.
    _ep = os.environ.get('STO_EP_RATIO', '').strip()
    if _ep:
        _r = float(_ep)
        D['sto_ep_ratio'] = _r
        if 'k_bar_sto_p' in D:
            D['k_bar_sto_e'] = _r * D['k_bar_sto_p']
        for _nd, _np_ in D.get('node_params', {}).items():
            if 'k_bar_sto_p' in _np_:
                _np_['k_bar_sto_e'] = _r * _np_['k_bar_sto_p']
        print(f"  ** STO_EP_RATIO: storage duration {_r:g} h on all nodes "
              f"(energy caps scaled to {_r:g} x power caps) **")
    # ---- S.100: TYNDP offshore hubs (see scenario_data.HUBS_*) ----
    # OFFSHORE_HUBS=capacity -> add hub installed MW to the node's offshore wind
    # (node PECD profile, so less energy than the far-offshore hubs deliver).
    # OFFSHORE_HUBS=energy   -> add the MW that reproduces the hubs' annual
    # energy on the node profile (K = E / sum_t w_t phi_t). Unset -> original.
    _hub = os.environ.get('OFFSHORE_HUBS', '').strip().lower()
    if _hub:
        from scenario_data import HUBS_MW, HUBS_TWH
        _sk = args.scenario
        if _sk not in HUBS_MW:
            print(f"  ** OFFSHORE_HUBS: no hub data for '{_sk}' (none in TYNDP) -> unchanged **")
        else:
            _W = D['weights']
            for _nd in HUBS_MW[_sk]:
                _phi = D['phi_wind_off'][_nd]
                _flh = sum(_W[t] * _phi[t] for t in D['h'])   # MWh per MW-year
                if _hub == 'capacity':
                    _add = float(HUBS_MW[_sk][_nd])
                elif _hub == 'energy':
                    _add = HUBS_TWH[_sk][_nd] * 1e6 / _flh
                else:
                    raise SystemExit(f"OFFSHORE_HUBS must be capacity|energy, got {_hub}")
                _targets = [D['node_params'][_nd]['k_bar_vre']] if _nd in D.get('node_params', {}) else []
                if _nd == 'DE' and 'k_bar_vre' in D and all(D['k_bar_vre'] is not t for t in _targets):
                    _targets.append(D['k_bar_vre'])
                _old = _targets[0]['wind_off']
                for _t in _targets:
                    _t['wind_off'] = _old + _add
                print(f"  ** OFFSHORE_HUBS={_hub}: {_nd} wind_off {_old/1000:.1f} -> "
                      f"{(_old+_add)/1000:.1f} GW (+{_add/1000:.1f} GW; node CF "
                      f"{_flh/sum(_W[t] for t in D['h']):.3f}, hub energy "
                      f"{HUBS_TWH[_sk][_nd]:.1f} TWh) **")
    # ---- S.103: remaining TYNDP components (env TYNDP_FIXES, see tyndp_fixes.py)
    import tyndp_fixes as _tf
    if _tf.requested() and os.environ.get('OFFSHORE_HUBS', '').strip():
        raise SystemExit('Use either OFFSHORE_HUBS (S.100 test) or TYNDP_FIXES, not both')
    _tf.apply(D, args.scenario)
    D['p2g_endogenous'] = bool(args.p2g_endogenous)
    if args.ev_2040:
        from data_pipeline import build_ev_2040
        D['ev_capex'] = 0.0  # exogenous fleet (consumer-owned) -> no system capex
        build_ev_2040(D)
    if args.demand_2040:
        from data_pipeline import scale_demand_to_targets
        print("\n  ** --demand-2040: scaling base demand to TYNDP DE2040 "
              "final-demand totals (S.61) **")
        scale_demand_to_targets(D)
    if args.h2_2040:
        eta = D.get('sc_eta', 0.70)
        _EL = D.get('h2_elec_twh')
        if not _EL:
            from data_pipeline import TYNDP_DE2040_ELECTROLYSER_TWH as _EL
        D['h2_targets'] = {n: _EL[n] * 1e6 * eta for n in D['n'] if n in _EL}
        print("\n  ** --h2-2040: per-node H2 quota = TYNDP DE2040 electrolyser "
              "x eta (S.61) [TWh H2]: "
              + ", ".join(f"{n}={D['h2_targets'][n]/1e6:.1f}" for n in D['h2_targets'])
              + " ** (DK is a TYNDP H2-export hub -> large flexible load)")
        # ---- S.87: per-node H2-quota override for sensitivity runs ----
        # Env H2_OVERRIDES="DK=5" (TWh H2) -- e.g. DK domestic-only variant
        # (drops the TYNDP export-hub role). Applied BEFORE the preflight.
        _h2ovr = os.environ.get('H2_OVERRIDES', '').strip()
        if _h2ovr:
            for _tok in _h2ovr.split(';'):
                _tok = _tok.strip()
                if not _tok:
                    continue
                _nd, _twh = _tok.split('=')
                _nd = _nd.strip()
                if _nd not in D['h2_targets']:
                    raise SystemExit(f"[H2_OVERRIDES] unknown node '{_nd}'")
                _oldv = D['h2_targets'][_nd] / 1e6
                D['h2_targets'][_nd] = float(_twh) * 1e6
                print(f"  ** H2 OVERRIDE: {_nd} quota {_oldv:.1f} -> "
                      f"{float(_twh):.1f} TWh H2 **")
        # ---- S.84.1 preflight: EVERY node's quota vs its electrolyser cap.
        # (de2050-AT breached silently -> cryptic infeasible B_opt after a
        # full data load; the old check covered DE/DK only. Fail fast + loud.)
        _W = D.get('weights') or {}
        _H = sum(_W.get(t, 1.0) for t in D['h']) or 8736.0
        _bad, _tight = [], []
        for _n, _h2 in D['h2_targets'].items():
            _need = (_h2 / eta) / _H          # MW of P2G at CF 100%
            _cap = D.get('node_params', {}).get(_n, {}).get(
                'k_bar_sc', D.get('k_bar_sc', 200000))
            if _need > _cap:
                _bad.append(f"{_n}: quota needs >={_need:,.0f} MW P2G at CF "
                            f"100% > k_bar_sc {_cap:,.0f} MW")
            elif _need > 0.8 * _cap:
                _tight.append(f"{_n}: quota forces P2G CF >={_need/_cap:.0%}")
        if _tight:
            print("  [preflight WARN] H2 quota tight vs cap: " + "; ".join(_tight))
        if _bad:
            raise SystemExit("[PREFLIGHT FAIL] H2 quota INFEASIBLE vs "
                             "electrolyser caps (B_opt would be infeasible):\n  "
                             + "\n  ".join(_bad))

    # ---- Residual-interconnector sensitivity overrides (S.56) ----------
    if args.no_resid:
        n_dropped = len(D.get('resid_intercon', {}))
        D['resid_intercon'] = {}
        print(f"\n  ** --no-resid: removed {n_dropped} residual "
              f"interconnector(s) (isolated-system bound) **")
    else:
        for _rn, _rd in D.get('resid_intercon', {}).items():
            if args.resid_export_price is not None:
                _rd['price_export'] = float(args.resid_export_price)
                _rd.pop('price_export_series', None)
            if args.resid_import_price is not None:
                _rd['price_import'] = float(args.resid_import_price)
                _rd.pop('price_import_series', None)
        if args.resid_export_price is not None or args.resid_import_price is not None:
            print(f"\n  ** Residual price override: export="
                  f"{args.resid_export_price}, import="
                  f"{args.resid_import_price} EUR/MWh (all residual links) **")

    # Override EV cap if specified via CLI
    if args.ev_cap is not None:
        D['k_bar_ev'] = args.ev_cap
        print(f"\n  ** EV capacity cap overridden: {args.ev_cap:,.0f} MW ({args.ev_cap/1000:.0f} GW) **")
    else:
        print(f"\n  EV capacity cap: {D.get('k_bar_ev', 50000):,.0f} MW (default)")

    print(f"\n  H2 demand target: {D['sc_h2_target']/1e6:.0f} TWh_H2 (active only when P2G > 0)")

    # ==== DIAGNOSTIC MODE: single VF target, high eval budget ====
    # Used both as a budget-starvation diagnostic and as the workhorse
    # mode for Tier 1 / Tier 2 sensitivity runs. With --n-restarts > 1
    # the same VF target is solved with N seeds (42, 43, ...); the best
    # *penalised fitness* across restarts is the reported result and
    # all per-restart records are saved on the 'restarts' sheet.
    if args.diagnostic_vf is not None:
        vf_d = float(args.diagnostic_vf)
        ev_d = int(args.diagnostic_evals)
        print(f"\n[2/2] DIAGNOSTIC MODE — single VF target = {vf_d:.3f}, "
              f"budget = {ev_d} evals, workers = {args.n_workers} "
              f"({args.parallel_mode} mode), restarts = {args.n_restarts}")
        print(f"  Sensitivity tag: {args.run_tag or _build_run_tag(args)}")
        print(f"  (skipping monotone + validation; focus on one solve)")
        diag_start = time.time()

        df, restart_records, baseline_cost = _diagnostic_multi_restart(
            D, vf_target=vf_d, max_evals=ev_d,
            n_workers=args.n_workers,
            parallel_mode=args.parallel_mode,
            base_seed=args.cma_seed,
            n_restarts=args.n_restarts,
        )

        diag_dt = time.time() - diag_start
        print(f"\n  Diagnostic target VF*={vf_d} complete in {diag_dt:.1f}s "
              f"({diag_dt/60:.1f} min, {diag_dt/3600:.2f} h)")

        # ---- Consolidated xlsx output (Tier 1 / Tier 2 ready) -------
        run_tag = args.run_tag or _build_run_tag(args)
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'output', 'sensitivity')
        out_xlsx = os.path.join(
            outdir, f'sensitivity_vf{vf_d:.3f}_{run_tag}.xlsx')

        if df is not None and not df.empty:
            print(f"\n  === DIAGNOSTIC RESULT ===")
            print(f"  Rows in result: {len(df)}")
            last = df.iloc[-1]
            print(f"  Achieved VF_min = {last['vf_min']:.4f} "
                  f"(target {vf_d:.3f})")
            print(f"  Premium = {last['premium_bn']:.3f} bn EUR")
            # Print which (node, tech) pair is binding
            pair_vfs = []
            for n in D['n']:
                for v in D['tech_vre']:
                    k = f'vf_{v}_{n}'
                    if k in last and not pd.isna(last[k]) and last[k] > 1e-9:
                        pair_vfs.append((float(last[k]), n, v))
            pair_vfs.sort()
            if pair_vfs:
                print(f"  Binding pair: {pair_vfs[0][1]} {pair_vfs[0][2]}, "
                      f"VF={pair_vfs[0][0]:.4f}")
                if len(pair_vfs) >= 2:
                    print(f"  Runner-up:    {pair_vfs[1][1]} {pair_vfs[1][2]}, "
                          f"VF={pair_vfs[1][0]:.4f}")

            # Write the consolidated multi-sheet xlsx.
            write_sensitivity_xlsx(
                out_xlsx, args, D, df, baseline_cost,
                restart_records=restart_records,
                eval_log_tail=None,
                wall_time_s=diag_dt,
            )

            # Also keep a legacy CSV for backwards compatibility with
            # earlier diagnostic-mode tooling.
            legacy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'output', f'diagnostic_vf_{vf_d:.3f}'
                                      + (f'_{args.run_tag}' if args.run_tag else ''))
            os.makedirs(legacy_dir, exist_ok=True)
            df.to_csv(os.path.join(legacy_dir, 'diagnostic_frontier.csv'),
                      index=False)
        else:
            print(f"\n  ** DIAGNOSTIC FAILED: no frontier returned **")

        wall_elapsed = time.time() - wall_start
        print(f"\n{'='*60}")
        print(f"  TOTAL WALL-CLOCK TIME: {wall_elapsed:.1f} s "
              f"({wall_elapsed/60:.1f} min, {wall_elapsed/3600:.2f} h)")
        print(f"{'='*60}")
        sys.exit(0)

    # VF target grid: slim for --fast, full otherwise
    if args.fast and args.target == 'cr':
        vf_grid = [0.85, 0.95, 1.00, 1.05]
        print(f"\n  ** FAST MODE: slim CR grid ({len(vf_grid)} targets) **")
    else:
        vf_grid = None  # adaptive VF grid / default CR grid in trace_pareto_frontier

    if args.baseline_only:
        print("\n[baseline-only] Solving cost-optimal B_opt and exiting...")
        _o = LPOracle(D)
        _b = _o.solve_cost_optimal()
        if _b is None:
            print("  ** B_opt FAILED **"); sys.exit(1)
        print(f"\n  B_opt cost {_b['sys_cost']/1e9:.3f} bn | VF_min "
              f"{_o.compute_vf_min(_b):.4f} | VF_genw {_o.compute_vf_genw(_b):.4f}"
              f" | CR_min {_o.compute_cr_min(_b):.4f}")
        for _n in _o.all_nodes:
            print(f"    {_n}: " + ", ".join(
                f"VF_{_v}={_b.get(f'vf_{_v}_{_n}', 0):.3f}" for _v in D['tech_vre']))
        sys.exit(0)

    print(f"\n[2/{n_steps}] Tracing global frontier ({len(D['h'])}h model, {opt_label})...")
    notify(f"Run start: {mode_str}, scenario={args.scenario}, "
           f"target={args.target}, workers={args.n_workers}, "
           f"restarts={args.n_restarts}")
    _rid = _run_id(args, nodes)
    _ck_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'output', f'checkpoint_frontier_{_rid}.csv')
    print(f"  Run id: {_rid}")
    os.makedirs(os.path.dirname(_ck_path), exist_ok=True)
    if args.resume:
        print(f"  Resume requested; checkpoint: {_ck_path}")
    else:
        print(f"  Per-target checkpoint: {_ck_path}")
    df, baseline_cost = trace_pareto_frontier(
        D, vf_targets=vf_grid, optimizer=args.optimizer,
        n_workers=args.n_workers, parallel_mode=args.parallel_mode,
        target_metric=args.target, n_restarts=args.n_restarts,
        cma_seed=args.cma_seed, max_evals_override=args.max_evals,
        target_slack=args.target_slack, polish_evals=args.polish_evals,
        honesty_evals=args.honesty_evals, refine_iters=args.refine_iters,
        checkpoint_path=_ck_path, resume=args.resume,
    )

    if not args.no_monotone:
        print(f"\n[3/{n_steps}] Tracing monotone (path-dependent) frontier ({opt_label})...")
        import copy
        D2 = copy.deepcopy(D)
        _ck_mono = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'output', f'checkpoint_monotone_{_rid}.csv')
        df_mono, _ = trace_monotone_frontier(
            D2, vf_targets=vf_grid, optimizer=args.optimizer,
            n_workers=args.n_workers, parallel_mode=args.parallel_mode,
            target_metric=args.target, max_evals_override=args.max_evals,
            target_slack=args.target_slack, honesty_evals=args.honesty_evals,
            checkpoint_path=_ck_mono, resume=args.resume,
        )
    else:
        print(f"\n[3/{n_steps}] Monotone frontier SKIPPED (--no-monotone)")
        df_mono = pd.DataFrame()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

    # If EV cap overridden, save to a subfolder so runs don't overwrite each other
    if args.ev_cap is not None:
        ev_label = f"ev_{int(args.ev_cap/1000)}gw"
        outdir = os.path.join(outdir, ev_label)

    print(f"\n[4/{n_steps}] Plotting...")
    if df.empty:
        print("  Skipping plot — no frontier points (baseline solve failed).")
    else:
        plot_frontier(df, baseline_cost, outdir, df_monotone=df_mono if not df_mono.empty else None, tag=_rid)

    # Print path-dependency cost comparison
    if not df.empty and not df_mono.empty:
        print("\n  PATH DEPENDENCY COST COMPARISON:")
        for _, row_m in df_mono.iterrows():
            cr_m = row_m['cr_min']
            # Closest global frontier point at the same CR level
            closest_g = df.iloc[(df['cr_min'] - cr_m).abs().argsort()[:1]]
            if len(closest_g) > 0 and abs(closest_g['cr_min'].iloc[0] - cr_m) < 0.02:
                # Path dependency = absolute system-cost gap (baseline-free;
                # progress.md Section 36.3). Monotone premium is vs B_today
                # and global premium vs B_opt, so premium differences are
                # NOT comparable -- compare sys_cost directly.
                gap = (row_m['sys_cost'] - closest_g['sys_cost'].iloc[0]) / 1e9
                if abs(gap) > 0.001:
                    print(f"    CR~{cr_m:.3f}: global cost={closest_g['sys_cost'].iloc[0]/1e9:.3f}, "
                          f"monotone cost={row_m['sys_cost']/1e9:.3f}, "
                          f"path-dep. penalty={gap:+.3f} bn EUR")

    # ---- Post-run frontier quality validation ----
    if not args.no_validate and not df.empty:
        print(f"\n[5/{n_steps}] Validating frontier quality...")
        import copy
        D_val = copy.deepcopy(D)
        val_results = validate_frontier_quality(D_val, df, n_perturb=5, target_metric=args.target)
        if val_results:
            print(f"\n  VALIDATION SUMMARY: {val_results.get('status', 'N/A')}")
            if val_results.get('status') == 'WARN':
                print(f"    Cost improvement potential >{val_results['max_improvement_pct']:.1f}% — "
                      f"consider increasing eval budget")

    wall_elapsed = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"  TOTAL WALL-CLOCK TIME: {wall_elapsed:.1f} s ({wall_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    notify(f"DONE: {mode_str}, target={args.target}, {wall_elapsed/3600:.2f} h")
    print("\nDone. Check output/ for frontier CSV and plots.")