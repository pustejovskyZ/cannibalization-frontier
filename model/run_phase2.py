"""
run_phase2.py -- phase 2 of the rerun (ceiling probes, robustness variants,
interaction experiments), all with TYNDP_FIXES=all.

    python run_phase2.py --queue A      (start next to phase-1 queue A)
    python run_phase2.py --queue B      (start next to phase-1 queue B)
    python run_phase2.py --queue B --dry-run   (print the commands only)

Each queue first WAITS until its phase-1 queue has produced all its outputs
(it checks the files, every 10 min, using no CPU), so it can be started right
away in a third/fourth window and takes over the cores the moment phase 1
ends. It then waits for the phase-1 frontiers it needs, derives the plan with
phase2_plan.py (written to logs/phase2_plan.json, identical for both queues)
and runs its items one by one. Restart after a crash with the same command:
finished items are skipped (output file or log marker present); an
interrupted item starts again from scratch.

Queue A: ceiling probes DE 2040 (weak-node storage x2 and 8-h storage, 4 x ~15 h)
Queue B: robustness baselines, interaction experiments, DE 2050 probes,
         robustness re-solves of the elbow target, 52-week baseline
"""
import os, sys, json, time, argparse, subprocess, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output'); LOG = os.path.join(HERE, 'logs')
S = 'tfall'
PY = sys.executable
CFG = ['--climate-year', '2009', '--demand-2040', '--h2-2040', '--ev-2040',
       '--p2g-endogenous', '--n-weeks', '12']
DIAG = ['--no-monotone', '--no-validate', '--diagnostic-evals', '2000',
        '--n-restarts', '3', '--n-workers', '4']
PHASE1 = {
    'A': [f'directed_search_frontier_5node_vf_de2040_cy2009_dhep_w12_{S}.csv',
          f'directed_search_frontier_5node_vf_de2050_cy2009_dhep_w12_{S}.csv',
          f'directed_search_frontier_5node_vf_nt2030_cy2009_dhep_w12_{S}.csv',
          f'grid_floor_5node_de2040_{S}.csv.done',
          f'directed_search_frontier_5node_vfw_de2050_cy2009_dhep_w12_{S}.csv'],
    'B': [f'directed_search_frontier_5node_vfw_de2040_cy2009_dhep_w12_{S}.csv',
          f'directed_search_frontier_5node_vf_de2035_cy2009_dhep_w12_{S}.csv',
          f'grid_floor_5node_de2035_{S}.csv.done', f'grid_floor_5node_de2050_{S}.csv.done',
          f'directed_search_frontier_5node_vfw_de2035_cy2009_dhep_w12_{S}.csv',
          f'directed_search_frontier_5node_vfw_nt2030_cy2009_dhep_w12_{S}.csv']}
NEEDS = [f'directed_search_frontier_5node_vf_de2040_cy2009_dhep_w12_{S}.csv',
         f'directed_search_frontier_5node_vf_de2050_cy2009_dhep_w12_{S}.csv']
BASE_ENV = {'LP_SOLVER_THREADS': '1', 'GUROBI_METHOD': '2', 'SOLVE_TIME_LIMIT_S': '1800',
            'TYNDP_FIXES': 'all', 'CAP_OVERRIDES': '', 'STO_EP_RATIO': '', 'OFFSHORE_HUBS': '',
            'GRID_STEP_MULT': '', 'H2_OVERRIDES': '', 'QUEUE_NAME': '', 'QUEUE_ITEM': '',
            'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}   # logs are files, not a console


def stamp():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M')


class Q:
    def __init__(self, name, dry):
        self.name, self.dry = name, dry
        self.status = os.path.join(LOG, f'QUEUE_STATUS_{name}2.txt')

    def log(self, m):
        line = f'{stamp()}  [{self.name}2] {m}'
        print(line, flush=True)
        if not self.dry:
            with open(self.status, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

    def notify(self, m):
        if self.dry:
            return
        try:
            from notify import notify
            notify(f'[Phase 2 queue {self.name}] {m}')
        except Exception:
            pass


def wait_for(q, files, what):
    miss = [f for f in files if not os.path.exists(os.path.join(OUT, f))]
    if not miss:
        return
    q.log(f'waiting for {what}: {len(miss)} file(s) missing')
    if q.dry:
        q.log('(dry run: not waiting)'); return
    while miss:
        time.sleep(600)
        miss = [f for f in files if not os.path.exists(os.path.join(OUT, f))]
    q.log(f'{what}: all present')


def ds(sc, *extra):
    return [PY, 'directed_search.py', '--scenario', sc] + CFG + list(extra)


def nf(base, out, sweeps, base_row=None, relative=False):
    a = [PY, 'node_factorial.py', '--scenario', 'de2040'] + CFG
    if base == 'bopt':
        a += ['--base', 'bopt']
    else:
        a += ['--base', 'csv', '--base-csv', os.path.join('output', base), '--base-row', str(base_row)]
    if relative:
        a += ['--relative']
    for k, v in sweeps:
        a += ['--sweep', f'{k}=' + ','.join(f'{x:.0f}' for x in v)]
    return a + ['--out', os.path.join('output', out)]


def items_for(queue, p):
    """(label, command, env, done_check, expected_hours)"""
    fr40 = f'directed_search_frontier_5node_vf_de2040_cy2009_dhep_w12_{S}.csv'
    el = p['de2040']['elbow_target']
    it = []

    def probe(pr):
        return (f"probe {pr['scenario']} {pr['variant']} @ {pr['target']:.3f}",
                ds(pr['scenario'], *DIAG, '--diagnostic-vf', f"{pr['target']:.3f}", '--run-tag', pr['tag']),
                pr['env'], ('file', os.path.join(OUT, f"diagnostic_vf_{pr['target']:.3f}_{pr['tag']}",
                                                 'diagnostic_frontier.csv')), 15)

    if queue == 'A':
        it += [probe(pr) for pr in p['probes'] if pr['scenario'] == 'de2040']
        return it
    # ---- queue B ----
    it.append(('robustness baseline noresid', ds('de2040', '--baseline-only', '--no-resid'), {},
               ('log', f'rob_noresid_base_{S}.log', 'B_opt cost'), 0.1))
    it.append(('robustness baseline widep', ds('de2040', '--baseline-only', '--resid-export-price', '0',
                                               '--resid-import-price', '150'), {},
               ('log', f'rob_widep_base_{S}.log', 'B_opt cost'), 0.1))
    for cy in (2010, 2003, 1995, 2008):
        a = ds('de2040', '--baseline-only'); a[a.index('--climate-year') + 1] = str(cy)
        it.append((f'robustness baseline cy{cy}', a, {}, ('log', f'rob_cy{cy}_base_{S}.log', 'B_opt cost'), 0.1))
    it.append(('robustness baseline dkdom', ds('de2040', '--baseline-only'), {'H2_OVERRIDES': 'DK=5'},
               ('log', f'rob_dkdom_base_{S}.log', 'B_opt cost'), 0.1))
    # interaction experiments
    L2 = p['lens2']; sk, mk, k2 = p['sto_key'], p['main_ntc_key'], p['second_ntc_key']
    sw = [(sk, L2['sto']), (mk, L2['main'])] + ([(k2, L2['second'])] if k2 else [])
    it.append(('interactions lens 2 (node factorial at B_opt)',
               nf('bopt', f'node_factorial_de2040_bopt_{S}.csv', sw), {},
               ('rows', os.path.join(OUT, f'node_factorial_de2040_bopt_{S}.csv'),
                len(L2['sto']) * len(L2['main']) * max(1, len(L2['second']))), 0.5))
    for L in p['lens3']:
        it.append((f"interactions lens 3 row {L['row']}",
                   nf(fr40, f"node_local_de2040_row{L['row']}_{S}.csv", [(sk, L['sto']), (mk, L['main'])],
                      base_row=L['row'], relative=True), {},
                   ('rows', os.path.join(OUT, f"node_local_de2040_row{L['row']}_{S}.csv"), 9), 0.2))
    if p['farside']:
        F = p['farside']
        for r in F['rows']:
            if r is None:
                continue
            nm = f'farside_de2040_bopt_{S}.csv' if r == 'bopt' else f'farside_de2040_row{r}_{S}.csv'
            it.append((f'interactions far side ({r})',
                       nf('bopt' if r == 'bopt' else fr40, nm, [('sto_mw', F['sto']), (mk, F['main'])],
                          base_row=None if r == 'bopt' else r), {},
                       ('rows', os.path.join(OUT, nm), len(F['sto']) * len(F['main'])), 0.3))
    it += [probe(pr) for pr in p['probes'] if pr['scenario'] == 'de2050']
    # robustness re-solves of the elbow target
    if el is not None:
        dg = ['--diagnostic-vf', f'{el:.3f}']
        it.append(('robustness elbow no-resid', ds('de2040', *DIAG, *dg, '--no-resid', '--run-tag', f'rob_noresid_{S}'),
                   {}, ('log', f'rob_noresid_diag_{S}.log', 'Achieved VF_min'), 15))
        for cy in (2003, 2010):
            a = ds('de2040', *DIAG, *dg, '--run-tag', f'rob_cy{cy}_{S}'); a[a.index('--climate-year') + 1] = str(cy)
            it.append((f'robustness elbow cy{cy}', a, {}, ('log', f'rob_cy{cy}_diag_{S}.log', 'Achieved VF_min'), 15))
    a = ds('de2040', '--baseline-only'); a[a.index('--n-weeks') + 1] = '52'
    it.append(('robustness baseline 52 weeks', a, {}, ('log', f'rob_gold52_base_{S}.log', 'B_opt cost'), 2))
    return it


def logname(item):
    chk = item[3]
    if chk[0] == 'log':
        return os.path.join(LOG, chk[1])
    return os.path.join(LOG, 'phase2_' + item[0].replace(' ', '_').replace('(', '').replace(')', '')
                        .replace('@', 'at').replace('/', '_') + '.log')


def done(item):
    chk = item[3]
    if chk[0] == 'file':
        return os.path.exists(chk[1])
    if chk[0] == 'log':
        p = os.path.join(LOG, chk[1])
        return os.path.exists(p) and chk[2] in open(p, encoding='utf-8', errors='ignore').read()
    if chk[0] == 'rows':
        if not os.path.exists(chk[1]):
            return False
        try:
            import pandas as pd
            return len(pd.read_csv(chk[1])) >= chk[2]
        except Exception:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True, choices=['A', 'B'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-wait', action='store_true', help='do not wait for the phase-1 queue')
    ap.add_argument('--plan-base', default=None, help='(testing) folder with the phase-1 frontiers')
    a = ap.parse_args()
    os.chdir(HERE); os.makedirs(LOG, exist_ok=True)
    q = Q(a.queue, a.dry_run)
    q.log('=== run_phase2.py START ===')
    if not a.no_wait:
        wait_for(q, PHASE1[a.queue], f'phase-1 queue {a.queue}')
    wait_for(q, NEEDS, 'the DE 2040 and DE 2050 weakest-link frontiers')
    import phase2_plan
    plan = phase2_plan.make_plan(a.plan_base or OUT)
    if not a.dry_run:
        json.dump(plan, open(os.path.join(LOG, 'phase2_plan.json'), 'w'), indent=1)
    q.log(f"plan: weak node DE2040 {plan['weak_node_de2040']}, elbow {plan['de2040']['elbow_target']}, "
          f"ceiling {plan['de2040']['ceiling']:.3f} (DE2050 weak node {plan['de2050']['weak_node']}, "
          f"ceiling {plan['de2050']['ceiling']:.3f})")
    items = items_for(a.queue, plan)
    todo = [i for i in items if not done(i)]
    exp = sum(i[4] for i in todo)
    q.log(f'{len(items)} items, {len(todo)} to run, expected ~{exp:.0f} h')
    q.notify(f"START: {len(todo)} of {len(items)} items, ~{exp:.0f} h. Weak node {plan['weak_node_de2040']}, "
             f"elbow {plan['de2040']['elbow_target']}, ceiling {plan['de2040']['ceiling']:.3f}")
    for k, item in enumerate(items):
        label, cmd, env, chk, eh = item
        if done(item):
            q.log(f'SKIP  {label} (done)'); continue
        e = dict(os.environ); e.update(BASE_ENV); e.update(env)
        q.log(f'START {label}  env={env or "-"}')
        if a.dry_run:
            print('      ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd[1:])); continue
        t0 = time.time()
        with open(logname(item), 'w', encoding='utf-8') as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=HERE)
        h = (time.time() - t0) / 3600
        ok = done(item)
        exp -= eh
        q.log(f"{'DONE ' if ok else 'FAIL '} {label} ({h:.1f} h, exit {rc})")
        nxt = (dt.datetime.now() + dt.timedelta(hours=max(exp, 0))).strftime('%a %d %b %H:%M')
        q.notify(f"{'DONE' if ok else 'WARNING, no result:'} {label} in {h:.1f} h. "
                 f"Item {k+1}/{len(items)}. Queue done ~ {nxt}")
    q.log('=== run_phase2.py FINISHED ===')
    q.notify('FINISHED')


if __name__ == '__main__':
    main()
