"""
queue_progress.py -- S.105: queue-aware progress + ETA lines for the Telegram
messages (notify.py) during multi-week runs.

The PowerShell queue writes logs/queue_<Q>_plan.json (ordered items with an
expected number of targets per frontier, or expected hours per grid) and sets
env QUEUE_NAME / QUEUE_ITEM before each item. directed_search.py calls
on_target() after every finished target; the queue script calls
`python queue_progress.py item_done <Q> <idx> [hours]`. Events go to
logs/queue_<Q>_events.jsonl. The ETA uses the mean duration of the targets
finished so far in THIS queue (fallback 20 h/target until the first one ends).
Never raises: any problem returns an empty string so a run is never disturbed.
"""
import os, sys, json, time, datetime as _dt

_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
FALLBACK_TARGET_H = 20.0


def _paths(q):
    return (os.path.join(_LOG, f'queue_{q}_plan.json'),
            os.path.join(_LOG, f'queue_{q}_events.jsonl'))


def _load(q):
    plan_p, ev_p = _paths(q)
    plan = json.load(open(plan_p))
    ev = []
    if os.path.exists(ev_p):
        for line in open(ev_p):
            line = line.strip()
            if line:
                try: ev.append(json.loads(line))
                except Exception: pass
    return plan, ev


def _append(q, rec):
    _, ev_p = _paths(q)
    os.makedirs(_LOG, exist_ok=True)
    with open(ev_p, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def _fmt_eta(hours):
    t = _dt.datetime.now() + _dt.timedelta(hours=hours)
    days = hours / 24.0
    span = f'{hours:.0f} h' if hours < 36 else f'{days:.1f} d'
    return f"{t.strftime('%a %d %b %H:%M')} (in ~{span})"


def summary(q, idx, extra_done_now=False):
    plan, ev = _load(q)
    items = plan['items']
    tdur = [e['dt_h'] for e in ev if e.get('type') == 'target']
    avg = sum(tdur) / len(tdur) if tdur else FALLBACK_TARGET_H
    done_items = {e['item'] for e in ev if e.get('type') == 'item_done'}
    n_t_item = sum(1 for e in ev if e.get('type') == 'target' and e['item'] == idx)

    def rem_hours(i, it):
        if i in done_items:
            return 0.0
        if it['kind'] == 'grid':
            return float(it.get('exp_hours', 1.5))
        k = sum(1 for e in ev if e.get('type') == 'target' and e['item'] == i)
        left = max(it.get('exp_targets', 6) - k, 1 if i == idx else it.get('exp_targets', 6) - k)
        return max(left, 0) * avg

    item_h = rem_hours(idx, items[idx]) if idx < len(items) else 0.0
    queue_h = sum(rem_hours(i, it) for i, it in enumerate(items) if i >= idx)
    tot_h = sum((it.get('exp_hours', 1.5) if it['kind'] == 'grid' else it.get('exp_targets', 6) * avg)
                for it in items)
    pct = 100.0 * max(0.0, 1 - queue_h / tot_h) if tot_h > 0 else 0.0
    exp = items[idx].get('exp_targets', 0) if idx < len(items) else 0
    head = (f"[Queue {q} | run {idx+1}/{len(items)}: {items[idx]['label']}]"
            if idx < len(items) else f"[Queue {q}]")
    tail = (f"progress: queue ~{pct:.0f}% | avg {avg:.1f} h/target\n"
            f"this run done ~ {_fmt_eta(item_h)}\n"
            f"queue done ~ {_fmt_eta(queue_h)}")
    return head, n_t_item, exp, tail


def on_target(dt_seconds):
    """Record a finished target; return (prefix, suffix) for the message."""
    try:
        q = os.environ.get('QUEUE_NAME'); idx = os.environ.get('QUEUE_ITEM')
        if not q or idx is None:
            return '', ''
        idx = int(idx)
        _append(q, {'type': 'target', 'item': idx, 'dt_h': dt_seconds / 3600.0,
                    'at': time.strftime('%Y-%m-%d %H:%M')})
        head, k, exp, tail = summary(q, idx)
        return f"{head} target {k} (~{max(exp, k)} expected)", tail
    except Exception:
        return '', ''


def line(q=None, idx=None):
    try:
        q = q or os.environ.get('QUEUE_NAME'); idx = int(idx if idx is not None else os.environ.get('QUEUE_ITEM', 0))
        if not q:
            return ''
        head, k, exp, tail = summary(q, idx)
        return head + '\n' + tail
    except Exception:
        return ''


if __name__ == '__main__':
    # CLI used by the PowerShell queue:
    #   python queue_progress.py start <Q>          -> notify queue start
    #   python queue_progress.py item_start <Q> <i> -> notify item start
    #   python queue_progress.py item_done <Q> <i> [hours] -> record + notify
    #   python queue_progress.py finished <Q>       -> notify queue end
    from notify import notify
    cmd, q = sys.argv[1], sys.argv[2]
    try:
        if cmd == 'start':
            plan, _ = _load(q)
            notify(f"Queue {q} START: {len(plan['items'])} runs: "
                   + ', '.join(it['label'] for it in plan['items']) + '\n' + line(q, 0))
        elif cmd == 'item_start':
            i = int(sys.argv[3]); notify('START ' + line(q, i))
        elif cmd == 'item_done':
            i = int(sys.argv[3]); h = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
            _append(q, {'type': 'item_done', 'item': i, 'hours': h, 'at': time.strftime('%Y-%m-%d %H:%M')})
            plan, _ = _load(q)
            nxt = line(q, i + 1) if i + 1 < len(plan['items']) else f'Queue {q}: that was the last run.'
            notify(f"DONE run {i+1}/{len(plan['items'])} ({plan['items'][i]['label']}) in {h:.1f} h\n" + nxt)
        elif cmd == 'finished':
            notify(f"Queue {q} FINISHED")
    except Exception as e:
        print('queue_progress:', e)
