"""Chunk-wise driver for make_paper_numbers.py.

    PAPER_SET=tfall python build_numbers.py        (Windows: set PAPER_SET=tfall)

* splits make_paper_numbers.py at its '# ----' section markers and runs the
  sections one by one in a shared namespace,
* a section whose inputs are missing (file not there yet, frontier not
  finished, judgement pin not set) is skipped and reported, any other error
  prints a traceback and is reported as a BUG,
* before running, every output listed in gen/manifest.json is deleted, so no
  stale table, figure or macro from an earlier result set can survive,
* afterwards every macro, table and figure of the manifest that was not
  produced gets a red placeholder, so the paper always compiles and a missing
  number is impossible to overlook.
The manifest is (re)written by a complete run of the old set:
    PAPER_SET=old python build_numbers.py --write-manifest
"""
import os, re, sys, json, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'make_paper_numbers.py')
GEN = os.path.join(HERE, 'gen')
MAN = os.path.join(GEN, 'manifest.json')
KEEP = {'cf_annual.json', 'manifest.json'}          # inputs, never deleted
EXPECTED = (FileNotFoundError, KeyError)
_MS = ['NtThirty', 'DeThirtyFive', 'DeForty', 'DeFifty']
# sections that cannot run on the old set; their outputs are declared here so
# the manifest still covers them
TFALL_ONLY = {
 'S15': (['tab_benchmark.tex'], [f'bm{k}{m}' for k in ('Vf', 'Prem', 'Sto', 'Ratio') for m in _MS]),
 'S17': (['tab_prices.tex'], []),
}

def chunks(src):
    out, cur, name = [], [], 'preamble'
    for line in src.splitlines(True):
        m = re.match(r'#\s*-{2,}\s*(.*?)\s*-*\s*$', line)
        if m and cur:
            out.append((name, ''.join(cur))); cur = []; name = m.group(1) or 'section'
        cur.append(line)
    out.append((name, ''.join(cur)))
    return out

def macros(path):
    if not os.path.exists(path): return []
    return re.findall(r'\\newcommand\{\\(\w+)\}', open(path).read())

def main():
    write_manifest = '--write-manifest' in sys.argv
    man = json.load(open(MAN)) if os.path.exists(MAN) else None
    if man:
        for f in man['files']:
            if f not in KEEP and os.path.exists(os.path.join(GEN, f)):
                os.remove(os.path.join(GEN, f))
    os.chdir(HERE)
    ns = {'__name__': 'make_paper_numbers_chunked', '__file__': SRC, '_CHUNKED': True}
    report = []
    def _snap():
        return {f: os.path.getmtime(os.path.join(GEN, f)) for f in os.listdir(GEN)}
    status = {}
    for name, code in chunks(open(SRC).read()):
        key = name.split(':')[0]
        need = re.search(r'needs (S\w+)', name)
        if need and status.get(need.group(1)) != 'ok':
            report.append(('SKIP', name, f'depends on {need.group(1)}')); status[key] = 'SKIP'
            continue
        before = _snap()
        try:
            exec(compile(code, f'{SRC}:{name}', 'exec'), ns)
            report.append(('ok', name, '')); status[key] = 'ok'
        except Exception as e:
            status[key] = 'failed'
            # a failed section may leave half-written tables: remove whatever
            # it touched (numbers.tex only gets complete macro lists appended)
            for f, t in _snap().items():
                if f not in KEEP and f != 'numbers.tex' and before.get(f) != t:
                    os.remove(os.path.join(GEN, f))
            pin = ns.get('PinMissing')
            if isinstance(e, EXPECTED) or (pin and isinstance(e, pin)):
                report.append(('SKIP', name, f'{type(e).__name__}: {e}'))
            else:
                traceback.print_exc()
                report.append(('BUG', name, f'{type(e).__name__}: {e}'))
    import matplotlib.pyplot as _plt; _plt.close('all')
    made_files = sorted(f for f in os.listdir(GEN) if f not in KEEP and not f.startswith('_'))
    made_mac = macros(os.path.join(GEN, 'numbers.tex'))
    dup = sorted({m for m in made_mac if made_mac.count(m) > 1})
    if dup:
        report.append(('BUG', 'numbers.tex', f'duplicate macros {dup}'))
    if write_manifest:
        bad = [r for r in report if r[0] != 'ok' and r[1].split(':')[0] not in TFALL_ONLY]
        if bad:
            sys.exit(f'manifest NOT written, sections failed: {bad}')
        for fs, ms in TFALL_ONLY.values():
            made_files = sorted(set(made_files) | set(fs)); made_mac = made_mac + [m for m in ms if m not in made_mac]
        json.dump({'set': ns.get('PAPER_SET'), 'files': made_files, 'macros': made_mac},
                  open(MAN, 'w'), indent=1)
        print(f'manifest: {len(made_files)} files, {len(made_mac)} macros')
    ph_mac, ph_files = [], []
    if man:
        missing_mac = [m for m in man['macros'] if m not in made_mac]
        with open(os.path.join(GEN, 'numbers.tex'), 'a') as f:
            for m in missing_mac:
                f.write('\\newcommand{\\%s}{\\textcolor{red}{\\textbf{??}}}%% placeholder\n' % m)
        ph_mac = missing_mac
        for fn in man['files']:
            p = os.path.join(GEN, fn)
            if os.path.exists(p): continue
            if fn.endswith('.tex'):
                open(p, 'w').write('\\textcolor{red}{\\textbf{[%s pending: rerun results not yet in]}}\n'
                                   % fn.replace('_', '\\_'))
            elif fn.endswith('.pdf'):
                import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
                fig, a = plt.subplots(figsize=(4.6, 2.0)); a.axis('off')
                a.text(0.5, 0.5, f'{fn}\npending: rerun results not yet in', color='red',
                       ha='center', va='center', fontsize=11, transform=a.transAxes)
                fig.savefig(p); plt.close(fig)
            else:
                continue
            ph_files.append(fn)
        extra_mac = [m for m in made_mac if m not in man['macros']]
        if extra_mac: print('new macros (not in manifest):', extra_mac)
    print('\n=== build_numbers report (PAPER_SET=%s) ===' % ns.get('PAPER_SET'))
    for st, name, msg in report:
        print(f'{st:5s} {name[:60]:60s} {msg[:110]}')
    print(f'placeholders: {len(ph_mac)} macros, {len(ph_files)} files {ph_files}')
    if any(r[0] == 'BUG' for r in report):
        sys.exit(1)

if __name__ == '__main__':
    main()
