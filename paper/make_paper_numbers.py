"""Generate ALL numbers, tables and figures for the paper from output CSVs.
Rule: no hand-copied numbers in paper.tex -- rerun me after any new run.
Outputs: gen/numbers.tex, gen/tab_*.tex, gen/fig_*.pdf

Run it through build_numbers.py (chunk-wise, placeholders for anything whose
input is not there yet). Direct `python make_paper_numbers.py` still works but
stops at the first missing input.

PAPER_SET (env) selects the result set:
  old   -- results of the first submission draft (default)
  tfall -- the TYNDP_FIXES=all rerun (files carry the suffix _tfall)
PAPER_ALLOW_CHECKPOINT=1 lets an unfinished frontier be read from its
checkpoint file (preview only, never for the final build).
Judgement values that are not mechanical functions of the data (which
frontier point is called the elbow, probe targets) live in PINS below. An
empty pin makes the dependent chunk fail on purpose, so build_numbers.py
writes a red placeholder instead of a stale number."""
import pandas as pd, numpy as np, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, 'gen')
os.makedirs(OUT, exist_ok=True)
_PV = os.environ.get('PAPER_PYDIR') or os.path.join(_HERE, '..', 'model')
_PV = os.path.abspath(_PV)
BASE = os.path.join(_PV, 'output')
LOGD = os.path.join(_PV, 'logs')
PAPER_SET = os.environ.get('PAPER_SET', 'old')
assert PAPER_SET in ('old', 'tfall'), PAPER_SET
ALLOW_CKPT = os.environ.get('PAPER_ALLOW_CHECKPOINT') == '1'

class PinMissing(Exception):
    """a judgement value for this paper set has not been fixed yet"""

# Old-set values are the ones the first draft used. tfall: fill in after
# looking at the new frontiers (elbow = last cheap point before the marginal
# mitigation cost turns steep, see the printout of build_numbers.py).
PINS = {
 'old': {'elbow_target_de2040': 0.738,
         'probes': [
  ('DE 2040','NL storage $\\times$2','0.790','diagnostic_vf_0.790/diagnostic_frontier.csv','ProbeSevenNine'),
  ('DE 2040','NL storage $\\times$2','0.820','diagnostic_vf_0.820/diagnostic_frontier.csv','ProbeEightTwo'),
  ('DE 2040','8-h storage','0.790','diagnostic_vf_0.790_ep8_de2040_079/diagnostic_frontier.csv','EpSevenNine'),
  ('DE 2040','8-h storage','0.820','diagnostic_vf_0.820_ep8_de2040_082/diagnostic_frontier.csv','EpEightTwo'),
  ('DE 2050','NL storage $\\times$2','0.720','diagnostic_vf_0.720/diagnostic_frontier.csv','ProbeSevenTwo'),
  ('DE 2050','NL storage $\\times$2','0.800','diagnostic_vf_0.800_nlx2_de2050_080/diagnostic_frontier.csv','FiftyEightZero')],
         'local_rows': 'auto'},
 'tfall': {'elbow_target_de2040': 'auto', 'probes': 'auto', 'local_rows': 'auto'},
}
def PIN(k):
    v = PINS[PAPER_SET].get(k)
    if k == 'elbow_target_de2040' and os.environ.get('PAPER_PIN_ELBOW'):
        v = float(os.environ['PAPER_PIN_ELBOW'])   # trial value from the shell
    if v == 'auto' and k == 'probes':
        v = PLAN()['probes']
    if v is None:
        raise PinMissing(f"PINS['{PAPER_SET}']['{k}'] not set")
    return v

# ---- phase-2 design (weak node, links, probes). Old set: the first campaign's
# design (Dutch node). tfall: logs/phase2_plan.json written by run_phase2.py,
# whose rules (phase2_plan.py) also define the elbow ('auto' pins).
ADJ = {'DE':'German','CZ':'Czech','DK':'Danish','AT':'Austrian','NL':'Dutch'}
DSM_POT = {'DE':16000.0,'CZ':2000.0,'DK':2000.0,'AT':1600.0,'NL':3000.0}
NTC_TOTAL_MAX = {'de_cz':5600.0,'de_dk':9100.0,'de_at':11500.0,'de_nl':8000.0,'dk_nl':2700.0,'cz_at':2400.0}
_PLAN_CACHE = {}
def PLAN():
    if 'p' in _PLAN_CACHE: return _PLAN_CACHE['p']
    if PAPER_SET == 'old':
        p = {'W':'NL','sto_key':'sto_nl_mw','main_ntc_key':'ntc_de_nl_mw','second_ntc_key':'ntc_dk_nl_mw',
             'sto_pot':40000.0,'main_max':3000.0,'far_sto_top':80000.0,'far_sto_mid':40000.0,
             'probes':PINS['old']['probes']}
    else:
        f = os.path.join(LOGD, 'phase2_plan.json')
        if not os.path.exists(f):
            raise PinMissing('logs/phase2_plan.json not found (written by run_phase2.py)')
        j = json.load(open(f))
        W = j['weak_node_de2040']
        names = {('de2040','w2x','a'):'ProbeSevenNine',('de2040','w2x','b'):'ProbeEightTwo',
                 ('de2040','ep8','a'):'EpSevenNine',('de2040','ep8','b'):'EpEightTwo',
                 ('de2050','w2x','a'):'ProbeSevenTwo',('de2050','w2x','b'):'FiftyEightZero'}
        pr = []
        for q in j['probes']:
            kind = q['tag'].split('_')[0]; ab = q['tag'].split('_')[2]
            var = (q['variant'].replace(' x2', ' $\\times$2'))
            pr.append(({'de2040':'DE 2040','de2050':'DE 2050'}[q['scenario']], var, f"{q['target']:.3f}",
                       f"diagnostic_vf_{q['target']:.3f}_{q['tag']}/diagnostic_frontier.csv",
                       names[(q['scenario'], kind, ab)]))
        p = {'W':W,'sto_key':j['sto_key'],'main_ntc_key':j['main_ntc_key'],'second_ntc_key':j['second_ntc_key'],
             'sto_pot':max(j['lens2']['sto']),'main_max':max(j['lens2']['main']),
             'far_sto_top':max(j['farside']['sto']) if j['farside'] else None,
             'far_sto_mid':sorted(j['farside']['sto'])[-2] if j['farside'] else None,
             'probes':pr}
    _PLAN_CACHE['p'] = p
    return p

def P(name):
    """Path of an output file for the active paper set. tfall inserts
    _tfall before the extension (convention of the phase-1/2 run scripts)."""
    if PAPER_SET == 'old':
        return os.path.join(BASE, name)
    special = {'grid_floor_5node_matched.csv': 'grid_floor_5node_de2040_tfall.csv'}
    if name in special:
        return os.path.join(BASE, special[name])
    stem, ext = os.path.splitext(name)
    return os.path.join(BASE, f'{stem}_{PAPER_SET}{ext}')
def PL(name):
    """same for run logs"""
    if PAPER_SET == 'old':
        return os.path.join(LOGD, name)
    stem, ext = os.path.splitext(name)
    return os.path.join(LOGD, f'{stem}_{PAPER_SET}{ext}')

F_OLD = {
 ('nt2030','vf') : 'directed_search_frontier_5node_vf_nt2030_cy2009_dhep_w12.csv',
 ('nt2030','vfw'): 'directed_search_frontier_5node_vfw_nt2030_cy2009_dhep_w12.csv',
 ('de2035','vf') : 'directed_search_frontier_5node_vf_de2035_cy2009_dhep_w12.csv',
 ('de2035','vfw'): 'directed_search_frontier_5node_vfw_de2035_cy2009_dhep_w12.csv',
 ('de2040','vf') : 'checkpoint_frontier_5node_vf_de2040_cy2009_dhep_w12.csv',
 ('de2040','vfw'): 'directed_search_frontier_5node_vfw_de2040_cy2009_dhep_w12.csv',
 ('de2050','vf') : 'directed_search_frontier_5node_vf_de2050_cy2009_dhep_w12.csv',
 ('de2050','vfw'): 'directed_search_frontier_5node_vfw_de2050_cy2009_dhep_w12.csv',
}
def _frontier_path(sc, tg):
    if PAPER_SET == 'old':
        return os.path.join(BASE, F_OLD[(sc, tg)])
    fin = os.path.join(BASE, f'directed_search_frontier_5node_{tg}_{sc}_cy2009_dhep_w12_{PAPER_SET}.csv')
    ck = os.path.join(BASE, f'checkpoint_frontier_5node_{tg}_{sc}_cy2009_dhep_w12_{PAPER_SET}.csv')
    if os.path.exists(fin) or not ALLOW_CKPT:
        return fin
    return ck
D = {}; D_MISSING = []
for _k in F_OLD:
    try:
        D[_k] = pd.read_csv(_frontier_path(*_k))
    except FileNotFoundError:
        D_MISSING.append(_k)
if D_MISSING and __name__ == '__main__' and not globals().get('_CHUNKED'):
    raise FileNotFoundError(f'frontiers missing: {D_MISSING}')
open(os.path.join(OUT, 'numbers.tex'), 'w').close()
SC = ['nt2030','de2035','de2040','de2050']
LBL = {'nt2030':'NT 2030','de2035':'DE 2035','de2040':'DE 2040','de2050':'DE 2050'}
MAC = {'nt2030':'NtThirty','de2035':'DeThirtyFive','de2040':'DeForty','de2050':'DeFifty'}
NODES=['DE','CZ','DK','AT','NL']; TECH=['pv','wind_on','wind_off']
_NW={1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}
def NW(n): return _NW.get(int(n), str(int(n)))

def hits(d, m):
    """interior rows that achieved their target"""
    out=[]
    for i,r in d.iterrows():
        if i==0: continue
        if r[m] >= r['vf_target']-0.003: out.append(r)
    return out
def misses(d, m):
    return [r for i,r in d.iterrows() if i>0 and r[m] < r['vf_target']-0.003]

mac = []
def M(name,val,fmt='%.3f'):
    mac.append(r'\newcommand{\%s}{%s}' % (name, fmt%val if isinstance(val,(int,float,np.floating)) else val))

# ---------- scalar macros ----------
def _gridmax(sc, col):
    """highest value of col in the scenario's factorial grid (all-in corner);
    None for the old set (the search's best point was above it there) or if no grid"""
    if PAPER_SET == 'old':
        return None
    f = os.path.join(BASE, f'grid_floor_5node_{sc}_{PAPER_SET}.csv')
    if not os.path.exists(f):
        return None
    g = pd.read_csv(f); g = g[g.status == 'ok']
    return float(g[col].max()) if len(g) else None
def _ceil(sc, d, m):
    """ceiling = highest value reached by the search (attained or not) or by the grid"""
    v = float(d[m].max()); gm = _gridmax(sc, m)
    return max(v, gm) if gm is not None else v
def _elbow_row(d):
    """frontier row of the pinned elbow target (de2040 weakest-link)"""
    t = PIN('elbow_target_de2040')
    if t == 'auto':                       # Kneedle rule, same code as phase 2
        import sys as _s; _s.path.insert(0, _PV)
        from phase2_plan import kneedle
        i = kneedle(d)
        if i is None: raise PinMissing('elbow: fewer than two attained points')
        return i, d.loc[i]
    i = (d['vf_target'] - t).abs().idxmin()
    assert abs(d.loc[i, 'vf_target'] - t) < 1e-3, (t, d.loc[i, 'vf_target'])
    return i, d.loc[i]
for sc in SC:
    if (sc,'vf') not in D or (sc,'vfw') not in D:
        print('scalar macros: skip', sc); continue
    b = D[(sc,'vf')].iloc[0]
    m0 = MAC[sc]
    M('baseVf'+m0,      b['vf_min'])
    M('baseVfw'+m0,     b['vf_genw'])
    M('baseCost'+m0,    b['sys_cost']/1e9, '%.1f')
    M('baseCr'+m0,      b['cr_min'],'%.2f')
    M('baseLoss'+m0,    b['cannib_loss_bn'],'%.1f')
    d=D[(sc,'vf')]; H=hits(d,'vf_min'); X=misses(d,'vf_min')
    M('lastHitVf'+m0,   H[-1]['vf_min'])
    M('lastHitPrem'+m0, H[-1]['premium_bn'],'%.2f')
    M('ceilVf'+m0,      _ceil(sc, d, 'vf_min'))
    M('headroomVf'+m0,  H[-1]['vf_min']-b['vf_min'])
    dw=D[(sc,'vfw')]; Hw=hits(dw,'vf_genw'); Xw=misses(dw,'vf_genw')
    M('lastHitVfw'+m0,  Hw[-1]['vf_genw'])
    M('lastHitPremW'+m0,Hw[-1]['premium_bn'],'%.2f')
    M('ceilVfw'+m0,     _ceil(sc, dw, 'vf_genw'))
# headline extras (de2040)
d=D[('de2040','vf')]
M('lossAtLastHitDeForty', hits(d,'vf_min')[-1]['cannib_loss_bn'],'%.1f')
# marginal cost bottom/top (de2040 interior)
hh=[d.iloc[0]]+hits(d,'vf_min')
mc=[(hh[i+1]['premium_bn']-hh[i]['premium_bn'])/(hh[i+1]['vf_min']-hh[i]['vf_min'])*0.01 for i in range(len(hh)-1)]
M('mcBottomDeForty', mc[0],'%.2f'); M('mcTopDeForty', mc[-1],'%.2f')
M('mcEscalation', mc[-1]/mc[0],'%.0f')
# vf/vfw asymmetry (de2040): the last attained point of the fleet-average
# (vfw) frontier lifts VF_min to some level v*. Compare its premium with the
# cheapest weakest-link frontier point that reaches v* (within the 0.003
# tolerance used everywhere). Old set: v*=0.7369 -> vf row 0.738, 0.744 vs 3.30.
_lw = hits(D[('de2040','vfw')],'vf_genw')[-1]
_vs = _lw['vf_min']
_dir = [r for r in [d.iloc[0]]+hits(d,'vf_min') if r['vf_min'] >= _vs-0.003]
if not _dir:
    raise ValueError('asymmetry: no vf frontier point reaches the vfw end level %.3f' % _vs)
M('asymVfLevel', _vs); M('asymDirect', _dir[0]['premium_bn'],'%.2f')
M('asymViaVfw', _lw['premium_bn'],'%.2f'); M('asymFactor', _lw['premium_bn']/_dir[0]['premium_bn'],'%.1f')
# the asymmetry is not uniform: smallest ratio over all attained fleet-average
# points of all scenarios, direct cost by linear interpolation on the
# weakest-link frontier (rerun only)
if PAPER_SET != 'old':
    _am=[]
    for s_ in SC:
        if (s_,'vf') not in D or (s_,'vfw') not in D: continue
        _hv=[D[(s_,'vf')].iloc[0]]+hits(D[(s_,'vf')],'vf_min'); _xs=[r['vf_min'] for r in _hv]; _ys=[r['premium_bn'] for r in _hv]
        for r in hits(D[(s_,'vfw')],'vf_genw'):
            if _xs[0] < r['vf_min'] <= _xs[-1]:
                _am.append((r['premium_bn']/float(np.interp(r['vf_min'],_xs,_ys)), s_, r))
    if _am:
        _f,_s,_r=min(_am, key=lambda z:z[0])
        M('asymMinFactor', _f, '%.2f'); M('asymMinSc', LBL[_s].replace(' ','~'), '%s')
        M('asymMinVf', _r['vf_min']); M('asymMinPremW', _r['premium_bn'], '%.2f')
        M('asymMinDirect', _r['premium_bn']/_f, '%.2f')
# de2050 window collapse
M('headroomDeFortyPct', 100*(hits(D[('de2040','vf')],'vf_min')[-1]['vf_min']-D[('de2040','vf')].iloc[0]['vf_min']),'%.1f')
M('headroomDeFiftyPct', 100*(hits(D[('de2050','vf')],'vf_min')[-1]['vf_min']-D[('de2050','vf')].iloc[0]['vf_min']),'%.1f')
M('baseStoNlDeFifty', D[('de2050','vf')].iloc[0]['sto_nl_mw']/1000,'%.0f')
M('baseStoNlDeForty', D[('de2040','vf')].iloc[0]['sto_nl_mw']/1000,'%.1f')
# extra prose macros (S2)
d40=D[('de2040','vf')]; h40=hits(d40,'vf_min'); m40=misses(d40,'vf_min')
M('mcCeilDeForty', (m40[0]['premium_bn']-h40[-1]['premium_bn'])/(m40[0]['vf_min']-h40[-1]['vf_min'])*0.01,'%.1f')
def stot(r): return (r['sto_mw']+r['sto_cz_mw']+r['sto_dk_mw']+r['sto_at_mw']+r['sto_nl_mw'])/1000
M('stoBaseDeForty', stot(d40.iloc[0]),'%.0f')
M('stoLastDeForty', stot(h40[-1]),'%.0f')
M('stoNlFirstDeForty', h40[0]['sto_nl_mw']/1000,'%.1f')
M('stoDeFirstDeForty', h40[0]['sto_mw']/1000,'%.1f')
M('premPctCost', 100*h40[-1]['premium_bn']/(d40.iloc[0]['sys_cost']/1e9),'%.1f')
M('lossCutPct', 100*(1-h40[-1]['cannib_loss_bn']/d40.iloc[0]['cannib_loss_bn']),'%.0f')
M('ceilAbsVfDeForty', _ceil('de2040', d40, 'vf_min'))
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac)+'\n')
# help for choosing the elbow pin: marginal mitigation cost per segment
print('de2040 vf frontier segments (target, VF_min, premium, marginal cost per +0.01):')
for _i in range(len(hh)-1):
    print('   %.3f  %.4f  %6.3f  %7.3f' % (hh[_i+1]['vf_target'], hh[_i+1]['vf_min'], hh[_i+1]['premium_bn'], mc[_i]))

# ---------- S1c: cross-scenario summary macros (rerun audit, B items) ----------
mac=[]
_ADJ={'DE':'German','CZ':'Czech','DK':'Danish','AT':'Austrian','NL':'Dutch'}
_TL={'pv':'solar','wind_on':'onshore wind','wind_off':'offshore wind'}
def _bind(r):
    best=None
    for n in NODES:
        for t_ in TECH:
            v_=r.get(f'vf_{t_}_{n}',0)
            if v_ and v_>0 and (best is None or v_<best[0]): best=(v_,t_,n)
    return best
for sc in SC:
    if (sc,'vf') not in D: continue
    v_,t_,n_=_bind(D[(sc,'vf')].iloc[0]); m0=MAC[sc]
    M('bindNode'+m0, n_, '%s'); M('bindAdj'+m0, _ADJ[n_], '%s'); M('bindTech'+m0, _TL[t_], '%s')
    _e=hits(D[(sc,'vf')],'vf_min')[-1]; v2,t2,n2=_bind(_e)
    M('bindAdjEnd'+m0, _ADJ[n2], '%s'); M('bindTechEnd'+m0, _TL[t2], '%s')
if all((sc,tg) in D for sc in SC for tg in ('vf','vfw')):
    _cv=[]; _cw=[]; _gaps=[]
    for sc in SC:
        d=D[(sc,'vf')]; X=misses(d,'vf_min'); H=hits(d,'vf_min')
        _cv.append(_ceil(sc, d, 'vf_min'))
        dw=D[(sc,'vfw')]; Xw=misses(dw,'vf_genw'); Hw=hits(dw,'vf_genw')
        _cw.append(_ceil(sc, dw, 'vf_genw'))
        for tg,mm in (('vf','vf_min'),('vfw','vf_genw')):
            for r in hits(D[(sc,tg)],mm):
                g_=r.get('restart_gap_pct',np.nan)
                if g_ is not None and np.isfinite(g_): _gaps.append(g_)
    M('ceilVfLo', min(_cv)); M('ceilVfHi', max(_cv)); M('ceilVfwLo', min(_cw)); M('ceilVfwHi', max(_cw))
    if _gaps: M('restartGapMaxPct', max(_gaps), '%.1f' if PAPER_SET == 'old' else '%.2f')
d40=D[('de2040','vf')]; h40=hits(d40,'vf_min'); m40=misses(d40,'vf_min')
_hh=[d40.iloc[0]]+h40
_mc=[(_hh[i+1]['premium_bn']-_hh[i]['premium_bn'])/(_hh[i+1]['vf_min']-_hh[i]['vf_min'])*0.01 for i in range(len(_hh)-1)]
if m40:
    _mcc=(m40[0]['premium_bn']-h40[-1]['premium_bn'])/(m40[0]['vf_min']-h40[-1]['vf_min'])*0.01
    M('mcCeilEscalation', _mcc/_mc[0], '%.0f')
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac)+'\n')

# ---------- S1d: narrative macros for the rerun text ----------
mac=[]
import sys as _sy; _sy.path.insert(0, _PV)
from scenario_data import get_scenario
STOK=['sto_mw','sto_cz_mw','sto_dk_mw','sto_at_mw','sto_nl_mw']
NTCK=['ntc_de_cz_mw','ntc_de_dk_mw','ntc_de_at_mw','ntc_de_nl_mw','ntc_dk_nl_mw','ntc_cz_at_mw']
d40=D[('de2040','vf')]; h40=hits(d40,'vf_min'); b40=d40.iloc[0]
M('baseVfDeFortyR', b40['vf_min'], '%.2f'); M('lastHitVfDeFortyR', h40[-1]['vf_min'], '%.2f')
M('lastHitPremDeFortyR', h40[-1]['premium_bn'], '%.1f')
_hh=[b40]+h40
_seg=[(_hh[i+1]['premium_bn']-_hh[i]['premium_bn'])/(_hh[i+1]['vf_min']-_hh[i]['vf_min'])*0.01 for i in range(len(_hh)-1)]
M('mcPeakDeForty', max(_seg), '%.2f'); M('mcPeakEscalation', max(_seg)/_seg[0], '%.0f')
M('firstStepDeForty', h40[0]['vf_min']-b40['vf_min']); M('premFirstDeForty', h40[0]['premium_bn'], '%.2f')
for _n,_k in (('Dk','sto_dk_mw'),('Nl','sto_nl_mw'),('De','sto_mw')):
    M(f'stoFirst{_n}DeForty', h40[0][_k]/1e3, '%.1f')
_bs={sc: D[(sc,'vf')].iloc[0] for sc in SC}
_lo=min(SC,key=lambda s_: _bs[s_]['vf_min']); _hi=max(SC,key=lambda s_: _bs[s_]['vf_min'])
M('baseVfLoAll', _bs[_lo]['vf_min']); M('baseVfHiAll', _bs[_hi]['vf_min'])
M('baseVfLoSc', LBL[_lo].replace(' ','~'), '%s'); M('baseVfHiSc', LBL[_hi].replace(' ','~'), '%s')
_wv=[_bs[s_][f'vf_{t_}_{n_}'] for s_ in SC for n_ in NODES for t_ in ('wind_on','wind_off') if _bs[s_].get(f'vf_{t_}_{n_}',0)>0]
M('windVfLo', min(_wv), '%.2f'); M('windVfHi', max(_wv), '%.2f')
_dsp=[_bs[s_]['vf_genw']-_bs[s_]['vf_min'] for s_ in SC]
M('dispBaseLo', min(_dsp), '%.2f'); M('dispBaseHi', max(_dsp), '%.2f')
M('vfDiscountLoPct', 100*(1-_bs[_hi]['vf_min']), '%.0f'); M('vfDiscountHiPct', 100*(1-_bs[_lo]['vf_min']), '%.0f')
_cre=[]; _pct=[]
for sc in SC:
    d=D[(sc,'vf')]; H=hits(d,'vf_min'); m0=MAC[sc]
    M('crEnd'+m0, H[-1]['cr_min'], '%.2f'); _cre.append(H[-1]['cr_min'])
    M('stoBaseTot'+m0, sum(d.iloc[0][k] for k in STOK)/1e3, '%.0f')
    M('ntcAddMax'+m0, max(sum(r[k] for k in NTCK) for r in [d.iloc[0]]+H)/1e3, '%.1f')
    M('premPctEnd'+m0, 100*H[-1]['premium_bn']/(d.iloc[0]['sys_cost']/1e9), '%.1f'); _pct.append(100*H[-1]['premium_bn']/(d.iloc[0]['sys_cost']/1e9))
    if 'subsidy_bill_bn' in d:
        M('subsBillEnd'+m0, H[-1]['subsidy_bill_bn'], '%.2f')
        M('nPosNetThirty'+m0, sum(1 for r in H if r['net_benefit_mcf30_bn']>0), '%d')
M('crEndLo', min(_cre), '%.2f'); M('crEndHi', max(_cre), '%.2f')
M('premPctEndLo', min(_pct), '%.1f'); M('premPctEndHi', max(_pct), '%.1f')
_crg=[]
for sc in ('de2035','de2040','de2050'):
    if PAPER_SET=='old': break
    _f=os.path.join(BASE, f'grid_floor_5node_{sc}_{PAPER_SET}.csv')
    if os.path.exists(_f):
        _g=pd.read_csv(_f); _g=_g[_g.status=='ok']; _crg.append(float(_g.loc[_g.vf_min.idxmax(),'cr_min']))
if _crg: M('crGridHi', max(_crg), '%.3f')
# fleet-average vs weakest-link allocation (DE 2040)
_dw=D[('de2040','vfw')]; _hw=hits(_dw,'vf_genw')
M('vfwFirstStoDe', _hw[0]['sto_mw']/1e3, '%.0f'); M('vfwFirstStoNl', _hw[0]['sto_nl_mw']/1e3, '%.0f')
M('avgByProductGw', h40[-1]['vf_genw'])
_xw=[_dw.iloc[0]['vf_genw']]+[r['vf_genw'] for r in _hw]; _yw=[0.0]+[r['premium_bn'] for r in _hw]
_g0=h40[-1]['vf_genw']
if _g0 <= _xw[-1]: M('avgDirectCost', float(np.interp(_g0,_xw,_yw)), '%.2f')
# asymmetry in DE 2035 (same rule as DE 2040: cheapest attained vf point reaching the vfw end level)
try:
    _d35=D[('de2035','vf')]; _w35=hits(D[('de2035','vfw')],'vf_genw')[-1]
    _r35=[r for r in [_d35.iloc[0]]+hits(_d35,'vf_min') if r['vf_min']>=_w35['vf_min']-0.003]
    if _r35: M('asymFactorThirtyFive', _w35['premium_bn']/_r35[0]['premium_bn'], '%.1f')
except (KeyError, IndexError): pass
# ceiling evidence (DE 2040)
_mis=misses(d40,'vf_min')
if _mis:
    M('nMissDeForty', len(_mis), '%d'); M('missVfLoDeForty', min(r['vf_min'] for r in _mis)); M('missVfHiDeForty', max(r['vf_min'] for r in _mis))
_gm=_gridmax('de2040','vf_min')
if _gm is not None: M('ceilGridGapPts', 100*(_gm-float(d40['vf_min'].max())), '%.1f')
# in how many scenarios the grid corner sets the ceiling, and by how much at most
_gg=[(_gridmax(s_,'vf_min'), float(D[(s_,'vf')]['vf_min'].max())) for s_ in SC if (s_,'vf') in D]
_gg=[g_-s_ for g_,s_ in _gg if g_ is not None]
if _gg:
    M('nCeilGrid', sum(1 for x_ in _gg if x_>0), '%d'); M('nCeilGridOf', len(_gg), '%d')
    M('nCeilGridW', NW(sum(1 for x_ in _gg if x_>0)), '%s'); M('nCeilGridOfW', NW(len(_gg)), '%s')
    M('ceilGridGapMaxPts', 100*max(_gg), '%.1f')
# fleet-average frontiers: distance of the search's best point to the grid corner,
# and grid points that reach the first target the search missed
_vg=[]; _vr=[]
for s_ in SC:
    if (s_,'vfw') not in D: continue
    gm_=_gridmax(s_,'vf_genw')
    if gm_ is None: continue
    d_=D[(s_,'vfw')]; gap_=100*(gm_-float(d_['vf_genw'].max())); _vg.append(gap_)
    M('ceilVfwGap'+MAC[s_]+'Pts', gap_, '%.1f')
    m_=misses(d_,'vf_genw')
    if m_:
        ft_=min(r['vf_target'] for r in m_)
        g_=pd.read_csv(os.path.join(BASE, f'grid_floor_5node_{s_}_{PAPER_SET}.csv')); g_=g_[g_.status=='ok']
        g_=g_[g_['vf_genw']>=ft_-0.003]
        if len(g_): _vr.append(float(g_['premium_bn'].min()))
if _vg:
    M('ceilVfwGapLoPts', min(_vg), '%.1f'); M('ceilVfwGapHiPts', max(_vg), '%.1f')
    M('nVfwMissReach', len(_vr), '%d'); M('nVfwMissReachW', NW(len(_vr)), '%s')
    if _vr: M('vfwMissReachPremLo', min(_vr), '%.1f'); M('vfwMissReachPremHi', max(_vr), '%.1f')
# saturation of Dutch storage and the German-Dutch link at each scenario's best point
_sat=[]; _lnk=[]
for sc in SC:
    d=D[(sc,'vf')]; r=d.loc[d['vf_min'].idxmax()]
    _pot=50000.0 if (sc=='de2050' and PAPER_SET!='old') else 40000.0
    _sat.append(100*r['sto_nl_mw']/_pot)
    if sc!='nt2030': _lnk.append(100*r['ntc_de_nl_mw']/3000.0)
M('satNlStoLoPct', min(_sat), '%.0f'); M('satDeNlLinkLoPct', min(_lnk), '%.0f')
# rounded ranges for prose
_cv2=[_ceil(sc, D[(sc,'vf')], 'vf_min') for sc in SC]; _cw2=[_ceil(sc, D[(sc,'vfw')], 'vf_genw') for sc in SC]
M('ceilVfLoR', min(_cv2), '%.2f'); M('ceilVfHiR', max(_cv2), '%.2f')
M('ceilVfwLoR', min(_cw2), '%.2f'); M('ceilVfwHiR', max(_cw2), '%.2f')
M('discCeilLoPct', 100*(1-max(_cv2)), '%.0f'); M('discCeilHiPct', 100*(1-min(_cv2)), '%.0f')
M('discFleetPct', 100*(1-max(_cw2)), '%.0f'); M('discFleetHiPct', 100*(1-min(_cw2)), '%.0f')
_crb=[_bs[s_]['cr_min'] for s_ in SC]
M('baseCrLoPct', 100*min(_crb), '%.0f'); M('baseCrHiPct', 100*max(_crb), '%.0f')
_hr=[hits(D[(s_,'vf')],'vf_min')[-1]['vf_min']-_bs[s_]['vf_min'] for s_ in SC]
M('headroomLoPts', 100*min(_hr), '%.0f'); M('headroomHiPts', 100*max(_hr), '%.0f')
M('nHitVfwDeForty', len(_hw), '%d')
for sc in SC:
    if 'base_price_DK' in _bs[sc]: M('priceDk'+MAC[sc], _bs[sc]['base_price_DK'], '%.0f')
M('crEndLoPct', 100*min(_cre), '%.0f'); M('crEndHiPct', 100*max(_cre), '%.0f')
M('nlPvBaseDeForty', b40['vf_pv_NL']); M('searchBestDeFifty', float(D[('de2050','vf')]['vf_min'].max()))
if 'subsidy_bill_bn' in b40:
    _sb=[_bs[s_]['subsidy_bill_bn'] for s_ in SC]; _se=[hits(D[(s_,'vf')],'vf_min')[-1]['subsidy_bill_bn'] for s_ in SC]
    M('subsBillLo', min(_sb), '%.1f'); M('subsBillHi', max(_sb), '%.1f'); M('subsBillEndLo', min(_se), '%.2f'); M('subsBillEndHi', max(_se), '%.2f')
    _n0=[sum(1 for r in hits(D[(s_,'vf')],'vf_min') if r['net_benefit_mcf0_bn']>0) for s_ in SC]
    _n3=[sum(1 for r in hits(D[(s_,'vf')],'vf_min') if r['net_benefit_mcf30_bn']>0) for s_ in SC]
    M('nPosNetLo', min(_n0), '%d'); M('nPosNetHi', max(_n0), '%d'); M('nPosNetThirtyLo', min(_n3), '%d'); M('nPosNetThirtyHi', max(_n3), '%d')
    _bst=[max(hits(D[(s_,'vf')],'vf_min'), key=lambda r: r['net_benefit_mcf30_bn']) for s_ in SC]
    M('netBestLo', min(r['net_benefit_mcf30_bn'] for r in _bst), '%.1f'); M('netBestHi', max(r['net_benefit_mcf30_bn'] for r in _bst), '%.1f')
    M('netBestVfLo', min(r['vf_min'] for r in _bst), '%.2f'); M('netBestVfHi', max(r['vf_min'] for r in _bst), '%.2f')
# cost recovery of the lowest-value-factor (binding) asset vs the lowest cost recovery of any asset
_TLB={'pv':'solar','wind_on':'onshore wind','wind_off':'offshore wind'}
_cbb=[]; _cbe=[]
for sc in SC:
    d=D[(sc,'vf')]; m0=MAC[sc]
    for _lab,_r in (('Base',d.iloc[0]),('End',hits(d,'vf_min')[-1])):
        _b=min(((_r[f'vf_{t_}_{n_}'],t_,n_) for n_ in NODES for t_ in TECH if _r.get(f'vf_{t_}_{n_}',0)>0))
        _c=_r.get(f'cr_{_b[1]}_{_b[2]}', np.nan)
        M('crBind'+_lab+m0, _c, '%.2f'); (_cbb if _lab=='Base' else _cbe).append(_c)
        _crs={(t_,n_): _r[f'cr_{t_}_{n_}'] for n_ in NODES for t_ in TECH if _r.get(f'vf_{t_}_{n_}',0)>0 and f'cr_{t_}_{n_}' in _r}
        _lw=min(_crs, key=_crs.get)
        M('crLowAsset'+_lab+m0, f"{ADJ[_lw[1]]} {_TLB[_lw[0]]}", '%s')
M('crBindEndLo', min(_cbe), '%.2f'); M('crBindEndHi', max(_cbe), '%.2f')
# attempts beyond the last attained DE 2040 point: cost and gain increments
_l40=h40[-1]; _ms=misses(d40,'vf_min')
if _ms:
    M('missPremAddLo', min(r['premium_bn']-_l40['premium_bn'] for r in _ms), '%.1f')
    M('missPremAddHi', max(r['premium_bn']-_l40['premium_bn'] for r in _ms), '%.1f')
    M('missGainHiPts', 100*max(r['vf_min']-_l40['vf_min'] for r in _ms), '%.1f')
M('segLastDropPct', 100*(1-_seg[-1]/_seg[-2]) if len(_seg)>1 else 0.0, '%.0f')
# Danish PV capacity by scenario (TYNDP input)
for sc in SC:
    M('pvDk'+MAC[sc], get_scenario(sc)['vre']['DK']['pv']/1e3, '%.0f')
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac)+'\n')

# ---------- S1b: elbow-dependent scalar macros ----------
mac=[]
_ie, _re = _elbow_row(D[('de2040','vf')])
M('elbowTargetDeForty', _re['vf_target'])
M('elbowVfDeForty', _re['vf_min'])
M('premElbowDeForty', _re['premium_bn'], '%.2f')
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac)+'\n')

# ---------- T2 baselines ----------
rows=[]
for sc in SC:
    if (sc,'vf') not in D: raise KeyError(('tab_baselines needs', sc))
    b=D[(sc,'vf')].iloc[0]
    # binding pair
    best=None
    for n in NODES:
        for t in TECH:
            v=b.get(f'vf_{t}_{n}',0)
            if v and v>0 and (best is None or v<best[0]): best=(v,t,n)
    tl={'pv':'PV','wind_on':'onsh.','wind_off':'offsh.'}[best[1]]
    rows.append((LBL[sc], b['sys_cost']/1e9, b['vf_min'], f"{tl} {best[2]}",
                 b['vf_genw'], b['vf_genw']-b['vf_min'], b['cr_min'], b['cannib_loss_bn']))
with open(os.path.join(OUT,'tab_baselines.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrcrrrr}\n\\toprule\n")
    f.write("Scenario & Cost [bn\\euro/yr] & $\\vfmin$ & binding & $\\vfgw$ & disp. & $\\crm_{\\min}$ & $L$ [bn\\euro/yr]\\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{r[0]} & {r[1]:.1f} & {r[2]:.3f} & {r[3]} & {r[4]:.3f} & {r[5]:.3f} & {r[6]:.2f} & {r[7]:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- T3 headline frontier (de2040 vf) ----------
d=D[('de2040','vf')]
with open(os.path.join(OUT,'tab_headline.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrrrrc}\n\\toprule\n")
    f.write("Target & $\\vfmin$ & $\\Pi$ [bn\\euro/yr] & $L$ [bn\\euro/yr] & $\\vfgw$ & Storage [GW] & feas.\\\\\n\\midrule\n")
    for i,r in d.iterrows():
        sto=(r['sto_mw']+r['sto_cz_mw']+r['sto_dk_mw']+r['sto_at_mw']+r['sto_nl_mw'])/1000
        tgt='--' if i==0 else (f"{r['vf_target']:.3f}" if PAPER_SET == 'old' else f"{r['vf_target']:.4f}")
        feas='--' if i==0 else (f"{int(r['n_feasible_restarts'])}/4" if not np.isnan(r.get('n_feasible_restarts',np.nan)) else '0/4')
        star='' if i==0 or r['vf_min']>=r['vf_target']-0.003 else '$^{\\dagger}$'
        f.write(f"{tgt}{star} & {r['vf_min']:.3f} & {r['premium_bn']:.2f} & {r['cannib_loss_bn']:.1f} & {r['vf_genw']:.3f} & {sto:.0f} & {feas}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- F2 frontiers ----------
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})  # embed TrueType, no Type-3 fonts
col={'nt2030':'#888888','de2035':'#0072B2','de2040':'#D55E00','de2050':'#009E73'}
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9))
for k,(metric,a) in enumerate([('vf_min',ax[0]),('vf_genw',ax[1])]):
    tg='vf' if k==0 else 'vfw'
    for sc in SC:
        d=D[(sc,tg)]
        hh=[d.iloc[0]]+hits(d,metric); xx=misses(d,metric)
        a.plot([r[metric] for r in hh],[r['premium_bn'] for r in hh],'-o',ms=3.5,lw=1.4,color=col[sc],label=LBL[sc])
        if xx:
            a.plot([r[metric] for r in xx],[r['premium_bn'] for r in xx],'o',ms=4.5,mfc='white',color=col[sc])
            a.axvline(_ceil(sc, d, metric),color=col[sc],lw=0.6,ls=':',alpha=0.6)
    a.set_xlabel(r'achieved $VF_{\min}$' if k==0 else r'achieved $VF_{\mathrm{gw}}$')
    a.set_ylabel('mitigation premium [bn EUR/yr]' if k==0 else '')
    a.set_title('(a) weakest-link targeting' if k==0 else '(b) generation-weighted targeting',fontsize=9,loc='left')
ax[0].legend(frameon=False,fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_frontiers.pdf')); plt.close(fig)

# ---------- F3 marginal mitigation cost ----------
fig,a=plt.subplots(figsize=(4.4,2.9))
for sc in SC:
    d=D[(sc,'vf')]; hh=[d.iloc[0]]+hits(d,'vf_min')
    xs=[]; ys=[]
    for i in range(len(hh)-1):
        dv=hh[i+1]['vf_min']-hh[i]['vf_min']
        if dv<=1e-9: continue
        xs.append(0.5*(hh[i]['vf_min']+hh[i+1]['vf_min']))
        ys.append((hh[i+1]['premium_bn']-hh[i]['premium_bn'])/dv*0.01)
    a.plot(xs,ys,'-o',ms=3.5,lw=1.4,color=col[sc],label=LBL[sc])
a.set_yscale('log'); a.set_xlabel(r'$VF_{\min}$'); a.set_ylabel('marginal mitigation cost\n[bn EUR per +0.01 VF point]')
a.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_marginal.pdf')); plt.close(fig)
# ---------- T1 scenarios + T-potentials (S4) ----------
import sys
sys.path.insert(0, _PV)
from scenario_data import get_scenario
rows=[]
for sc in SC:
    s_=get_scenario(sc)
    vre_de=s_['vre']['DE']; dem=sum(s_['demand_twh'].values())
    el=sum(s_['electro_twh'].values()); ev=sum(v['K_ev'] for v in s_['ev'].values())/1000
    rows.append((LBL[sc], s_['tau'], vre_de['pv']/1000, vre_de['wind_on']/1000,
                 vre_de['wind_off']/1000, dem, el, ev))
with open(os.path.join(OUT,'tab_scenarios.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
    f.write("Scenario & CO$_2$ [\\euro/t] & \\multicolumn{3}{c}{DE VRE [GW]} & Demand & Electrolysis & EV fleet\\\\\n")
    f.write(" & & PV & onsh. & offsh. & [TWh] & [TWh$_{el}$] & [GW]\\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{r[0]} & {r[1]:.0f} & {r[2]:.0f} & {r[3]:.0f} & {r[4]:.0f} & {r[5]:.0f} & {r[6]:.0f} & {r[7]:.0f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# potentials snapshot: extracted from data_pipeline node_params on 2026-07-27
# (see progress.md S.86; regenerate via the loader if caps ever change)
CAPS={'DE':(200,800,200,16.0),'CZ':(15,60,15,2.0),'DK':(40,160,20,2.0),
      'AT':(30,120,10,1.6),'NL':(40,160,20,3.0)}  # stoP GW, stoE GWh, P2G GW, DSM GW
LINKS={'DE--CZ':(2.6,5.6),'DE--DK':(4.1,9.1),'DE--AT':(7.5,11.5),
       'DE--NL':(5.0,8.0),'DK--NL':(0.7,2.7),'CZ--AT':(0.9,2.4)}
with open(os.path.join(OUT,'tab_potentials.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrrr@{\\qquad}lrr}\n\\toprule\n")
    f.write("Node & Sto.\\ P & Sto.\\ E & P2G & DSM & Link & existing & max\\\\\n")
    f.write(" & [GW] & [GWh] & [GW] & [GW] & & [GW] & [GW]\\\\\n\\midrule\n")
    ln=list(LINKS.items())
    for i,(n,c) in enumerate(CAPS.items()):
        l=ln[i] if i<len(ln) else ('',('',''))
        lk=f"{l[0]} & {l[1][0]} & {l[1][1]}" if l[0] else " & & "
        f.write(f"{n} & {c[0]} & {c[1]} & {c[2]}$^{{a}}$ & {c[3]} & {lk}\\\\\n" if n=='AT'
                else f"{n} & {c[0]} & {c[1]} & {c[2]} & {c[3]} & {lk}\\\\\n")
    l=ln[5]; f.write(f" & & & & & {l[0]} & {l[1][0]} & {l[1][1]}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- S5: certificate + probe macros, tab_probes, fig_metric ----------
g=pd.read_csv(P('grid_floor_5node_matched.csv'))
mac2=[]
def M2(n_,v,fmt='%.3f'): mac2.append(r'\newcommand{\%s}{%s}'%(n_,fmt%v))
M2('envMaxVf', g['vf_min'].max())
# envelope multiple at the elbow target (old set: 0.738, CMA 0.744 vs envelope min)
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac2)+'\n')
# ---------- S5a: envelope multiple at the elbow ----------
mac2=[]
_ie5, _re5 = _elbow_row(D[('de2040','vf')])
env738=g[g['vf_min']>=_re5['vf_target']-0.003]['premium_bn'].min()
M2('envAtElbow', env738, '%.2f')
M2('envMultElbow', env738/_re5['premium_bn'], '%.0f' if PAPER_SET == 'old' else '%.1f')
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac2)+'\n'); mac2=[]
# ---------- S5b: ceiling probes ----------
mac2=[]
# Ceiling probes (S.86 + S.98). Each diagnostic CSV: row 0 = that variant's own
# cost optimum, last row = returned portfolio. Premiums are vs the variant's
# own B_opt. Variants: NL storage potential x2 (40->80 GW), or 8-h storage
# duration at all nodes with potentials unchanged (STO_EP_RATIO=8).
if PAPER_SET == 'old':
    probes=PIN('probes')
    prow=[]; pv={}
    for sc,var,t,f,nm in probes:
        d=pd.read_csv(os.path.join(BASE,f)); r=d.iloc[-1]; pv[nm]=r
        M2('vf'+nm, r['vf_min']); M2('prem'+nm, r['premium_bn'], '%.2f')
        hit = 'yes' if r['vf_min']>=float(t)-0.003 else 'no'
        prow.append((sc, var, t, f"{r['vf_min']:.3f}", hit, f"{r['premium_bn']:.2f}",
                     f"{r[PLAN()['sto_key']]/1000:.0f}"))
    # ceiling shift per doubling of NL storage, in value-factor points
    _c40=D[('de2040','vf')]; _c50=D[('de2050','vf')]
    _ceil40=_ceil('de2040', _c40, 'vf_min'); _ceil50=_ceil('de2050', _c50, 'vf_min')
    M2('ceilShiftForty', 100*(pv['ProbeEightTwo']['vf_min']-_ceil40), '%.1f')
    M2('ceilShiftFifty', 100*(pv['FiftyEightZero']['vf_min']-_ceil50), '%.1f')
    with open(os.path.join(OUT,'tab_probes.tex'),'w') as f:
        f.write("\\begin{tabular}{lllcccc}\n\\toprule\n")
        f.write("Scenario & variant & target & achieved & attained? & $\\Pi$ [bn\\euro/yr] & %s sto.\\ [GW]\\\\\n\\midrule\n" % PLAN()['W'])
        for r in prow: f.write(" & ".join(r)+"\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
else:
    # rerun: probes from logs/phase2_plan.json; a probe that has not run yet
    # gets red placeholder macros and a 'pending' row, so the paper compiles
    _PH = r'\textcolor{red}{\textbf{??}}'
    _TL = {'pv':'solar','wind_on':'onshore wind','wind_off':'offshore wind'}
    _jp = json.load(open(os.path.join(LOGD, 'phase2_plan.json')))
    _SCN = {'de2040':'Forty','de2050':'Fifty'}
    prow=[]; _best={}; _allbest=[]
    for q in _jp['probes']:
        kind, sc_, ab = q['tag'].split('_')[0], q['scenario'], q['tag'].split('_')[2]
        nm = {'w2x':'Wx','ep8':'Ep'}[kind] + _SCN[sc_] + ab.upper()
        co = q['env'].get('CAP_OVERRIDES', '')
        rel = co.split('.')[0] if co else None
        var = (f"{rel} storage $\\times$2" if rel else '8-h storage')
        f_ = os.path.join(BASE, f"diagnostic_vf_{q['target']:.3f}_{q['tag']}", 'diagnostic_frontier.csv')
        mac2.append(r'\newcommand{\target%s}{%.3f}' % (nm, q['target']))
        if rel: mac2.append(r'\newcommand{\relAdj%s}{%s}' % (nm, ADJ[rel]))
        if not os.path.exists(f_):
            for k_ in ('vf','prem','bindAdj','bindTech','stoRel','stoW','pvRel'):
                mac2.append(r'\newcommand{\%s%s}{%s}' % (k_, nm, _PH))
            prow.append((LBL[sc_], var, f"{q['target']:.3f}", r'\multicolumn{6}{c}{\textcolor{red}{pending}}'))
            continue
        r = pd.read_csv(f_).iloc[-1]
        cols = [f'vf_{t_}_{n_}' for n_ in NODES for t_ in TECH if f'vf_{t_}_{n_}' in r.index]
        v_ = r[cols].astype(float); v_ = v_[v_ > 0]; b_ = v_.idxmin()
        bn, bt = b_.split('_')[-1], '_'.join(b_.split('_')[1:-1])
        M2('vf'+nm, r['vf_min']); M2('prem'+nm, r['premium_bn'], '%.2f'); M2('stoDe'+nm, r['sto_mw']/1000, '%.0f')
        mac2.append(r'\newcommand{\bindAdj%s}{%s}' % (nm, ADJ[bn])); mac2.append(r'\newcommand{\bindTech%s}{%s}' % (nm, _TL[bt]))
        M2('stoW'+nm, r[_jp['sto_key']]/1000, '%.0f')
        if rel:
            M2('stoRel'+nm, r[{'DE':'sto_mw'}.get(rel, f'sto_{rel.lower()}_mw')]/1000, '%.0f')
            M2('pvRel'+nm, r[f'vf_pv_{rel}'])
        hit = 'yes' if r['vf_min'] >= q['target']-0.003 else 'no'
        prow.append((LBL[sc_], var, f"{q['target']:.3f}", f"{r['vf_min']:.3f}", hit, f"{r['premium_bn']:.2f}",
                     f"{r['sto_dk_mw']/1000:.0f}/{r['sto_nl_mw']/1000:.0f}", f"{bn} {_TL[bt]}"))
        # best restart by true VF_min (min of the per-node columns; the 'vf_min'
        # column of the restarts sheet held cr_min before 2026-11-04)
        _bv = float(r['vf_min'])
        _sx = os.path.join(BASE, 'sensitivity', f"sensitivity_vf{q['target']:.3f}_{q['tag']}.xlsx")
        if os.path.exists(_sx):
            _rs = pd.read_excel(_sx, sheet_name='restarts')
            _vc = [c for c in _rs.columns if c.startswith('vf_') and c.split('_')[-1] in NODES]
            _bv = max(_bv, float(_rs[_vc].where(_rs[_vc] > 0).min(axis=1).max()))
        M2('bestRs'+nm, _bv)
        prow[-1] = prow[-1][:4] + (f"{_bv:.3f}", 'yes' if _bv >= q['target']-0.003 else 'no') + prow[-1][5:]
        k2 = (sc_, kind); _best[k2] = max(_best.get(k2, 0.0), _bv)
        _allbest.append(_bv)
    # best value factor per scenario and relaxation, and its distance to the base ceiling
    for (sc_, kind), b_ in _best.items():
        nm = {'w2x':'Wx','ep8':'Ep'}[kind] + _SCN[sc_]
        c_ = _ceil(sc_, D[(sc_,'vf')], 'vf_min')
        M2('best'+nm, b_); M2('shift'+nm+'Pts', 100*(b_-c_), '%.1f')
    for sc_ in ('de2040','de2050'):
        for kind in ('w2x','ep8'):
            if (sc_, kind) not in _best and any(q['scenario']==sc_ and q['tag'].startswith(kind) for q in _jp['probes']):
                nm = {'w2x':'Wx','ep8':'Ep'}[kind] + _SCN[sc_]
                mac2 += [r'\newcommand{\best%s}{%s}' % (nm, _PH), r'\newcommand{\shift%sPts}{%s}' % (nm, _PH)]
    M2('nProbes', len(_jp['probes']), '%d')
    if _allbest and len(_allbest) == len(_jp['probes']): M2('probeMaxVf', max(_allbest))
    else: mac2.append(r'\newcommand{\probeMaxVf}{%s}' % _PH)
    # premium range of all DE 2040 probes
    _p40=[]
    for q in _jp['probes']:
        f_ = os.path.join(BASE, f"diagnostic_vf_{q['target']:.3f}_{q['tag']}", 'diagnostic_frontier.csv')
        if os.path.exists(f_) and q['scenario']=='de2040': _p40.append(float(pd.read_csv(f_).iloc[-1]['premium_bn']))
    if _p40:
        M2('premProbeFortyLo', min(_p40), '%.1f'); M2('premProbeFortyHi', max(_p40), '%.1f')
        _lh = hits(D[('de2040','vf')],'vf_min')[-1]['premium_bn']
        M2('probeRatioLo', min(_p40)/_lh, '%.1f'); M2('probeRatioHi', max(_p40)/_lh, '%.1f')
    # the base frontier at its highest point: relaxed node vs weak node W (solar VF, storage, potential)
    import sys as _s5; _s5.path.insert(0, _PV)
    from phase2_plan import STO_POT as _SP, STO_POT_2050 as _SP50
    _W = _jp['weak_node_de2040']
    for sc_ in ('de2040','de2050'):
        qs = [q for q in _jp['probes'] if q['scenario']==sc_ and q['tag'].startswith('w2x')]
        if not qs: continue
        rel = qs[0]['env']['CAP_OVERRIDES'].split('.')[0]
        d_ = D[(sc_,'vf')]; r_ = d_.loc[d_['vf_min'].astype(float).idxmax()]
        pot = _SP50 if sc_=='de2050' else _SP
        sk = lambda n_: 'sto_mw' if n_=='DE' else f'sto_{n_.lower()}_mw'
        tag_ = _SCN[sc_]
        M2('topPvRel'+tag_, r_[f'vf_pv_{rel}']); M2('topStoRel'+tag_, r_[sk(rel)]/1000, '%.0f'); M2('potRel'+tag_, pot[rel]/1000, '%.0f')
        M2('topPvW'+tag_, r_[f'vf_pv_{_W}']); M2('topStoW'+tag_, r_[sk(_W)]/1000, '%.0f'); M2('potW'+tag_, pot[_W]/1000, '%.0f')
        M2('topVf'+tag_, r_['vf_min'])
    with open(os.path.join(OUT,'tab_probes.tex'),'w') as f:
        f.write("\\begin{tabular}{lllcccccl}\n\\toprule\n")
        f.write("Scenario & relaxation & target & returned & best & attained & $\\Pi$ [bn\\euro] & DK/NL [GW] & binding\\\\\n\\midrule\n")
        for r in prow: f.write(" & ".join(r)+"\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac2)+'\n'); mac2=[]
# ---------- S5c: dispersion + nt2030 macros, fig_metric ----------
mac2=[]
def M2(n_,v,fmt='%.3f'): mac2.append(r'\newcommand{\%s}{%s}'%(n_,fmt%v))
dvfw=D[('de2040','vfw')]
M2('nlPvEndVfw', hits(dvfw,'vf_genw')[-1]['vf_pv_'+PLAN()['W']])
M2('dispBase', D[('de2040','vf')].iloc[0]['vf_genw']-D[('de2040','vf')].iloc[0]['vf_min'])
_lv=hits(D[('de2040','vf')],'vf_min')[-1]; _lw2=hits(dvfw,'vf_genw')[-1]
M2('dispEndVf', _lv['vf_genw']-_lv['vf_min']); M2('dispEndVfw', _lw2['vf_genw']-_lw2['vf_min'])
# nt2030 CR flatness
dnt=D[('nt2030','vf')]
M2('crNtLo', dnt['cr_min'].min(),'%.2f'); M2('crNtHi', dnt['cr_min'].max(),'%.2f')
M2('vfNtEnd', hits(dnt,'vf_min')[-1]['vf_min'])
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac2)+'\n')

# fig_metric: per-node PV VF along the de2040 vfw frontier + both dispersions
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9))
d=D[('de2040','vfw')]; hh=[d.iloc[0]]+hits(d,'vf_genw')
xs=[r['vf_genw'] for r in hh]
ncol={'DE':'#D55E00','CZ':'#CC79A7','DK':'#0072B2','AT':'#009E73','NL':'#000000'}
for n in NODES:
    ax[0].plot(xs,[r[f'vf_pv_{n}'] for r in hh],'-o',ms=3,lw=1.3,color=ncol[n],label=n)
ax[0].set_xlabel(r'achieved $VF_{\mathrm{gw}}$ (targeted)')
ax[0].set_ylabel('solar value factor by node')
ax[0].set_title('(a) fleet-average targeting lifts every node',fontsize=9,loc='left')
ax[0].legend(frameon=False,fontsize=7,ncol=2)
for tg,lab,c in [('vf','weakest-link targeting','#0072B2'),('vfw','fleet-average targeting','#D55E00')]:
    d=D[('de2040',tg)]; m='vf_min' if tg=='vf' else 'vf_genw'
    hh=[d.iloc[0]]+hits(d,m)
    ax[1].plot([r['premium_bn'] for r in hh],[r['vf_genw']-r['vf_min'] for r in hh],
               '-o',ms=3,lw=1.3,color=c,label=lab)
ax[1].set_xlabel('mitigation premium [bn EUR/yr]')
ax[1].set_ylabel(r'dispersion $VF_{\mathrm{gw}}-VF_{\min}$')
ax[1].set_title('(b) dispersion narrows under both targets',fontsize=9,loc='left')
ax[1].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_metric.pdf')); plt.close(fig)

# ---------- S6: appendix tables ----------
from scenario_data import _costs, _EV
# B1: year-specific annualized costs
yrs=[('2030','NT 2030'),('2035','DE 2035'),('2040','DE 2040'),('2050','DE 2050')]
with open(os.path.join(OUT,'tab_costs.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    f.write("Annualized cost & 2030 & 2035 & 2040 & 2050\\\\\n\\midrule\n")
    rows=[('PV [\\euro/kW/yr]','capex','pv'),('Onshore wind','capex','wind_on'),
          ('Offshore wind','capex','wind_off'),('CCGT','capex','ccgt'),
          ('Battery power [\\euro/kW/yr]',None,'sto_capex_p'),
          ('Battery energy [\\euro/kWh/yr]',None,'sto_capex_e'),
          ('Electrolyser [\\euro/kW/yr]',None,'sc_capex')]
    for lab,grp,k in rows:
        vals=[]
        for y,_ in yrs:
            c=_costs(y)
            v=(c[grp][k] if grp else c[k])/1000.0
            vals.append(f"{v:.1f}")
        f.write(lab+" & "+" & ".join(vals)+"\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# B2: EV fleet inputs (2040 vintage, per node)
with open(os.path.join(OUT,'tab_ev.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
    f.write("Node & $K^{ev}$ [GW] & $E^{ev}$ [GWh] & charging [TWh/yr]\\\\\n\\midrule\n")
    for n in NODES:
        e=_EV['2040'][n]
        f.write(f"{n} & {e['K_ev']/1000:.1f} & {e['E']/1000:.0f} & {e['elec']/1e6:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# ---------- S6b: all-scenario frontier table (App. D1) ----------
with open(os.path.join(OUT,'tab_allfront.tex'),'w') as f:
    f.write("\\begin{tabular}{llrrrrcr}\n\\toprule\n")
    f.write("Scenario & metric & target & achieved & $\\Pi$ [bn\\euro/yr] & $\\vfgw$ & feas. & spread [\\%]\\\\\n\\midrule\n")
    for sc in SC:
        for tg in ['vf','vfw']:
            d=D[(sc,tg)]; m='vf_min' if tg=='vf' else 'vf_genw'
            lab={'vf':'$\\vfmin$','vfw':'$\\vfgw$'}[tg]
            for i,r in d.iterrows():
                if i==0: continue
                feas = f"{int(r['n_feasible_restarts'])}/4" if not np.isnan(r.get('n_feasible_restarts',np.nan)) else '0/4'
                star = '' if r[m]>=r['vf_target']-0.003 else '$^{\\dagger}$'
                gp = r.get('restart_gap_pct', np.nan)
                gps = '--' if gp is None or not np.isfinite(gp) else f"{gp:.2f}"
                _tf = f"{r['vf_target']:.3f}" if PAPER_SET == 'old' else f"{r['vf_target']:.4f}"
                f.write(f"{LBL[sc]} & {lab} & {_tf}{star} & {r[m]:.3f} & "
                        f"{r['premium_bn']:.2f} & {r['vf_genw']:.3f} & {feas} & {gps}\\\\\n")
        f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# ---------- S7: robustness table from run logs ----------
import re
def _dec(p):
    raw=open(p,'rb').read()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff') or b'\x00' in raw[:400]:
        try: return raw.decode('utf-16')
        except Exception: pass
    return raw.decode('utf-8',errors='replace')
def _bopt(f):
    m=re.search(r'B_opt cost ([\d.]+) bn \| VF_min ([\d.]+) \| VF_genw ([\d.]+) \| CR_min ([\d.]+)', _dec(PL(f)))
    return [float(x) for x in m.groups()]
def _diag(f):
    t=_dec(PL(f))
    vf=float(re.search(r'Achieved VF_min = ([\d.]+)',t).group(1))
    pr=float(re.search(r'Premium = ([\d.]+) bn',t).group(1))
    bind=re.search(r'Binding pair: (\w+) (\w+)',t).groups()
    return vf,pr,f"{bind[1].upper()} {bind[0].upper()}"
def _binding(r):
    best=None
    for n in NODES:
        for t_ in TECH:
            v_=r.get(f'vf_{t_}_{n}',0)
            if v_ and v_>0 and (best is None or v_<best[0]): best=(v_,t_,n)
    return f"{ {'pv':'PV','wind_on':'onsh.','wind_off':'offsh.'}[best[1]] } {best[2]}"
if PAPER_SET == 'old':
    _h0=D[('de2040','vf')].iloc[0]; _he=_elbow_row(D[('de2040','vf')])[1]
    _hcost=_h0['sys_cost']/1e9
    rows=[]
    rows.append(("Headline (borders, CY2009, 12 wk)", _hcost, _h0['vf_min'], _he['premium_bn'], _binding(_he)))
    c,v,gw,cr=_bopt('rob_noresid_base.log'); dv,dp,db=_diag('rob_noresid_diag.log')
    rows.append(("Isolated system (no residual borders)", c, v, dp, db))
    c2,v2,gw2,cr2=_bopt('rob_widep_base.log')
    rows.append(("Wide border prices (0/150 \\euro)", c2, v2, None, "--"))
    _vcy=[]
    for cy,lab in [(2010,'Climate year 2010 (low resource)'),(2003,'Climate year 2003 (high solar)'),
                   (1995,'Climate year 1995'),(2008,'Climate year 2008')]:
        c3,v3,gw3,cr3=_bopt(f'rob_cy{cy}_base.log'); _vcy.append(v3)
        if cy in (2010,2003):
            dv3,dp3,db3=_diag(f'rob_cy{cy}_diag.log'); rows.append((lab,c3,v3,dp3,db3))
        else: rows.append((lab,c3,v3,None,"--"))
    c4,v4,gw4,cr4=_bopt('rob_gold52_base.log')
    rows.append(("52-week full chronology", c4, v4, None, "--"))
    c5,v5,gw5,cr5=_bopt('rob_dkdom_base.log')
    rows.append(("Denmark domestic-only H$_2$ (5 TWh)", c5, v5, None, "PV DK"))
    with open(os.path.join(OUT,'tab_robust.tex'),'w') as f:
        f.write("\\begin{tabular}{lrrrc}\n\\toprule\n")
        f.write("Variant & $C(\\Bopt)$ [bn] & $\\vfmin(\\Bopt)$ & $\\Pi(%.3f)$ [bn] & binding\\\\\n\\midrule\n" % _he['vf_target'])
        for lab,c_,v_,p_,b_ in rows:
            ps=f"{p_:.2f}" if p_ is not None else "--"
            f.write(f"{lab} & {c_:.1f} & {v_:.3f} & {ps} & {b_}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    mac3=[]
    def M3(n_,v,fmt='%.3f'): mac3.append(r'\newcommand{\%s}{%s}'%(n_,fmt%v))
    M3('vfIso',v); M3('vfWideP',v2); M3('premIso',dp,'%.2f')
    M3('vfCyLo',min([v]+_vcy))
    M3('vfGoldFifty',v4); M3('costGoldGapPct',100*(_hcost-c4)/c4,'%.1f')
    M3('vfDkDom',v5); M3('crDkDom',cr5,'%.2f')
    M3('premCyThree',_diag('rob_cy2003_diag.log')[1],'%.2f')
    M3('premCyTen',_diag('rob_cy2010_diag.log')[1],'%.2f')
    open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac3)+'\n')
else:
    # rerun: baseline per-node value factors parsed from the run logs, binding
    # asset at the baseline and at the elbow re-solve
    def _bnodes(f):
        t_=_dec(PL(f)); i_=t_.find('B_opt cost'); seg=t_[i_:i_+1200]
        out={}
        for n_,pv_,wo_,wf_ in re.findall(r'(DE|CZ|DK|AT|NL): VF_pv=([\d.]+), VF_wind_on=([\d.]+), VF_wind_off=([\d.]+)', seg):
            for tk,v_ in (('pv',pv_),('wind_on',wo_),('wind_off',wf_)):
                if float(v_)>0: out[(n_,tk)]=float(v_)
        return out
    _TLS={'pv':'solar','wind_on':'onshore','wind_off':'offshore'}
    def _bstr(nv): (n_,tk)=min(nv,key=nv.get); return f"{n_} {_TLS[tk]}"
    def _dg(f):
        t_=_dec(PL(f))
        vf=float(re.search(r'Achieved VF_min = ([\d.]+)',t_).group(1)); pr=float(re.search(r'Premium = ([\d.]+) bn',t_).group(1))
        b_=re.search(r'Binding pair: (\w+) (\w+)',t_).groups()
        return vf,pr,f"{b_[0].upper()} {_TLS.get(b_[1].lower(),b_[1])}"
    _h0=D[('de2040','vf')].iloc[0]; _ie,_he=_elbow_row(D[('de2040','vf')]); _tg=float(f"{_he['vf_target']:.3f}")   # re-solve target as run
    def _rb(r): 
        nv={(n_,tk):float(r[f'vf_{tk}_{n_}']) for n_ in NODES for tk in TECH if float(r.get(f'vf_{tk}_{n_}',0) or 0)>0}
        return _bstr(nv), nv
    _hb,_hnv=_rb(_h0); _heb,_=_rb(_he)
    rows=[("Headline (hourly borders, CY 2009, 12 wk)", _h0['sys_cost']/1e9, _h0['vf_min'], _hb, _he['premium_bn'], _heb)]
    R7={}
    for key,lab,base,diag in [('iso','Isolated (no residual borders)','rob_noresid_base.log','rob_noresid_diag.log'),
                              ('wide','Fixed border prices (0/150 \\euro)','rob_widep_base.log',None),
                              ('cy2010','Climate year 2010','rob_cy2010_base.log','rob_cy2010_diag.log'),
                              ('cy2003','Climate year 2003','rob_cy2003_base.log','rob_cy2003_diag.log'),
                              ('cy1995','Climate year 1995','rob_cy1995_base.log',None),
                              ('cy2008','Climate year 2008','rob_cy2008_base.log',None),
                              ('gold','52 weeks (full year)','rob_gold52_base.log',None),
                              ('dkdom','Danish H$_2$ domestic only (5 TWh)','rob_dkdom_base.log',None)]:
        c_,v_,gw_,cr_=_bopt(base); nv=_bnodes(base)
        d_=_dg(diag) if diag else None
        R7[key]=dict(cost=c_,vf=v_,gw=gw_,cr=cr_,nv=nv,bind=_bstr(nv),diag=d_)
        rows.append((lab,c_,v_,_bstr(nv),d_[1] if d_ else None,d_[2] if d_ else '--'))
    with open(os.path.join(OUT,'tab_robust.tex'),'w') as f:
        f.write("\\begin{tabular}{lrrlrl}\n\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{cost optimum} & \\multicolumn{2}{c}{$\\vfmin\\ge %.3f$}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n" % _tg)
        f.write("Variant & $C$ [bn\\euro] & $\\vfmin$ & binding & $\\Pi$ [bn\\euro] & binding\\\\\n\\midrule\n")
        for lab,c_,v_,b_,p_,pb_ in rows:
            ps = "--" if p_ is None else (f"{p_:.2f}" + ("$^{a}$" if v_ >= _tg-0.003 else ""))
            f.write(f"{lab} & {c_:.1f} & {v_:.3f} & {b_} & {ps} & {pb_}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    mac3=[]
    def M3(n_,v,fmt='%.3f'): mac3.append(r'\newcommand{\%s}{%s}'%(n_,fmt%v))
    _hc=_h0['sys_cost']/1e9
    M3('vfIso',R7['iso']['vf']); M3('costIsoAdd',R7['iso']['cost']-_hc,'%.1f'); M3('vfIsoGainPts',100*(R7['iso']['vf']-_h0['vf_min']),'%.0f')
    M3('vfWideP',R7['wide']['vf']); M3('costWideAdd',R7['wide']['cost']-_hc,'%.1f')
    _cy={2009:(_h0['vf_min'],_hb)}
    for y in (1995,2003,2008,2010): _cy[y]=(R7[f'cy{y}']['vf'],R7[f'cy{y}']['bind'])
    M3('vfCyLo',min(v for v,_ in _cy.values())); M3('vfCyHi',max(v for v,_ in _cy.values()))
    _ylo=min(_cy,key=lambda y:_cy[y][0]); _yhi=max(_cy,key=lambda y:_cy[y][0])
    M3('cyLoYear',_ylo,'%d'); M3('cyHiYear',_yhi,'%d')
    _wn=_W_=PLAN()['W']
    M3('nCyWeak',sum(1 for v,b in _cy.values() if b.startswith(_wn)),'%d'); M3('nCy',len(_cy),'%d')
    mac3.append(r'\newcommand{\nCyWeakW}{%s}' % NW(sum(1 for v,b in _cy.values() if b.startswith(_wn)))); mac3.append(r'\newcommand{\nCyW}{%s}' % NW(len(_cy)))
    for y,nm in ((2003,'Three'),(2010,'Ten')):
        dv,dp,db=R7[f'cy{y}']['diag']
        M3('premCy'+nm,dp,'%.2f'); M3('gapCy'+nm+'Pts',100*(_tg-R7[f'cy{y}']['vf']),'%.1f')
        mac3.append(r'\newcommand{\bindCy%s}{%s}' % (nm, ADJ[db.split()[0]]+' '+db.split()[1]))
    M3('gapHeadPts',100*(_tg-_h0['vf_min']),'%.1f')
    g_=R7['gold']
    M3('vfGoldFifty',g_['vf']); M3('costGoldGapPct',100*(_hc-g_['cost'])/g_['cost'],'%.1f')
    M3('vfGoldGapPts',100*(g_['vf']-_h0['vf_min']),'%.1f')
    M3('vfwGoldGapPts',abs(100*(g_['gw']-_h0['vf_genw'])),'%.1f'); M3('costGoldAbovePct',100*(_hc-g_['cost'])/g_['cost'],'%.1f')
    M3('pvDkGold',g_['nv'][('DK','pv')]); M3('pvNlGold',g_['nv'][(_wn,'pv')])
    M3('pvDkHead',_hnv[('DK','pv')]); M3('pvNlHead',_hnv[(_wn,'pv')])
    M3('vfDkDom',R7['dkdom']['vf']); M3('crDkDom',R7['dkdom']['cr'],'%.2f'); M3('vfDkDomDropPts',100*(_h0['vf_min']-R7['dkdom']['vf']),'%.0f')
    open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac3)+'\n')
# ---------- S8: fig_saturation (ceiling mechanism) + fig_portfolio ----------
d=D[('de2040','vf')]
xs=d['vf_min'].tolist()
fig,a=plt.subplots(figsize=(4.6,2.9))
_pl=PLAN(); _W=_pl['W']; _w=_W.lower()
a.plot(xs,[100*r/_pl['sto_pot'] for r in d[_pl['sto_key']]],'-o',ms=3,lw=1.4,color='#000000',label=f"{_W} storage / {_pl['sto_pot']/1000:.0f} GW")
_dk=('dsm_mw' if _W=='DE' else f'dsm_{_w}_mw')
a.plot(xs,[100*r/DSM_POT[_W] for r in d[_dk]],'-s',ms=3,lw=1.2,color='#0072B2',label=f"{_W} DSM / {DSM_POT[_W]/1000:g} GW")
for _key,_mk_,_col in ((_pl['main_ntc_key'],'-^','#D55E00'),(_pl['second_ntc_key'],'-v','#009E73')):
    if not _key: continue
    _l=_key[4:-3]; _tm=NTC_TOTAL_MAX[_l]
    a.plot(xs,[100*r/_tm for r in d[f'ntc_total_{_l}_mw']],_mk_,ms=3,lw=1.2,color=_col,
           label=f"{_l.upper().replace('_','-')} NTC / {_tm/1000:g} GW")
a.axhline(100,color='gray',lw=0.7,ls='--'); a.axvline(_ceil('de2040', d, 'vf_min'),color='gray',lw=0.6,ls=':')
a.set_xlabel(r'achieved $VF_{\min}$'); a.set_ylabel('utilization of potential [%]')
a.set_ylim(0,110); a.legend(frameon=False,fontsize=7,loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_saturation.pdf')); plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(7.0,2.9),sharey=True)
for k,(tg,metric,ttl) in enumerate([('vf','vf_min','(a) weakest-link targeting'),
                                     ('vfw','vf_genw','(b) fleet-average targeting')]):
    d=D[('de2040',tg)]; hh=[d.iloc[0]]+hits(d,metric)
    xs=[r[metric] for r in hh]
    series={'DE':[r['sto_mw']/1000 for r in hh],'NL':[r['sto_nl_mw']/1000 for r in hh],
            'AT':[r['sto_at_mw']/1000 for r in hh],'CZ':[r['sto_cz_mw']/1000 for r in hh],
            'DK':[r['sto_dk_mw']/1000 for r in hh]}
    ax[k].stackplot(xs,series.values(),labels=series.keys(),
                    colors=['#D55E00','#000000','#009E73','#CC79A7','#0072B2'],alpha=0.85)
    ax[k].set_title(ttl,fontsize=9,loc='left')
    ax[k].set_xlabel(r'achieved $VF_{\min}$' if k==0 else r'achieved $VF_{\mathrm{gw}}$')
ax[0].set_ylabel('battery power [GW]'); ax[0].legend(frameon=False,fontsize=7,loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_portfolio.pdf')); plt.close(fig)
# ---------- S9: capture prices (absolute cannibalization), baselines ----------
# Method: annual potential generation from the production loader (PECD cy2009,
# 12wk weights; gen/cf_annual.json snapshot). p_bar_n anchored on ONSHORE WIND
# (least-curtailed tech): CP_won = CR_won*AC_won/gen_won; p_bar = CP_won/VF_won;
# then CP_vn = VF_vn * p_bar_n exactly. Offshore gives an independent p_bar
# estimate -> max discrepancy reported as validation. Rerun files: p_bar_n is
# the stored nodal average price (base_price_<n>), onshore anchor = validation.
import json as _json
CF=_json.load(open(os.path.join(OUT,'cf_annual.json')))
from scenario_data import _costs as _sc_costs
YR={'nt2030':'2030','de2035':'2035','de2040':'2040','de2050':'2050'}
cap_rows=[]; maxdisc=0.0
for sc in SC:
    s_=get_scenario(sc); cost=_sc_costs(YR[sc]); b=D[(sc,'vf')].iloc[0]
    for n in NODES:
        K=s_['vre'][n]
        def cp_anchor(tech):
            k=K[tech]
            if k<1 or CF[tech][n]<1: return None
            AC=k*(cost['capex'][tech]+cost['fom'][tech])
            gen=k*CF[tech][n]
            cr=b.get(f'cr_{tech}_{n}',0); vf=b.get(f'vf_{tech}_{n}',0)
            if not cr or not vf: return None
            return (cr*AC/gen)/vf
        pb=cp_anchor('wind_on'); pb2=cp_anchor('wind_off')
        if f'base_price_{n}' in b:
            # rerun outputs store the nodal average price directly; the onshore
            # anchor becomes the validation (the offshore anchor is biased once
            # the TYNDP hubs add offshore capacity with their own profile)
            pa=float(b[f'base_price_{n}'])
            if pb: maxdisc=max(maxdisc,abs(pb-pa)/pa*100)
            pb=pa
        else:
            if pb is None: continue
            if pb2: maxdisc=max(maxdisc,abs(pb2-pb)/pb*100)
        row=[LBL[sc],n,f"{pb:.1f}"]
        for tech in ['pv','wind_on','wind_off']:
            vf=b.get(f'vf_{tech}_{n}',0)
            row.append(f"{vf*pb:.1f}" if vf>0 else "--")
        cap_rows.append(row)
with open(os.path.join(OUT,'tab_capture.tex'),'w') as f:
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
    f.write("Scenario & node & $\\bar p_n$ & CP$_{PV}$ & CP$_{on}$ & CP$_{off}$\\\\\n")
    f.write(" & & \\multicolumn{4}{c}{[\\euro/MWh]}\\\\\n\\midrule\n")
    last=None
    for r in cap_rows:
        lab=r[0] if r[0]!=last else ''
        if r[0]!=last and last is not None: f.write("\\midrule\n")
        last=r[0]
        f.write(f"{lab} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT,'numbers.tex'),'a').write(r'\newcommand{\capDisc}{%.1f}'%maxdisc+'\n')
print(f"capture table ok, max onshore/offshore p_bar discrepancy {maxdisc:.1f}%")
# ---------- S10: interaction analysis (factorial grid, de2040 matched) ----------
gm=pd.read_csv(P('grid_floor_5node_matched.csv'))
gm=gm[gm.status=='ok']
# effective design: sto x dsmev x ntc (f_sc vestigial under p2g-endogenous ->
# collapse duplicates by unique effective combo, keep first)
gm=gm.drop_duplicates(subset=['f_sto','f_dsmev','f_ntc'])
mac4=[]
def M4(n_,v,fmt='%.3f'): mac4.append(r'\newcommand{\%s}{%s}'%(n_,fmt%v))
# sto x ntc surface at dsmev=1.0 (complete slice)
sfc=gm[gm.f_dsmev==1.0].pivot_table(index='f_sto',columns='f_ntc',values='vf_min')
b=sfc.loc[0.0,0.0]
M4('iaBase', b)
M4('iaStoFirst', sfc.loc[0.1,0.0]-b)        # first 10% storage
M4('iaStoLastFifteen', sfc.loc[1.0,0.0]-sfc.loc[0.85,0.0])
M4('iaStoFull', sfc.loc[1.0,0.0]-b)
M4('iaNtcFull', sfc.loc[0.0,1.0]-b)
# interaction extremes
I={} 
for i in sfc.index:
    for j in sfc.columns:
        if i>0 and j>0: I[(i,j)]=sfc.loc[i,j]-sfc.loc[i,0.0]-sfc.loc[0.0,j]+b
M4('iaMostNeg', min(I.values()))
M4('iaMostPos', max(I.values()))
# cost superadditivity at VF>=0.70 (dsmev=1 slice)
g1=gm[gm.f_dsmev==1.0]
solo=g1[(g1.f_ntc==0)&(g1.vf_min>=0.70)].premium_bn.min()
both=g1[g1.vf_min>=0.70].premium_bn.min()
M4('iaSoloCost', solo, '%.2f'); M4('iaMixCost', both, '%.2f')
_bt=g1[(g1.f_ntc>0)&(g1.vf_min>=0.70-0.003)].premium_bn.min(); _st=g1[(g1.f_ntc==0)&(g1.vf_min>=0.70-0.003)].premium_bn.min()
M4('iaMixCostTol', _bt, '%.2f'); M4('iaSoloCostTol', _st, '%.2f')
M4('iaMixSavePct', 100*(1-both/solo), '%.0f')
# DSM dimension (if extension rows present): VF gain of DSM alone + interaction w/ storage
have_dsm = set(gm.f_dsmev.unique())>= {0.0,1.0}
if have_dsm:
    sd=gm[gm.f_ntc==0.0].pivot_table(index='f_sto',columns='f_dsmev',values='vf_min')
    if 0.0 in sd.columns and 0.0 in sd.index:
        b2=sd.loc[0.0,0.0]
        M4('iaDsmAlone', sd.loc[0.0,1.0]-b2)
        # interaction dsm x sto at high storage
        if 0.7 in sd.index:
            M4('iaDsmStoInt', sd.loc[0.7,1.0]-sd.loc[0.7,0.0]-(sd.loc[0.0,1.0]-b2))
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac4)+'\n')
print("interaction macros:", len(mac4), "| dsm rows present:", have_dsm)

# figure: VF_min vs storage scale at three NTC levels (dsmev=1)
fig,a=plt.subplots(figsize=(4.6,2.9))
cols={0.0:'#0072B2',0.5:'#888888',1.0:'#D55E00'}
for j in sorted(sfc.columns):
    a.plot(sfc.index*100, sfc[j],'-o',ms=3.5,lw=1.4,color=cols[j],
           label=f'interconnection at {int(j*100)}% of potential')
a.set_xlabel('storage deployment [% of potential, all nodes]')
a.set_ylabel(r'$VF_{\min}$')
a.legend(frameon=False,fontsize=7.5,loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_interactions.pdf')); plt.close(fig)
# ---------- S11: completed factorials (DSM full + de2050) ----------
mac5=[]
def M5(n_,v,fmt='%.3f'): mac5.append(r'\newcommand{\%s}{%s}'%(n_,fmt%v))
try:
    h=pd.read_csv(P('grid_floor_5node_de2050.csv'))
    h=h[h.status=='ok'].drop_duplicates(subset=['f_sto','f_dsmev','f_ntc'])
    M5('envMaxVfFifty', h.vf_min.max())
    s2=h[h.f_dsmev==1.0].pivot_table(index='f_sto',columns='f_ntc',values='vf_min')
    b3=s2.loc[0.0,0.0]
    M5('iaFiftyEarly', s2.loc[0.2,1.0]-s2.loc[0.2,0.0]-(s2.loc[0.0,1.0]-b3))
except FileNotFoundError:
    pass
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac5)+'\n')
print("S11 macros:",len(mac5))
# ---------- S12: interaction experiments at three levels ----------
import itertools
mac6=[]
def M6(n_,v,fmt='%.3f'): mac6.append(r'\newcommand{\%s}{%s}'%(n_,fmt%v))
def _rd(f): 
    g=pd.read_csv(P(f)); return g[g.status=='ok']
# (a) 2^3 decomposition for three scenarios -> appendix table
def decomp(f):
    g=_rd(f).drop_duplicates(subset=['f_sto','f_dsmev','f_ntc'])
    q=lambda s_,d_,n_: g[(g.f_sto==s_)&(g.f_dsmev==d_)&(g.f_ntc==n_)].vf_min.iloc[0]
    v={c:q(float(c[0]),float(c[1]),float(c[2])) for c in itertools.product([0,1],[0,1],[0,1])}
    b=v[(0,0,0)]
    m={'sto':v[(1,0,0)]-b,'dsm':v[(0,1,0)]-b,'ntc':v[(0,0,1)]-b}
    p={'sd':v[(1,1,0)]-v[(1,0,0)]-v[(0,1,0)]+b,'sn':v[(1,0,1)]-v[(1,0,0)]-v[(0,0,1)]+b,'dn':v[(0,1,1)]-v[(0,1,0)]-v[(0,0,1)]+b}
    t=v[(1,1,1)]-sum(m.values())-sum(p.values())-b
    # peak sto x ntc interaction (dsm full) and its location
    s1=g[g.f_dsmev==1.0].pivot_table(index='f_sto',columns='f_ntc',values='vf_min'); b1=s1.loc[0.0,0.0]
    I={i: s1.loc[i,1.0]-s1.loc[i,0.0]-s1.loc[0.0,1.0]+b1 for i in s1.index if i>0}
    pk=max(I,key=I.get)
    return b,v[(1,1,1)],m,p,t,I[pk],pk
rows=[]
for sc,f,lab in [('de2035','grid_floor_5node_de2035.csv','DE 2035'),('de2040','grid_floor_5node_matched.csv','DE 2040'),('de2050','grid_floor_5node_de2050.csv','DE 2050')]:
    try:
        b,full,m,p,t,pk,pkloc=decomp(f)
        rows.append((lab,b,full,m,p,t))
        tag={'de2035':'ThirtyFive','de2040':'Forty','de2050':'Fifty'}[sc]
        M6('iaPeak'+tag,pk); M6('iaPeakLoc'+tag,100*pkloc,'%.0f')
        _gg=_rd(f).drop_duplicates(subset=['f_sto','f_dsmev','f_ntc']); _s1=_gg[_gg.f_dsmev==1.0].pivot_table(index='f_sto',columns='f_ntc',values='vf_min')
        M6('iaLow'+tag, min(_s1.loc[i,1.0]-_s1.loc[i,0.0]-_s1.loc[0.0,1.0]+_s1.loc[0.0,0.0] for i in _s1.index if i>0))
    except FileNotFoundError: pass
with open(os.path.join(OUT,'tab_decomp.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n")
    f.write("Scenario & corner & all full & sto & DSM & NTC & sto$\\times$DSM & sto$\\times$NTC & DSM$\\times$NTC & three-way\\\\\n\\midrule\n")
    z=lambda v: 0.0 if abs(v)<5e-4 else v   # avoid printing -0.000
    for lab,b,full,m,p,t in rows:
        f.write(f"{lab} & {b:.3f} & {full:.3f} & {z(m['sto']):+.3f} & {z(m['dsm']):+.3f} & {z(m['ntc']):+.3f} & {z(p['sd']):+.3f} & {z(p['sn']):+.3f} & {z(p['dn']):+.3f} & {z(t):+.3f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac6)+'\n'); mac6=[]
# ---------- S12b: node-level experiments (phase 2 files) ----------
mac6=[]
# (b) node-level, near-side (NL storage x link), with binding switch
_pl=PLAN(); _W=_pl['W']; _sk='act_'+_pl['sto_key']; _mk='act_'+_pl['main_ntc_key']; _LM=_pl['main_max']
g2=_rd('node_factorial_de2040_bopt.csv')
if _pl['second_ntc_key']: g2=g2[g2['act_'+_pl['second_ntc_key']]==0]
s2=g2.pivot_table(index=_sk,columns=_mk,values='vf_min')
pv=g2.pivot_table(index=_sk,columns=_mk,values='vf_pv_'+_W)
# weakest solar asset outside the weak node (old set: German solar)
_oth=[n_ for n_ in NODES if n_!=_W]
g2=g2.copy(); g2['vf_pv_other']=g2[[f'vf_pv_{n_}' for n_ in _oth]].min(axis=1)
g2['other_node']=g2[[f'vf_pv_{n_}' for n_ in _oth]].idxmin(axis=1).str[-2:]
de=g2.pivot_table(index=_sk,columns=_mk,values='vf_pv_other')
_onode=g2.loc[g2[_sk].idxmax(),'other_node']
mac6.append(r'\newcommand{\nsOtherAdj}{%s}' % ADJ[_onode])
M6('nsVfPlateau', s2.loc[_pl['sto_pot'],0.0]); M6('nsNlPvTop', pv.loc[_pl['sto_pot'],0.0]); M6('nsNlPvBase', pv.loc[0.0,0.0])
M6('nsDePv', de.loc[0.0,0.0]); M6('nsPlateauFrom', 20, '%d')   # old-set reading, see audit
M6('nsVfLo', s2[0.0].min()); M6('nsVfHi', s2[0.0].max())   # VF_min range along the storage sweep, link at 0
Ins={r_: s2.loc[r_,_LM]-s2.loc[r_,0.0]-s2.loc[0.0,_LM]+s2.loc[0.0,0.0] for r_ in s2.index if r_>0}
M6('nsIntMin', min(Ins.values())); M6('nsIntMax', max(Ins.values()))
_g2all=_rd('node_factorial_de2040_bopt.csv')
M6('nsAllVfLo', _g2all['vf_min'].min()); M6('nsAllVfHi', _g2all['vf_min'].max()); M6('nsAllPremMax', _g2all['premium_bn'].max(), '%.2f')
M6('nsAllN', len(_g2all), '%d')
# (c) frontier locals: reproduction + local interactions
ck=D[('de2040','vf')]
# rows: first attained point, elbow, last attained point (old set: 1, 3, 4)
_hi=[i for i,r in ck.iterrows() if i>0 and r['vf_min']>=r['vf_target']-0.003]
_rows=[_hi[0], _elbow_row(ck)[0], _hi[-1]]
loc={}
for rw in _rows:
    g3=_rd(f'node_local_de2040_row{rw}.csv')
    s3=g3.pivot_table(index='req_'+_pl['sto_key'],columns='req_'+_pl['main_ntc_key'],values='vf_min')
    r_=list(s3.index); c_=list(s3.columns)
    loc[rw]=s3.loc[r_[-1],c_[-1]]-s3.loc[r_[-1],c_[0]]-s3.loc[r_[0],c_[-1]]+s3.loc[r_[0],c_[0]]
    if rw==_rows[0]: M6('vfResolution', abs(s3.loc[0.0,0.0]-ck.loc[rw].vf_min))
    if rw==_rows[1]: M6('linkLocalElbow', (s3.loc[0.0,c_[-1]]-s3.loc[0.0,c_[0]])/((c_[-1]-c_[0])/1000.0))  # per GW
M6('locIntBottom',loc[_rows[0]]); M6('locIntElbow',loc[_rows[1]]); M6('locIntTop',loc[_rows[2]])
# solar assets tied at the elbow (within one value-factor point of the minimum)
_pe=ck.loc[_rows[1],[f'vf_pv_{n_}' for n_ in NODES]].astype(float)
_tied=_pe[_pe<=_pe.min()+0.01]
M6('elbowNTied', len(_tied), '%d'); M6('elbowPvLo', _tied.min()); M6('elbowPvHi', _tied.max())
mac6.append(r'\newcommand{\elbowNTiedW}{%s}' % NW(len(_tied)))
_ge=_rd(f'node_local_de2040_row{_rows[1]}.csv')
_se=_ge.pivot_table(index='req_'+_pl['sto_key'],columns='req_'+_pl['main_ntc_key'],values='vf_min')
_ze=lambda v: 0.0 if abs(v)<5e-4 else v
M6('elbowWStoGain', _ze(_se.loc[max(_se.index),0.0]-_se.loc[0.0,0.0])); M6('elbowWStoStep', max(_se.index)/1000, '%.0f')
# at the last attained point the weak node binds alone
_pt=ck.loc[_rows[2],[f'vf_pv_{n_}' for n_ in NODES]+[f'vf_wind_on_{n_}' for n_ in NODES]+[f'vf_wind_off_{n_}' for n_ in NODES]].astype(float)
_pt=_pt[_pt>0].sort_values()
M6('topGapPts', 100*(_pt.iloc[1]-_pt.iloc[0]), '%.1f')
# substitution at the top: link step at reduced and at own storage, storage step at own link
g3=_rd(f'node_local_de2040_row{_rows[2]}.csv')
s3=g3.pivot_table(index='req_'+_pl['sto_key'],columns='req_'+_pl['main_ntc_key'],values='vf_min')
r_=sorted(s3.index); c_=sorted(s3.columns)
M6('topLinkGainLowSto', s3.loc[r_[0],c_[1]]-s3.loc[r_[0],c_[0]])
_z=lambda v: 0.0 if abs(v)<5e-4 else v
M6('topLinkGainOwnSto', _z(s3.loc[r_[-1],c_[1]]-s3.loc[r_[-1],c_[0]]))
M6('topStoGain', s3.loc[r_[-1],c_[0]]-s3.loc[r_[0],c_[0]]); M6('topStoStep', (r_[-1]-r_[0])/1000, '%.0f')
M6('topLinkStep', (c_[1]-c_[0])/1000, '%.0f')
# (d) far-side test
for f,tag in [(f'farside_de2040_row{_rows[1]}.csv','Elbow'),('farside_de2040_bopt.csv','Bopt')]:
    g4=_rd(f); s4=g4.pivot_table(index='act_sto_mw',columns=_mk,values='vf_min')
    b4=s4.loc[0.0,0.0]; _T=_pl['far_sto_top']; _Mi=_pl['far_sto_mid']
    M6('fsLinkAlone'+tag, s4.loc[0.0,_LM]-b4)
    M6('fsLinkWithSto'+tag, (lambda v: 0.0 if abs(v)<5e-4 else v)(s4.loc[_T,_LM]-s4.loc[_T,0.0]))
    M6('fsIntTop'+tag, s4.loc[_T,_LM]-s4.loc[_T,0.0]-s4.loc[0.0,_LM]+b4)
    M6('fsIntMid'+tag, s4.loc[_Mi,_LM]-s4.loc[_Mi,0.0]-s4.loc[0.0,_LM]+b4)
    if tag=='Elbow':
        q0=g4[(g4.act_sto_mw==0)&(g4[_mk]==0)].iloc[0]; q1=g4[(g4.act_sto_mw==_T)&(g4[_mk]==0)].iloc[0]; q2=g4[(g4.act_sto_mw==_T)&(g4[_mk]==_LM)].iloc[0]
        M6('fsDePvNoSto', q0['vf_pv_DE']); M6('fsDePvSto', q1['vf_pv_DE']); M6('fsNlPvSto', q1['vf_pv_'+_W]); M6('fsNlPvStoLink', q2['vf_pv_'+_W])
        _pq=q1[[f'vf_pv_{n_}' for n_ in NODES]].astype(float); _bq=_pq.idxmin()[-2:]
        mac6.append(r'\newcommand{\fsBindStoAdj}{%s}' % ADJ[_bq]); M6('fsBindStoPv', _pq.min())
        M6('fsDeStoElbow', ck.loc[_rows[1],'sto_mw']/1000, '%.0f')
        M6('fsDropNoSto', 100*(ck.loc[_rows[1],'vf_min']-q0['vf_min']), '%.0f')
        _dn=[100*(float(ck.loc[_rows[1],f'vf_pv_{n_}'])-float(q0[f'vf_pv_{n_}'])) for n_ in NODES if n_ not in ('DE',_W)]
        M6('fsNbrDropLo', min(_dn), '%.1f'); M6('fsNbrDropHi', max(_dn), '%.1f')
        M6('fsDeStoAdd', (_T-ck.loc[_rows[1],'sto_mw'])/1000, '%.0f'); M6('fsDeStoGainPts', 100*(q1['vf_min']-ck.loc[_rows[1],'vf_min']), '%.1f')
    else:
        _pb=g4[[f'vf_pv_{n_}' for n_ in NODES]].astype(float)
        _bb=set(_pb.idxmin(axis=1).str[-2:])
        mac6.append(r'\newcommand{\fsBindBoptAdj}{%s}' % (ADJ[list(_bb)[0]] if len(_bb)==1 else 'mixed'))
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac6)+'\n')
print("S12 macros:",len(mac6))

# ---- S13: gross flexibility bill vs net premium (supervisor C007) ----
# Premium = Delta total annual system cost vs B_opt (model_builder obj_rule):
# annualised cost of added storage/DSM/NTC MINUS savings in fuel, carbon,
# curtailment, backup capacity and re-optimised electrolysis. Gross bill is
# rebuilt from the portfolio columns with the de2040 cost data actually used:
# storage 4h (directed_search fixes K_sto_e = 4*K_sto_p), costs from
# scenario_data.get_scenario('de2040'); DSM capex 10000 EUR/MW/yr
# (data_pipeline default, not overridden by DE scenarios); NTC 322 EUR/MW-km/yr
# x link distances (data_pipeline links).
import sys as _sys
_sys.path.insert(0, _PV)
from scenario_data import get_scenario as _gs
_s40=_gs('de2040'); _cp=_s40['sto_capex_p']+_s40['sto_fom_p']+4*_s40['sto_capex_e']
_dist={'de_cz':399,'de_dk':573,'de_at':464,'de_nl':326,'dk_nl':620,'cz_at':599}
_nd=['','_cz','_dk','_at','_nl']
def _gross(r):
    return (sum(r['sto'+n+'_mw'] for n in _nd)*_cp + sum(r['dsm'+n+'_mw'] for n in _nd)*10000
            + sum(r['ntc_%s_mw'%l]*d for l,d in _dist.items())*322.0)/1e9
_f40=D[('de2040','vf')]; _g0=_gross(_f40.iloc[0])
mac13=[]
for _r,_tag in [(_elbow_row(_f40)[1],'Elbow'),(hits(_f40,'vf_min')[-1],'Top')]:
    _gr=_gross(_r)-_g0
    mac13 += [r'\newcommand{\grossFlex%s}{%.2f}'%(_tag,_gr),
              r'\newcommand{\flexSave%s}{%.2f}'%(_tag,_gr-_r.premium_bn),
              r'\newcommand{\flexSavePct%s}{%.0f}'%(_tag,100*(_gr-_r.premium_bn)/_gr)]
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac13)+'\n')
print("S13 macros:",mac13)

# ---------- S12c: fig_interactions2 (needs S12b) ----------
# figure: (a) near-side vs far-side interaction; (b) binding switch
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9))
g4=_rd(f'farside_de2040_row{_rows[1]}.csv'); s4=g4.pivot_table(index='act_sto_mw',columns=_mk,values='vf_min'); b4=s4.loc[0.0,0.0]
xs=[r_/1000 for r_ in s4.index if r_>0]; ys=[s4.loc[r_,_LM]-s4.loc[r_,0.0]-s4.loc[0.0,_LM]+b4 for r_ in s4.index if r_>0]
_LN=_pl['main_ntc_key'][4:-3].upper().replace('_','-')
ax[0].plot(xs,ys,'-o',ms=4,lw=1.5,color='#D55E00',label='far side: German storage')
xs2=[r_/1000 for r_ in s2.index if r_>0]; ys2=[Ins[r_] for r_ in s2.index if r_>0]
ax[0].plot(xs2,ys2,'-s',ms=4,lw=1.5,color='#0072B2',label=f'near side: {ADJ[_W]} storage')
ax[0].axhline(0,color='gray',lw=0.7); ax[0].axhspan(-0.005,0.005,color='gray',alpha=0.15)
ax[0].set_xlabel('storage added [GW]'); ax[0].set_ylabel(f'interaction with {_LN} link [VF units]')
ax[0].set_title('(a) storage $\\times$ link interaction',fontsize=9,loc='left'); ax[0].legend(frameon=False,fontsize=7.5)
ax[1].plot([r_/1000 for r_ in s2.index],s2[0.0],'-o',ms=4,lw=1.5,color='#000000',label=r'$VF_{\min}$')
ax[1].plot([r_/1000 for r_ in s2.index],pv[0.0],'--',lw=1.3,color='#000000',label=f'{ADJ[_W]} solar')
ax[1].plot([r_/1000 for r_ in s2.index],de[0.0],':',lw=1.3,color='#D55E00',label=f'weakest other solar ({ADJ[_onode]})')
ax[1].set_xlabel(f'{ADJ[_W]} storage [GW], rest at cost optimum'); ax[1].set_ylabel('value factor')
ax[1].set_title(f'(b) {ADJ[_W]} storage alone',fontsize=9,loc='left'); ax[1].legend(frameon=False,fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_interactions2.pdf')); plt.close(fig)
print("OK: gen/ written")

# ---- S14: TYNDP system components added for the rerun (S.103-S.106) ----
import json
# Fixed model inputs taken from TYNDP 2024 outputs (tyndp_extras.json,
# tyndp_borders_dsr.json); same arithmetic as tyndp_fixes.py.
_TD = os.path.join(_PV, 'tyndp 2040')
_X = json.load(open(os.path.join(_TD, 'tyndp_extras.json')))
_XB = json.load(open(os.path.join(_TD, 'tyndp_borders_dsr.json')))
_SCS = ['nt2030', 'de2035', 'de2040', 'de2050']
_GAS = {'nt2030': 8.845, 'de2035': 0.5 * (5.857 + 10.438), 'de2040': 10.438, 'de2050': 14.804}
_FOS = {'nt2030': 0.82, 'de2035': 0.545, 'de2040': 0.35, 'de2050': 0.02}
_TAU = {'nt2030': 70.0, 'de2035': 100.5, 'de2040': 123.0, 'de2050': 168.0}
def _hydro_gw(sc, n):
    h = _X[sc][n]['hydro']; t = h['res_mw'] + h['open_turb_mw'] + h['closed_turb_mw']
    return (t + (1149.0 if n == 'AT' and t > 0 else 0.0)) / 1e3
rows14 = []
def _r(lab, f, fmt):
    rows14.append([lab] + [fmt % f(sc) if f(sc) is not None else '--' for sc in _SCS])
_r('Offshore hubs DE / DK / NL [GW]', lambda sc: None if sc == 'nt2030' else 0, '%s')
rows14[-1][1:] = ['--'] + [' / '.join('%.1f' % (_X[sc][n]['hub']['mw'] / 1e3) for n in ('DE', 'DK', 'NL')) for sc in _SCS[1:]]
rows14.append(['Reservoir + pumped hydro DE / CZ / AT [GW]'] + [' / '.join('%.1f' % _hydro_gw(sc, n) for n in ('DE', 'CZ', 'AT')) for sc in _SCS])
_r('Other generation, flat [TWh]', lambda sc: sum(sum(_X[sc][n]['other_gwh'].values()) for n in NODES) / 1e3, '%.0f')
_r('Hydrogen CCGT [GW]', lambda sc: sum(_X[sc][n]['h2ccgt_mw'] for n in NODES) / 1e3, '%.1f')
_r('Existing gas plants [GW]', lambda sc: sum(_X[sc][n]['gas_mw']['ccgt'] + _X[sc][n]['gas_mw']['ocgt'] for n in NODES) / 1e3, '%.1f')
_r('Oil plants [GW]', lambda sc: sum(_XB[sc]['oil_mw'].values()) / 1e3, '%.1f')
_r('Demand response, load reduction [GW]', lambda sc: sum(_X[sc][n]['dsr_mw'] for n in NODES) / 1e3, '%.1f')
_r('CCGT fuel + CO$_2$ cost [\\euro/MWh]', lambda sc: _GAS[sc] * 3.6 / 0.546 + 0.37 * _FOS[sc] * _TAU[sc], '%.0f')
_r('Hydrogen price [\\euro/MWh$_{\\mathrm{H_2}}$]', lambda sc: _X[sc]['_h2_price_eur_mwh'], '%.0f')
rows14.append(['Border prices DE / DK / AT / NL [\\euro/MWh]'] + [' / '.join('%.0f' % _XB[sc]['borders'][b]['mean'] for b in ('DE', 'DK', 'AT', 'NL')) for sc in _SCS])
with open(os.path.join(OUT, 'tab_tyndp.tex'), 'w') as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n & NT 2030 & DE 2035 & DE 2040 & DE 2050\\\\\n\\midrule\n")
    for r in rows14: f.write(' & '.join(r) + '\\\\\n')
    f.write("\\bottomrule\n\\end{tabular}\n")
mac14 = [r'\newcommand{\hubDEForty}{%.1f}' % (_X['de2040']['DE']['hub']['mw'] / 1e3),
         r'\newcommand{\hubTWhDEForty}{%.0f}' % _X['de2040']['DE']['hub']['twh'],
         r'\newcommand{\hubCFForty}{%.2f}' % (sum(_X['de2040']['DE']['hub']['profile']) / len(_X['de2040']['DE']['hub']['profile'])),
         r'\newcommand{\phsDEForty}{%.1f}' % _hydro_gw('de2040', 'DE'),
         r'\newcommand{\hTwoCcgtDEForty}{%.1f}' % (_X['de2040']['DE']['h2ccgt_mw'] / 1e3),
         r'\newcommand{\dsrDEForty}{%.1f}' % (_X['de2040']['DE']['dsr_mw'] / 1e3),
         r'\newcommand{\ccgtCostForty}{%.0f}' % (_GAS['de2040'] * 3.6 / 0.546 + 0.37 * _FOS['de2040'] * _TAU['de2040']),
         r'\newcommand{\borderNLForty}{%.0f}' % _XB['de2040']['borders']['NL']['mean'],
         r'\newcommand{\dsrBandsForty}{%s}' % ', '.join('%.0f\\,\\%%' % (100 * s) for s in _XB['de2040']['dsr']['shares'])]
open(os.path.join(OUT, 'numbers.tex'), 'a').write('\n'.join(mac14) + '\n')
print('S14 TYNDP inputs table + macros:', len(mac14))

# ---- S15: TYNDP battery benchmark vs the least-cost frontier ----
# tyndp_battery_benchmark.py: B_opt with storage power fixed at the TYNDP
# 2024 battery fleet (4-h, utility+residential), everything else re-optimised.
# Compare its VF_min/premium with what the weakest-link frontier pays for the
# same VF_min (linear interpolation between attained frontier points).
if PAPER_SET == 'old':
    raise PinMissing('the battery benchmark was run with TYNDP_FIXES=all only')
_bm = pd.read_csv(os.path.join(BASE,'tyndp_battery_benchmark_final.csv'))
def _front_prem(sc, v):
    if (sc,'vf') not in D: raise KeyError(('benchmark needs frontier', sc))
    d=D[(sc,'vf')]; hh=[d.iloc[0]]+hits(d,'vf_min')
    xs=[r['vf_min'] for r in hh]; ys=[r['premium_bn'] for r in hh]
    if v <= xs[0]: return 'below'      # TYNDP fleet does worse than B_opt
    if v > xs[-1]: return 'beyond'     # beyond the last attained frontier point
    return float(np.interp(v, xs, ys))
mac15=[]; rows15=[]
for sc in SC:
    q=_bm[_bm.scenario==sc].set_index('case')
    if 'TYNDP_batteries' not in q.index: continue
    t=q.loc['TYNDP_batteries']; b0=q.loc['B_opt']
    sto=sum(t[f'sto_{n.lower()}_mw'] for n in NODES)/1e3
    sto0=sum(b0[f'sto_{n.lower()}_mw'] for n in NODES)/1e3
    fp=_front_prem(sc, t['vf_min'])
    ratio = (t['premium_bn']/fp) if isinstance(fp,float) and fp>1e-6 else None
    rows15.append((LBL[sc], sto0, sto, D[(sc,'vf')].iloc[0]['vf_min'], t['vf_min'], t['premium_bn'], fp, ratio))
    m0=MAC[sc]
    mac15 += [r'\newcommand{\bmBaseVf%s}{%.3f}'%(m0,D[(sc,'vf')].iloc[0]['vf_min'])]
    for _n in ('de','dk','nl'):
        mac15 += [r'\newcommand{\bmSto%s%s}{%.1f}'%(_n.capitalize(),m0,t[f'sto_{_n}_mw']/1e3),
                  r'\newcommand{\boptSto%s%s}{%.1f}'%(_n.capitalize(),m0,b0[f'sto_{_n}_mw']/1e3)]
    mac15 += [r'\newcommand{\bmVf%s}{%.3f}'%(m0,t['vf_min']), r'\newcommand{\bmPrem%s}{%.2f}'%(m0,t['premium_bn']),
              r'\newcommand{\bmSto%s}{%.0f}'%(m0,sto)]
    if ratio: mac15.append(r'\newcommand{\bmRatio%s}{%.0f}'%(m0,ratio))
    else: mac15.append(r'\newcommand{\bmRatio%s}{--}'%m0)   # fleet below the cost optimum: no ratio
with open(os.path.join(OUT,'tab_benchmark.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
    f.write("Scenario & \\multicolumn{2}{c}{Battery power [GW]} & \\multicolumn{2}{c}{$\\vfmin$} & \\multicolumn{2}{c}{$\\Pi$ [bn\\euro/yr]} & ratio\\\\\n")
    f.write(" & $\\Bopt$ & TYNDP & $\\Bopt$ & TYNDP & TYNDP & frontier & \\\\\n\\midrule\n")
    for r in rows15:
        fp={'below':'$^{a}$','beyond':'$^{b}$'}.get(r[6], None) if isinstance(r[6],str) else f"{r[6]:.2f}"
        ra='--' if r[7] is None else f"{r[7]:.0f}$\\times$"
        f.write(f"{r[0]} & {r[1]:.0f} & {r[2]:.0f} & {r[3]:.3f} & {r[4]:.3f} & {r[5]:.2f} & {fp} & {ra}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac15)+'\n')
print('S15 benchmark rows:', len(rows15))

# ---- S16: support-scheme view (subsidy bill, avoided subsidy, net benefit) ----
# Columns written by directed_search.py (S.103): subsidy_bill_bn = CfD-style
# top-up making every VRE whole; avoided_subsidy_bn vs B_opt;
# net_benefit_mcf{0,30}_bn = (1+MCF)*avoided - premium.
mac16=[]
with open(os.path.join(OUT,'tab_welfare.tex'),'w') as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    f.write("Scenario & $\\vfmin$ & $\\Pi$ & subsidy bill & avoided & \\multicolumn{2}{c}{net benefit}\\\\\n")
    f.write(" & & & & & MCF 0 & MCF 0.3\\\\\n")
    f.write(" & & \\multicolumn{5}{c}{[bn\\euro/yr]}\\\\\n\\midrule\n")
    for sc in SC:
        d=D[(sc,'vf')]
        for r in [d.iloc[0]]+hits(d,'vf_min'):
            f.write(f"{LBL[sc] if r.name==0 else ''} & {r['vf_min']:.3f} & {r['premium_bn']:.2f} & {r['subsidy_bill_bn']:.2f} & "
                    f"{r['avoided_subsidy_bn']:.2f} & {r['net_benefit_mcf0_bn']:.2f} & {r['net_benefit_mcf30_bn']:.2f}\\\\\n")
        if sc!=SC[-1]: f.write("\\midrule\n")
        hh=hits(d,'vf_min'); b=d.iloc[0]; m0=MAC[sc]
        best=max(hh, key=lambda r: r['net_benefit_mcf30_bn'])
        mac16 += [r'\newcommand{\subsBill%s}{%.2f}'%(m0,b['subsidy_bill_bn']),
                  r'\newcommand{\netBestThirty%s}{%.2f}'%(m0,best['net_benefit_mcf30_bn']),
                  r'\newcommand{\netBestVf%s}{%.3f}'%(m0,best['vf_min']),
                  r'\newcommand{\nPosNet%s}{%d}'%(m0,sum(1 for r in hh if r['net_benefit_mcf0_bn']>0))]
    f.write("\\bottomrule\n\\end{tabular}\n")
open(os.path.join(OUT,'numbers.tex'),'a').write('\n'.join(mac16)+'\n')
print('S16 welfare macros:', len(mac16))

# ---- S17: nodal average prices at B_opt and at the last attained point ----
with open(os.path.join(OUT,'tab_prices.tex'),'w') as f:
    f.write("\\begin{tabular}{l"+"rr"*len(NODES)+"}\n\\toprule\n")
    f.write("Scenario & "+" & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for n in NODES)+"\\\\\n")
    f.write(" & "+" & ".join(["$\\Bopt$ & end"]*len(NODES))+"\\\\\n\\midrule\n")
    for sc in SC:
        d=D[(sc,'vf')]; b=d.iloc[0]; e=hits(d,'vf_min')[-1]
        f.write(LBL[sc]+" & "+" & ".join(f"{b['base_price_'+n]:.0f} & {e['base_price_'+n]:.0f}" for n in NODES)+"\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print('S17 price table ok')
