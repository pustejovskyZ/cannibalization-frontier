"""Recreate every derived TYNDP input of the model from the raw TYNDP 2024 files
(about 20 minutes). Run inside this folder after downloading the raw files
(see the README):

    python extract_all.py            -> overwrites the JSON inputs
    python extract_all.py --check    -> rebuilds into _check/ and compares

Steps:
 1. extract_hourly_mc_nt2030.py / extract_hourly_mc.py / extract_hourly_dsr.py:
    hourly marginal costs and demand response from the 'Hourly Market Data'
    sheets into _hourly/ (intermediate files)
 2. extract_borders_dsr.py -> tyndp_borders_dsr.json (border prices, DSR
    bands, oil plants)
 3. extract_tyndp_extras.py -> tyndp_extras.json (offshore hubs, hydro,
    other generation, hydrogen CCGTs, gas fleet, demand response, batteries)
 4. extract_scenarios_ev.py -> extracted_tyndp_scenarios.json,
    extracted_nt2030.json, ev_by_year.json
"""
import os, sys, json, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
D = {'de2035': 'DE2035CY2009/MMStandardOutputFile_DE2035_Plexos_CY2009_v11_SoS.xlsb',
     'de2040': 'MMStandardOutputFile_DE2040_Plexos_CY2009_v11_SoS.xlsb',
     'de2050': 'DE2050CY2009/MMStandardOutputFile_DE2050_Plexos_CY2009_v11_SoS.xlsb'}
NT = 'MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx'
OUTS = ['tyndp_borders_dsr.json', 'tyndp_extras.json', 'extracted_tyndp_scenarios.json',
        'extracted_nt2030.json', 'ev_by_year.json']


def run(*a, env=None):
    print('>', ' '.join(a[1:]), flush=True)
    subprocess.check_call(list(a), cwd=HERE, env=env)


def main():
    check = '--check' in sys.argv
    tmp = os.path.join(HERE, '_hourly'); os.makedirs(tmp, exist_ok=True)
    dst = os.path.join(HERE, '_check') if check else HERE
    os.makedirs(dst, exist_ok=True)
    env = dict(os.environ, TYNDP_TMP=tmp)
    run(PY, 'extract_hourly_mc_nt2030.py', NT, os.path.join(tmp, 'mcnt.json'))
    for sc, f in D.items():
        y = sc[2:]
        run(PY, 'extract_hourly_mc.py', f, os.path.join(tmp, f'mc{y}.json'))
        run(PY, 'extract_hourly_dsr.py', f, os.path.join(tmp, f'hmd{y}.json'))
    run(PY, 'extract_borders_dsr.py', '.', os.path.join(dst, 'tyndp_borders_dsr.json'), env=env)
    run(PY, 'extract_tyndp_extras.py', '.', os.path.join(dst, 'tyndp_extras.json'))
    if check:
        import extract_scenarios_ev as X
        new = X.build(HERE)
        for f, d in new.items():
            json.dump(d, open(os.path.join(dst, f), 'w'), indent=1)
        for f in OUTS:
            same = json.load(open(os.path.join(dst, f))) == json.load(open(os.path.join(HERE, f)))
            print(f, 'identical' if same else 'DIFFERENT')
    else:
        run(PY, 'extract_scenarios_ev.py')
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
