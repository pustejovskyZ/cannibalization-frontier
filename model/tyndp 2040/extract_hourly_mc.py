from pyxlsb import open_workbook
import json, sys
f=sys.argv[1]; out=sys.argv[2]
want={'DE00','NL00','DKW1','AT00','BE00','FR00','UK00','NOS0','NOM1','NON1','NOS1','NOS2','NOS3','SE03','SE04','CH00','ITN1','PL00','LUG1'}
keep=None; data=[]; cat=None
with open_workbook(f) as wb, wb.get_sheet('Hourly Market Data emarket') as s:
    for i,r in enumerate(s.rows()):
        v=[c.v for c in r]
        if i==10: cat=v
        elif i==11:
            keep=[j for j in range(len(v)) if v[j] in want and isinstance(cat[j],str) and cat[j].startswith('Marginal Cost')]
            names=[v[j] for j in keep]
        elif i>11 and keep:
            data.append([v[j] for j in keep])
json.dump({'names':names,'data':data},open(out,'w')); print(names, len(data))
