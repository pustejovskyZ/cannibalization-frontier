from pyxlsb import open_workbook
import json, sys
f=sys.argv[1]; out=sys.argv[2]
keep=None; data=[]; cat=cty=None
with open_workbook(f) as wb, wb.get_sheet('Hourly Market Data emarket' if 'emarket' in str(wb.sheets) else 'Hourly Market Data') as s:
    for i,r in enumerate(s.rows()):
        v=[c.v for c in r]
        if i==10: cat=v
        elif i==11:
            cty=v
            keep=[j for j in range(len(v)) if isinstance(v[j],str) and v[j] in ('DE00','NL00','AT00','CZ00','DKW1','DKE1')
                  and isinstance(cat[j],str) and any(k in cat[j] for k in ('Demand Side','Hydrogen CCGT','Price','price','Marginal','Others','Pump','Reservoir','Battery'))]
        elif i>11 and keep:
            data.append([v[j] for j in keep])
json.dump({'cols':[(cty[j],cat[j]) for j in keep],'data':data},open(out,'w'))
print(len(keep),len(data)); print([(cty[j],cat[j]) for j in keep][:60])
