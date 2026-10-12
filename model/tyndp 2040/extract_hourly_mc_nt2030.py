import openpyxl, json, sys, time
t=time.time()
f=sys.argv[1]; out=sys.argv[2]
want={'DE00','NL00','DKW1','DKE1','AT00','BE00','FR00','UK00','NOS0','NOM1','NON1','SE03','SE04','CH00','ITN1','PL00','LUG1','CZ00'}
ws=openpyxl.load_workbook(f,read_only=True,data_only=True)['Hourly Market Data']
cat=cty=None; keep=None; data=[]
for i,r in enumerate(ws.iter_rows(values_only=True)):
    if i==10: cat=r
    elif i==11:
        cty=r
        keep=[j for j in range(len(r)) if r[j] in want and isinstance(cat[j],str) and (cat[j].startswith('Marginal Cost') or cat[j].startswith('Demand Side Response'))]
    elif i>12 and keep:
        data.append([r[j] for j in keep])
json.dump({'cols':[(cty[j],cat[j]) for j in keep],'data':data},open(out,'w')); print(len(keep),len(data),time.time()-t)
