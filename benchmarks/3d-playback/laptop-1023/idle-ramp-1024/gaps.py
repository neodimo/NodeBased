import json,sys
ev=json.load(open(sys.argv[1]))
t0=ev[0][0]
h=[(t-t0) for t,th,l in ev if l=='handover']
g=[(round(h[i+1],2),round((h[i+1]-h[i])*1000)) for i in range(len(h)-1) if (h[i+1]-h[i])*1000>54]
print(len(h),'handovers; gaps>54 (time s, ms):',g)
