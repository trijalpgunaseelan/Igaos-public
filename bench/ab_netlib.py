import subprocess, glob, os, csv, re, sys
B="./build/igaos"; TL=30
files=sorted(glob.glob("benchmarks/netlib/*"))
def run(p,extra):
    try:
        cp=subprocess.run([B,p,"-q","--time-limit",str(TL)]+extra,capture_output=True,text=True,timeout=TL+60)
        o=cp.stdout+cp.stderr
    except subprocess.TimeoutExpired: return("timeout","")
    st=re.search(r"status=(\S+)",o); ob=re.search(r"obj=(\S+)",o)
    return (st.group(1) if st else "?", ob.group(1) if ob else "")
rows=[]
for f in files:
    a=run(f,[]); b=run(f,["--scale-with-q"])
    rows.append((os.path.basename(f),a[0],a[1],b[0],b[1]))
    print(rows[-1],flush=True)
with open("bench/results_netlib_scaleq_ab.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["instance","base_status","base_obj","scaleq_status","scaleq_obj"]); w.writerows(rows)
na=sum(1 for r in rows if r[1]=="optimal"); nb=sum(1 for r in rows if r[3]=="optimal")
diff=[r for r in rows if r[1]!=r[3] or (r[2] and r[4] and r[2]!=r[4])]
print("BASE",na,"SCALEQ",nb,"DIFFS",len(diff))
for d in diff: print("DIFF",d)
