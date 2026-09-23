#!/usr/bin/env python3
"""Run the IIS deletion filter over the Netlib infeasible set.

One row per instance: how many members the filter isolated, out of how many
rows and columns the model has, how many LP solves it cost, and whether the
result is proven irreducible or was cut short by the time limit.
"""
import csv, os, re, subprocess, sys

BIN = './build/igaos'
DIR = 'benchmarks/netlib_infeasible'
OUT = 'bench/results_iis.csv'
LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0

hdr = re.compile(r'^IIS: (\d+) members \((\d+) rows, (\d+) bounds\) '
                 r'from (\d+) LP solves in ([\d.]+)s -- (.*)$', re.M)
dim = re.compile(r'rows\s+(\d+).*?cols\s+(\d+)', re.S)

rows = []
for f in sorted(os.listdir(DIR)):
    if not f.endswith('.mps'):
        continue
    name = f[:-4]
    try:
        p = subprocess.run([BIN, os.path.join(DIR, f), '--iis',
                            '--iis-time-limit', str(LIMIT)],
                           capture_output=True, text=True, timeout=LIMIT + 120)
    except subprocess.TimeoutExpired:
        rows.append({'instance': name, 'status': 'timeout'})
        print(f'{name:16s} timeout')
        continue
    out = p.stdout
    if 'status=infeasible' not in out:
        st = re.search(r'status=(\w+)', out)
        rows.append({'instance': name, 'status': st.group(1) if st else 'unknown'})
        print(f'{name:16s} not infeasible -- skipped')
        continue
    m = hdr.search(out)
    if not m:
        rows.append({'instance': name, 'status': 'no_iis'})
        print(f'{name:16s} no IIS emitted')
        continue
    d = dim.search(out)
    rows.append({
        'instance': name, 'status': 'infeasible',
        'model_rows': d.group(1) if d else '', 'model_cols': d.group(2) if d else '',
        'iis_members': m.group(1), 'iis_rows': m.group(2), 'iis_bounds': m.group(3),
        'lp_solves': m.group(4), 'seconds': m.group(5),
        'irreducible': 'yes' if m.group(6).startswith('irreducible') else 'no',
    })
    print(f'{name:16s} {m.group(1):>5s} members  {m.group(4):>6s} solves  '
          f'{m.group(5):>7s}s  {m.group(6)}')

cols = ['instance', 'status', 'model_rows', 'model_cols', 'iis_members',
        'iis_rows', 'iis_bounds', 'lp_solves', 'seconds', 'irreducible']
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, '') for c in cols})

ok = [r for r in rows if r.get('irreducible') == 'yes']
print(f'\n{len(ok)} / {len(rows)} instances yielded a proven-irreducible IIS')
print(f'written to {OUT}')
