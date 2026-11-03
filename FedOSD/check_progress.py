import glob, re, pathlib, collections

# ---- CONFIG: edit to match your repo ----
RESULTS_GLOB = 'UnlearningTask/**/*.json'     # matches all result json files in repo
FINAL_METHODS = {'FedOSD', 'FedAvg', 'hard', 'one_sided_hard'}  # method folder names
FIG_DIR = 'figures'
# -----------------------------------------

ROOT = pathlib.Path('.')
CODE = [p for p in ROOT.rglob('*.py') if 'check_progress' not in p.name]
README = ROOT / 'README.md'

def code_has(pattern):
    rx = re.compile(pattern, re.I)
    for p in CODE:
        try:
            if rx.search(p.read_text(errors='ignore')):
                return True
        except OSError:
            pass
    return False

def readme_has(pattern):
    return README.exists() and re.search(pattern, README.read_text(errors='ignore'), re.I) is not None

def seeds_per_method():
    out = collections.defaultdict(set)
    for f in glob.glob(RESULTS_GLOB, recursive=True):
        p = pathlib.Path(f)
        m = re.search(r'seed(\d+)', p.stem, re.I)
        if m:
            seed = int(m.group(1))
            s = str(p).lower()
            out[p.parent.name].add(seed)
            if 'retrain' in s:
                out['retrain'].add(seed)
            if 'proj_one_sided' in s or 'one_sided' in s:
                out['one_sided_hard'].add(seed)
            if 'proj_hybrid' in s or 'hybrid' in s:
                out['hybrid'].add(seed)
            if 'wm_conflict' in s or 'conflict' in s:
                out['conflict'].add(seed)
            if 'wm_drop' in s or 'drop' in s:
                out['drop'].add(seed)
            if 'wm_deficit' in s or 'deficit' in s:
                out['deficit'].add(seed)
            if 'lam100000' in s or 'hard' in s:
                out['hard'].add(seed)
            if ('lam20' in s and 'eta0.4' in s) or 'adaptive' in s:
                out['adaptive'].add(seed)
    return out

def newest(paths):
    times = [pathlib.Path(p).stat().st_mtime for p in paths]
    return max(times) if times else 0

sp = seeds_per_method()
final = {m: s for m, s in sp.items() if m in FINAL_METHODS}
figs = glob.glob(FIG_DIR + '/*.png')
res = glob.glob(RESULTS_GLOB, recursive=True)

checks = [
    ('A1', code_has('pre_acc|baseline_acc|acc_before')),
    ('A2', code_has('retrain') and any('retrain' in m.lower() for m in sp)),
    ('A3', code_has('target_class_recall|forget_acc')),
    ('A4', code_has('membership|mia_')),
    ('A5', code_has('max_memory_allocated')),
    ('A6', 'MANUAL'),
    ('A7', any(ROOT.rglob('test_sign*.py'))),
    ('B1', (ROOT / 'diagnostics' / 'overlap_table.csv').exists()),
    ('B2', code_has('d_norm')),
    ('C1', 'MANUAL'),
    ('C2', bool(final) and all(len(s) >= 10 for s in final.values())),
    ('C3', (ROOT / 'SUCCESS_CRITERIA.md').exists()),
    ('C4', 'MANUAL'),
    ('D1', any('conflict' in m for m in sp) and any('drop' in m for m in sp)),
    ('D2', code_has('w_ema|ema_rate')),
    ('D3', 'MANUAL'),
    ('D4', code_has('d_norm_cap')),
    ('D5', any('hybrid' in m and len(s) >= 10 for m, s in sp.items())),
    ('E1', code_has('dirichlet')),
    ('E2', code_has('FedEraser') or code_has('fedavg_ga')),
    ('E3', 'MANUAL'),
    ('F1', not readme_has('rigorous|publication|Floor Lifted|132')),
    ('F2', 'MANUAL'),
    ('F3', bool(figs) and newest(figs) >= newest(res)),
]

for cid, ok in checks:
    status = ok if isinstance(ok, str) else ('PASS' if ok else 'MISSING')
    print(cid, status)
