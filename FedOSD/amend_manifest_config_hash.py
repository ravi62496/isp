"""Amendment 1: recompute the manifest config_hash WITHOUT the seed.

Why: build_manifest_confirm.py wrote a config_hash that changes with the seed, so the
pre-registered check 'one config hash per arm' (evaluate_confirmation.py) can never pass for
10 different seeds. This script reads NO metrics. It only reads each result file's stored
command line, removes '--seed N' and '--device N', hashes the rest, and verifies that every
arm used exactly one command line across all seeds and stages.

Usage:  python amend_manifest_config_hash.py [SRC.csv] [DST.csv]
"""
import csv, hashlib, json, re, sys, collections

SRC = sys.argv[1] if len(sys.argv) > 1 else 'RUN_MANIFEST_CONFIRM.csv'
DST = sys.argv[2] if len(sys.argv) > 2 else 'RUN_MANIFEST_CONFIRM_amended.csv'

def sha256_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def find_cli(d):
    for src in (d, d.get('provenance') or {}):
        for k in ('cli', 'command', 'cmd', 'cli_args'):
            v = src.get(k)
            if isinstance(v, list):
                v = ' '.join(map(str, v))
            if isinstance(v, str) and v.strip():
                return v
    raise KeyError('no command line stored in the result file; top-level keys: %s; provenance keys: %s'
                   % (sorted(d)[:25], sorted((d.get('provenance') or {}))[:25]))

def normalize(cli):
    cli = re.sub(r'--seed\s+\d+', ' ', cli)
    cli = re.sub(r'--device\s+\d+', ' ', cli)
    return ' '.join(cli.split())

with open(SRC, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

path_col = next(c for c in ('path', 'absolute_file_path', 'file_path') if c in fields)
by_arm = collections.defaultdict(set)
commits = set()
for r in rows:
    with open(r[path_col], encoding='utf-8') as jf:
        norm = normalize(find_cli(json.load(jf)))
    r['config_hash'] = hashlib.sha256(norm.encode('utf-8')).hexdigest()
    by_arm[r['arm']].add(norm)
    commits.add(r.get('commit') or r.get('git_commit') or '')

print('Cells:', len(rows), '| distinct commits:', sorted(commits))
bad = False
for arm in sorted(by_arm):
    n = len(by_arm[arm])
    print('%-8s distinct normalized command lines: %d' % (arm, n))
    for c in sorted(by_arm[arm]):
        print('    ' + c)
    bad |= n != 1
if bad or len(commits) != 1:
    print('ABORT: an arm used more than one command line (or more than one commit). Do not continue.')
    sys.exit(1)

with open(DST, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print('Wrote', DST)
print('sha256 ' + SRC + ': ' + sha256_file(SRC))
print('sha256 ' + DST + ': ' + sha256_file(DST))
