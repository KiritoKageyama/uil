import json
data = json.load(open('results/uil_journal_v19.json'))
normal = [e for e in data if e['classifier'] != 'FAILSAFE']
print(f'Total entries: {len(normal)}')
print(f'Entries with regret > 0: {sum(1 for e in normal if e["regret"] > 0)}')
print(f'Entries with investigation: {sum(1 for e in normal if e.get("investigation") is not None)}')
print()

for fold in sorted(set(e['fold'] for e in normal)):
    fold_entries = [e for e in normal if e['fold'] == fold]
    clf = fold_entries[0]['classifier']
    print(f'--- Fold {fold} ({clf}) ---')
    for e in fold_entries:
        s = e['system_stress']
        r = e['regret']
        a = e['outcome']['accuracy']
        bar = '#' * int(s * 40)
        reg = f'REGRET={r:.3f}' if r > 0 else ''
        inv = 'INVESTIGATE' if e.get('investigation') else ''
        print(f'  Stage {e["stage"]:2d}: stress={s:.3f} acc={a:.3f} {reg:15s} {inv:12s} |{bar}')
    print()

# Why no principles?
print('--- WHY NO PRINCIPLES? ---')
print(f'Total entries: {len(normal)}')
print(f'Miner min_entries threshold: 30')
print(f'Entries with varied regret (>0): {sum(1 for e in normal if e["regret"] > 0)}/{len(normal)}')

# Check if action.penalty varies enough
penalties = [e['action']['penalty'] for e in normal]
print(f'Penalty range: {min(penalties):.3f} to {max(penalties):.3f}')
print(f'Unique penalties: {len(set(round(p, 2) for p in penalties))}')

# Check regret variation
regrets = [e['regret'] for e in normal]
print(f'Regret range: {min(regrets):.4f} to {max(regrets):.4f}')
print(f'Non-zero regrets: {sum(1 for r in regrets if r > 0)}')
