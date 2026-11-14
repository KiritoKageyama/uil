"""Debug: manually run the Miner on the journal to see what it discovers."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.uil_miner import UIL_Miner

data = json.load(open('results/uil_journal_v19.json'))
normal = [e for e in data if e['classifier'] != 'FAILSAFE']

print(f"Total entries: {len(normal)}")
print(f"Entries with regret > 0: {sum(1 for e in normal if e['regret'] > 0)}")

miner = UIL_Miner(min_entries=15)  # Lower threshold for testing

# Within-dataset mining
rules = miner.mine_within_dataset(normal, "hard_noisy_100f", stress_level=0.8)
r2 = miner.get_last_r2_score()
print(f"\nWithin-dataset mining:")
print(f"  R2: {r2:.4f}")
print(f"  Rules found: {len(rules)}")

for i, rule in enumerate(rules):
    conds = "; ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in rule['conditions'])
    print(f"\n  Rule {i+1}: {conds}")
    print(f"    Predicted regret: {rule['predicted_regret']:.4f}")
    print(f"    Confidence: {rule['confidence']:.4f}")
    print(f"    Evidence: {rule['evidence_count']}")
    print(f"    Mining level: {rule.get('mining_level', 'unknown')}")
    
    # Would it be promoted?
    if rule['confidence'] >= 0.75 and rule['evidence_count'] >= 10:
        print(f"    >>> WOULD BE PROMOTED TO PRINCIPLE")
    else:
        reason = []
        if rule['confidence'] < 0.75:
            reason.append(f"confidence {rule['confidence']:.2f} < 0.75")
        if rule['evidence_count'] < 10:
            reason.append(f"evidence {rule['evidence_count']} < 10")
        print(f"    >>> NOT promoted: {', '.join(reason)}")

# Cross-dataset mining  
print(f"\nCross-dataset mining:")
cross_rules = miner.mine_cross_dataset(normal, stress_level=0.8)
r2_cross = miner.get_last_r2_score()
print(f"  R2: {r2_cross:.4f}")
print(f"  Rules found: {len(cross_rules)}")
for i, rule in enumerate(cross_rules[:5]):
    conds = "; ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in rule['conditions'])
    print(f"\n  Rule {i+1}: {conds}")
    print(f"    Predicted regret: {rule['predicted_regret']:.4f}")
    print(f"    Confidence: {rule['confidence']:.4f}")
    print(f"    Evidence: {rule['evidence_count']}")
    if rule['confidence'] >= 0.75 and rule['evidence_count'] >= 10:
        print(f"    >>> WOULD BE PROMOTED TO PRINCIPLE")
