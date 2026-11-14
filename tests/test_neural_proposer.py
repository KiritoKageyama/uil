"""Tests for the offline own-transformer proposer (R1 §5.4, neural_proposer.py).

Covers: tokenization, forward-pass shape, gradient learnability on a
separable synthetic task, open-box save/load round-trip, distill-out
explain(), and the generate-and-rank facade producing proposer="neural"
MutationProposals through LLMMutator's offline path.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from figpso_uil.cognition.neural_proposer import (
    TinyTransformer, NeuralProposer, tokenize_tree, build_vocab,
    build_training_samples,
)


def test_tokenize():
    tree = {"op": "sigmoid", "children": [{"feature": 3}]}
    toks = tokenize_tree(tree)
    assert toks == ["sigmoid", "feat"], toks
    tree2 = {"op": "sub", "children": [{"var": "y_true"}, {"const": 0.1}]}
    assert tokenize_tree(tree2) == ["sub", "var:y_true", "c_small"]
    print("  tokenize OK")


def test_learns_separable():
    """Model must learn a shape signal: trees with 'good' > trees with 'bad'."""
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(60):
        # label 1: contains 'sigmoid'; label 0: contains 'div' — a signal
        # a working encoder must pick up.
        if rng.random() < 0.5:
            samples.append((["sigmoid", "feat"], 1.0))
        else:
            samples.append((["div", "feat", "feat"], 0.0))
    vocab = build_vocab(samples)
    model = TinyTransformer(vocab, seed=0)
    stats = model.train(samples, epochs=80, seed=0)
    assert stats["holdout_acc"] >= 0.9, stats
    good = model.score({"op": "sigmoid", "children": [{"feature": 0}]})
    bad = model.score({"op": "div", "children": [{"feature": 0}, {"feature": 1}]})
    assert good > bad, (good, bad)
    print(f"  learns separable task: holdout_acc={stats['holdout_acc']} good={good:.3f} bad={bad:.3f}")


def test_save_load_roundtrip():
    samples = [(["sigmoid", "feat"], 1.0), (["div", "feat", "feat"], 0.0)] * 15
    model = TinyTransformer(build_vocab(samples), seed=1)
    model.train(samples, epochs=20, seed=1)
    tree = {"op": "sigmoid", "children": [{"feature": 0}]}
    before = model.score(tree)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "card.json")
        model.save(path)
        reloaded = TinyTransformer.load(path)
        assert reloaded is not None
        after = reloaded.score(tree)
        assert abs(before - after) < 1e-9, (before, after)
        # open-box: explain returns readable token deltas
        expl = reloaded.explain()
        assert expl and isinstance(expl[0][0], str)
    print("  save/load round-trip byte-identical scores OK")


def test_rank_and_proposals():
    samples = [(["sigmoid", "feat"], 1.0), (["div", "feat", "feat"], 0.0)] * 15
    model = TinyTransformer(build_vocab(samples), seed=2)
    model.train(samples, epochs=40, seed=2)
    proposer = NeuralProposer(model)
    from figpso_uil.cognition.formula_engine import FormulaEngine
    eng = FormulaEngine(seed=0)
    parents = list(eng.get_seed_formulas().values())
    cands = []
    for i in range(8):
        mut, mt = eng.mutate(parents[i % len(parents)], stress_level=0.5)
        cands.append((mut, mt))
    ranked = proposer.rank_candidates(cands)
    assert len(ranked) == len(cands)
    # sorted descending by score
    scores = [s for _, _, s in ranked]
    assert scores == sorted(scores, reverse=True), scores
    print(f"  rank_candidates orders {len(ranked)} candidates by score OK")


def test_mutator_neural_path():
    """LLMMutator uses the neural proposer when a model card exists."""
    from figpso_uil.cognition.llm_mutator import LLMMutator
    samples = [(["sigmoid", "feat"], 1.0), (["div", "feat", "feat"], 0.0)] * 15
    model = TinyTransformer(build_vocab(samples), seed=3)
    model.train(samples, epochs=20, seed=3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "card.json")
        model.save(path)
        # point loader at the temp card
        import figpso_uil.cognition.neural_proposer as npmod
        orig = npmod._MODEL_PATH
        npmod._MODEL_PATH = path
        try:
            mut = LLMMutator(api_key=None)
            props = mut.propose(fingerprint={}, cells=[], n_proposals=3)
            assert props, "no proposals"
            assert all(p.proposer == "neural" for p in props), [p.proposer for p in props]
            assert all(p.op == "formula_mutate" for p in props)
        finally:
            npmod._MODEL_PATH = orig
    print(f"  LLMMutator offline path -> {len(props)} proposer='neural' proposals OK")


if __name__ == "__main__":
    test_tokenize()
    test_learns_separable()
    test_save_load_roundtrip()
    test_rank_and_proposals()
    test_mutator_neural_path()
    print("\n" + "=" * 50)
    print("NEURAL PROPOSER: ALL TESTS PASSED")
    print("=" * 50)
