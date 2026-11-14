"""Test V20+.3: Editable Network — nodes, edges, traversal, loop decay."""
import sys
sys.path.insert(0, ".")
from datetime import datetime, timezone
import numpy as np
from figpso_uil.cognition.editable_network import (
    EditableNetwork, NetworkNode, NetworkEdge,
    EDGE_LEADS_TO, EDGE_CONTRADICTS, EDGE_DEPENDS_ON, EDGE_STRENGTHENS,
    LOOP_DECAY_FACTOR, LOOP_CUTOFF,
)


def test_node_creation():
    net = EditableNetwork(filepath=None)
    nid1 = net.add_node(
        conditions=[{"feature": "state.sparsity", "op": ">", "threshold": 0.5}],
        action_override={"max_penalty": 2.0},
        confidence=0.8, evidence_count=20,
    )
    nid2 = net.add_node(
        conditions=[{"feature": "action.penalty", "op": ">", "threshold": 3.0}],
        action_override={"max_penalty": 3.0},
        confidence=0.7, evidence_count=15,
    )
    assert net.node_count == 2
    assert net.get_node(nid1).confidence == 0.8
    print("TEST 1: Node creation OK")
    return nid1, nid2, net


def test_duplicate_merge():
    net = EditableNetwork(filepath=None)
    nid1 = net.add_node(
        conditions=[{"feature": "state.sparsity", "op": ">", "threshold": 0.5}],
        confidence=0.8, evidence_count=10,
    )
    # Same conditions, should merge
    nid2 = net.add_node(
        conditions=[{"feature": "state.sparsity", "op": ">", "threshold": 0.5}],
        confidence=0.9, evidence_count=5,
    )
    assert nid1 == nid2  # Same node
    assert net.node_count == 1
    assert net.get_node(nid1).evidence_count == 15  # Merged
    print("TEST 2: Duplicate merge OK")


def test_edge_creation():
    nid1, nid2, net = test_node_creation()
    eidx = net.add_edge(nid1, nid2, EDGE_LEADS_TO, weight=0.7)
    assert eidx is not None
    assert net.edge_count == 1
    # Adding same edge again should strengthen it
    eidx2 = net.add_edge(nid1, nid2, EDGE_LEADS_TO, weight=0.5)
    assert eidx == eidx2  # Same edge
    assert net.edge_count == 1  # Still 1 edge
    print("TEST 3: Edge creation + strengthening OK")


def test_traversal_basic():
    net = EditableNetwork(filepath=None)
    n1 = net.add_node(
        conditions=[{"feature": "sparsity", "op": ">", "threshold": 0.3}],
        action_override={"max_penalty": 2.0},
        confidence=0.9,
    )
    n2 = net.add_node(
        conditions=[{"feature": "n_features", "op": ">", "threshold": 50}],
        action_override={"min_penalty": 0.5},
        confidence=0.8,
    )
    net.add_edge(n1, n2, EDGE_LEADS_TO, weight=0.8)

    state = {"sparsity": 0.6, "n_features": 100, "n_samples": 200}
    result = net.traverse(state)
    assert n1 in result["activated_nodes"]
    assert n2 in result["activated_nodes"]
    assert "max_penalty" in result["overrides"]
    print("TEST 4: Basic traversal OK")
    return result


def test_contradicts_edge():
    net = EditableNetwork(filepath=None)
    n1 = net.add_node(
        conditions=[{"feature": "sparsity", "op": ">", "threshold": 0.3}],
        action_override={"max_penalty": 2.0},
        confidence=0.9,
    )
    n2 = net.add_node(
        conditions=[{"feature": "sparsity", "op": ">", "threshold": 0.1}],
        action_override={"max_penalty": 5.0},
        confidence=0.6,
    )
    # n1 CONTRADICTS n2
    net.add_edge(n1, n2, EDGE_CONTRADICTS, weight=0.9)

    state = {"sparsity": 0.6}
    result = net.traverse(state)
    # n1 should fire but n2 should be excluded
    assert n1 in result["activated_nodes"]
    assert n2 not in result["activated_nodes"]
    assert result["overrides"].get("max_penalty") == 2.0  # n1's override, not n2's
    print("TEST 5: CONTRADICTS edge OK")


def test_loop_decay():
    """Test that revisiting a node decays the weight."""
    net = EditableNetwork(filepath=None)
    n1 = net.add_node(
        conditions=[{"feature": "x", "op": ">", "threshold": 0}],
        action_override={"val": 1.0}, confidence=0.9,
    )
    n2 = net.add_node(
        conditions=[{"feature": "y", "op": ">", "threshold": 0}],
        action_override={"val2": 2.0}, confidence=0.9,
    )
    assert net.node_count == 2, f"Expected 2 nodes, got {net.node_count}"
    # Create a cycle: n1 -> n2 -> n1
    net.add_edge(n1, n2, EDGE_LEADS_TO, weight=0.9)
    net.add_edge(n2, n1, EDGE_LEADS_TO, weight=0.9)

    state = {"x": 1.0, "y": 1.0}
    result = net.traverse(state, max_depth=20)
    # Both should activate, but the loop should terminate
    # due to weight decay below 0.01
    assert len(result["activated_nodes"]) == 2
    print("TEST 6: Loop decay (cycle terminates) OK")


def test_edge_decay():
    net = EditableNetwork(filepath=None)
    n1 = net.add_node(conditions=[{"feature": "x", "op": ">", "threshold": 0}])
    n2 = net.add_node(conditions=[{"feature": "y", "op": ">", "threshold": 0}], node_id="N-DECAY-2")
    idx = net.add_edge(n1, n2, EDGE_LEADS_TO, weight=0.5)
    # Simulate many failed traversals
    edge = net._edges[idx]
    edge.times_traversed = 20
    edge.times_helped = 2  # Only 10% help rate
    removed = net.apply_edge_decay(min_traversals=10)
    assert removed == 1
    assert net.edge_count == 0
    print("TEST 7: Edge decay (bad edges removed) OK")


def test_import_from_principles():
    net = EditableNetwork(filepath=None)
    principles = [
        {
            "principle_id": "P-001",
            "conditions": [{"feature": "sparsity", "op": ">", "threshold": 0.5}],
            "action_override": {"max_penalty": 2.0},
            "tier": 2, "confidence": 0.85, "evidence_count": 50,
        },
        {
            "principle_id": "P-002",
            "conditions": [{"feature": "penalty", "op": ">", "threshold": 3.0}],
            "action_override": {"max_penalty": 3.0},
            "tier": 3, "confidence": 0.75, "evidence_count": 20,
        },
    ]
    count = net.import_from_principles(principles)
    assert count == 2
    assert net.node_count == 2
    assert net.get_node("P-001").tier == 2
    print("TEST 8: Import from principles OK")


def test_serialization():
    net = EditableNetwork(filepath=None)
    n1 = net.add_node(
        conditions=[{"feature": "x", "op": ">", "threshold": 0.5}],
        confidence=0.8,
    )
    n2 = net.add_node(
        conditions=[{"feature": "y", "op": "<", "threshold": 10}],
        confidence=0.7, node_id="N-SER-2",
    )
    net.add_edge(n1, n2, EDGE_STRENGTHENS, weight=0.6)

    # Serialize
    data = net.to_dict()
    assert "nodes" in data
    assert "edges" in data

    # Deserialize
    net2 = EditableNetwork.from_dict(data)
    assert net2.node_count == 2
    assert net2.edge_count == 1
    assert net2.get_node(n1).confidence == 0.8
    print("TEST 9: Serialization round-trip OK")


def test_relationship_discovery():
    net = EditableNetwork(filepath=None)
    n1 = net.add_node(conditions=[{"feature": "a", "op": ">", "threshold": 0}], node_id="N-RD-1")
    n2 = net.add_node(conditions=[{"feature": "b", "op": ">", "threshold": 0}], node_id="N-RD-2")
    n3 = net.add_node(conditions=[{"feature": "c", "op": ">", "threshold": 0}], node_id="N-RD-3")

    # Simulate co-activation history: n1 and n2 always co-activate and succeed.
    # (stage, activated_nodes) tuples -- matched by stage, not list position.
    activated_history = [
        (0, ["N-RD-1", "N-RD-2"]),
        (1, ["N-RD-1", "N-RD-2"]),
        (2, ["N-RD-1", "N-RD-2"]),
        (3, ["N-RD-1", "N-RD-2", "N-RD-3"]),
        (4, ["N-RD-1", "N-RD-2"]),
    ]
    journal = [
        {"stage": 0, "regret": 0.0}, {"stage": 1, "regret": 0.0},
        {"stage": 2, "regret": 0.0}, {"stage": 3, "regret": 0.5},
        {"stage": 4, "regret": 0.0},
    ]
    new_edges = net.discover_relationships(journal, activated_history)
    assert new_edges > 0
    assert net.edge_count > 0
    print(f"TEST 10: Relationship discovery OK ({new_edges} edges)")


def test_summary():
    net = EditableNetwork(filepath=None)
    n1 = net.add_node(conditions=[{"feature": "x", "op": ">", "threshold": 0}], tier=1)
    n2 = net.add_node(conditions=[{"feature": "y", "op": ">", "threshold": 0}], tier=2, node_id="N-SUM-2")
    net.add_edge(n1, n2, EDGE_LEADS_TO)
    summary = net.get_summary()
    assert summary["nodes"] == 2
    assert summary["edges"] == 1
    print(f"TEST 11: Summary OK — {summary}")


def test_metric_prune():
    """Bad/stale cross-domain items archive; good current-domain items stay."""
    from figpso_uil.cognition.knowledge_archive import KnowledgeArchive

    net = EditableNetwork(filepath=None)
    net.set_dataset_context("real_spambase")

    good = net.add_node(
        conditions=[{"feature": "x", "op": ">", "threshold": 0}],
        confidence=0.9,
        evidence_count=40,
        node_id="N-GOOD",
    )
    bad = net.add_node(
        conditions=[{"feature": "y", "op": ">", "threshold": 99}],
        confidence=0.2,
        evidence_count=1,
        node_id="N-BAD",
    )
    net.get_node(good).metadata["datasets"] = ["real_spambase"]
    net.get_node(good).activation_count = 20
    net.get_node(good).last_activated = datetime.now(timezone.utc).isoformat()
    net.get_node(bad).metadata["datasets"] = ["real_breast_cancer"]

    idx_good = net.add_edge(good, bad, EDGE_LEADS_TO, weight=0.85)
    idx_bad = net.add_edge(bad, good, EDGE_LEADS_TO, weight=0.15)
    good_edge = net._edges[idx_good]
    good_edge.times_traversed = 30
    good_edge.times_helped = 24
    good_edge.metadata["datasets"] = ["real_spambase"]
    good_edge.last_traversed = datetime.now(timezone.utc).isoformat()

    bad_edge = net._edges[idx_bad]
    bad_edge.times_traversed = 15
    bad_edge.times_helped = 1
    bad_edge.metadata["datasets"] = ["real_breast_cancer"]

    archive = KnowledgeArchive(filepath=None)
    stats = net.prune_working_graph(
        archive,
        dataset="real_spambase",
        system_stress=0.3,
        protect_nodes=set(),
        max_edges=100,
        max_nodes=100,
    )
    removed = stats["edges_archived"] + stats.get("edges_decayed", 0)
    assert removed >= 1, stats
    assert net.edge_count == 1
    assert net.get_node(good) is not None
    print(f"TEST 12: Metric prune OK — {stats}")


# Run all tests
test_node_creation()
test_duplicate_merge()
test_edge_creation()
test_traversal_basic()
test_contradicts_edge()
test_loop_decay()
test_edge_decay()
test_import_from_principles()
test_serialization()
test_relationship_discovery()
test_summary()
test_metric_prune()

print("\n" + "=" * 50)
print("ALL V20+.3 EDITABLE NETWORK TESTS PASSED")
