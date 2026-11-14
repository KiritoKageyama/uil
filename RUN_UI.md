# UIL Live Console — Visual Test UI

Interactive web UI to watch UIL make decisions in real time and verify memory/learning persists.

## Start the server

From the **parent** of `figpso_uil` (e.g. `Downloads/`):

```bash
pip install fastapi uvicorn python-multipart
uvicorn figpso_uil.webui.backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

## What you'll see

### Decision Trace (main tab)
- **Pipeline view** — each stage shows: State → Episodic memory → DT principles → KG traversal → Search → Stress → (Investigate / Mine / Causal)
- **Live counters** — journal entries, principles, KG nodes, episodes, causal hypotheses, miner R²
- **Stage log** — scrolling history of every decision

### Memory Proof
- **Disk persistence** — size + record count for journal, principles, network, causal, episodes (refreshes every 3s)
- **Baseline button** — snapshot file sizes before a run, then watch `+KB` deltas as UIL learns
- **Growth chart** — in-memory knowledge counts over stages
- **Episodes** — cross-run warm-start memories

### Knowledge Graph
- Interactive vis.js graph of DT-mined nodes and typed edges

## How to prove learning is real

1. **Memory Proof** → click **Set baseline**
2. **Run Control** → start `easy_200x15`, XGBoost, **40+ stages**
3. Watch **Decision Trace** — after ~30 stages, **DT-MINER** step appears with new rules
4. Watch **Memory Proof** — journal/principles file sizes grow (`+delta`)
5. Run the **same dataset again** — Decision Trace step 2 should say **Episodic warm-start**
6. **Principles & Causal** tab — rules accumulate with `evidence_status: hypothesis`

## Terminal tests + web UI together

**Before:** Only runs started in the browser appeared live. Terminal tests wrote to disk silently.

**Now:** While the web server is running, **every** `UniversalOptimizer.fit()` call streams to the UI via `results/uil_live_feed.jsonl`.

### Setup (two terminals)

**Terminal 1 — keep web UI open:**
```bash
uvicorn figpso_uil.webui.backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — run any test:**
```bash
cd Downloads
python figpso_uil/tests/test_comprehensive.py
python figpso_uil/tests/test_stress_run.py
python -m figpso_uil.experiments.ablation_harness --stages 20 --seeds 1
```

**Browser:** open **Terminal Runs** tab — you'll see live stages, script name, pid, and whether it uses shared memory.

### What connects automatically

| Run type | Live in UI | Shared memory grows | After completion |
|---|---|---|---|
| Web UI run | ✓ WebSocket + feed | ✓ | Journal Replay |
| `test_comprehensive.py` | ✓ live feed | ✓ | Journal Replay + Memory Proof |
| `test_stress_run.py` | ✓ live feed | ✓ | same |
| `ablation_harness.py` | ✓ live feed | ✗ isolated temp dirs | Ablation reports tab |
| Web UI **off**, no env | ✗ | ✓ disk only | Journal Replay if you start UI later |

### Force feed without web UI

```bash
set UIL_LIVE_FEED=1
python figpso_uil/tests/test_comprehensive.py
```
Then start the web UI and open **Terminal Runs** — events are in the feed file.

### Journal Replay tab

Replay **any finished run** (terminal or web) by dataset name — reads `uil_journal_v19.json` from disk.
