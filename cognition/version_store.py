"""VersionStore — event-sourced, lossless history for knowledge state.

v22 blueprint §1.2 (M1). Driving requirement: "the network loses at least
some of the information." The existing GraphChangeLog is append-only but
capped (oldest events truncated), cannot reconstruct past state, and
in-place decay erases belief trajectories. This store fixes all three:

- **Uncapped segment files** (`events_000001.jsonl`, ...): segments
  rotate at segment_size events; nothing is ever truncated.
- **State-based events**: every event carries the full canonical
  post-state of the affected cell (or None for removal), so replay is a
  trivial upsert/delete fold — reconstruction is byte-identical on
  canonical fields without re-executing any cognitive logic.
- **Snapshots**: written automatically every snapshot_every events (via
  a caller-provided snapshot_provider); `reconstruct(seq)` = nearest
  snapshot ≤ seq + replay of the tail. Answers "what did the system
  believe at seq N?"
- **Trajectories**: `trajectory(cell_id)` returns the full belief
  history of one cell (e.g. its confidence over time) — decay becomes
  a queryable sequence of events instead of a destructive overwrite.

Event schema (one JSON object per line):
    {seq, utc, cell_id, cell_type, from_version, to_version, op,
     payload: {post_state | None, details}, cause}

op vocabulary (free-form but conventional):
    create, mutate, merge, edit, strengthen, redirect, decay,
    promote, demote, quarantine, prune, remove, human_edit, human_reject

Removal-family ops (post_state None) delete the cell on replay; all
other ops upsert payload["post_state"].
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from figpso_uil.cognition.paths import results_path

_SEGMENT_RE = re.compile(r"^events_(\d{6})\.jsonl$")
# Snapshots may be stored plain (.json) or gzip-compressed (.json.gz);
# both carry byte-identical canonical state (gzip is lossless). compact()
# writes the compressed form; readers below transparently handle either.
_SNAPSHOT_RE = re.compile(r"^snapshot_(\d{9})\.json(\.gz)?$")

_SHARED: "VersionStore | None" = None
_SHARED_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_state(state: dict) -> str:
    """Stable content hash of a reconstructed canonical state dict."""
    blob = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _evenly_sample(items: list, k: int) -> list:
    """Up to k items spread evenly across the (ordered) list."""
    if k <= 0 or len(items) <= k:
        return list(items)
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _load_snapshot_json(path: str) -> dict:
    """Read a snapshot file, transparently decompressing .json.gz."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# The composite snapshot provider materializes canonical state from the
# LIVE graph + library files, several of which mutate OUTSIDE the per-cell
# event stream: edge/node bulk prune+decay emit one GRAPH-BULK summary
# event (editable_network `_bulk_log`, bug #18), and principle / routing_
# rule / formula / causal confidence decay + E3 tombstoning rewrite their
# library files without a matching per-cell event. Consequently NO bucket
# is guaranteed reproducible from events alone between two arbitrary
# snapshots — reconstruct() at a DELETED snapshot's seq coarsens those
# fields to the nearest KEPT snapshot. This was explicitly accepted for
# the compaction. The buckets are listed here only for the coarsening
# report; they are NOT a correctness gate.
_COMPOSITE_BUCKETS = (
    "node", "edge", "principle", "routing_rule",
    "formula", "causal_hypothesis", "episode", "goal",
)

# What compaction DOES guarantee, by construction (verified in the CLI):
#   - every event is preserved (the lossless history is untouched);
#   - reconstruct() at any KEPT snapshot seq — including latest — is
#     byte-identical before vs after (its base snapshot is unchanged,
#     only gzip-compressed; zero events are replayed on top);
#   - trajectories / lab reports (which read the event stream directly)
#     are unaffected.


class VersionStore:
    """Append-only, uncapped, replayable event store."""

    def __init__(
        self,
        directory: str | None = None,
        segment_size: int = 2000,
        snapshot_every: int = 500,
        compress_snapshots: bool = True,
    ):
        self.directory = directory or results_path("events")
        self.segment_size = int(segment_size)
        self.snapshot_every = int(snapshot_every)
        # New snapshots are born gzip-compressed so the events/ directory
        # stays lean going forward (a full composite snapshot is ~8 MB of
        # highly repetitive JSON -> ~1 MB gzipped). Lossless; readers below
        # handle either form. Set False only to preserve the legacy .json
        # layout (e.g. a test asserting on raw file bytes).
        self.compress_snapshots = bool(compress_snapshots)
        self._snapshot_provider: Callable[[], dict] | None = None
        # The shared() instance is appended to from webui worker threads;
        # a stopped run's thread can overlap the next run's for up to a
        # full stage, so seq allocation + segment writes must be atomic.
        # RLock because append() may call write_snapshot() while holding it.
        self._lock = threading.RLock()
        os.makedirs(self.directory, exist_ok=True)

        self._segments = self._list_segments()
        self._last_seq = self._read_last_seq()
        # Per-cell version counters, restored from latest snapshot + tail
        # so init cost stays proportional to one snapshot interval, not
        # the whole history.
        self._versions: dict[str, int] = {}
        self._restore_versions()

    @classmethod
    def shared(cls) -> "VersionStore":
        global _SHARED
        with _SHARED_LOCK:
            if _SHARED is None:
                _SHARED = cls()
        return _SHARED

    # ── Append ───────────────────────────────────────────────────────

    def set_snapshot_provider(self, provider: Callable[[], dict]) -> None:
        """Register a callable returning the current canonical state.

        Called automatically every snapshot_every events so reconstruction
        never replays more than one snapshot interval.
        """
        self._snapshot_provider = provider

    def append(
        self,
        *,
        cell_id: str,
        cell_type: str,
        op: str,
        payload: dict | None = None,
        cause: str | None = None,
    ) -> int:
        """Append one event; returns its seq."""
        with self._lock:
            self._last_seq += 1
            seq = self._last_seq
            from_version = self._versions.get(cell_id, 0)
            to_version = from_version + 1
            self._versions[cell_id] = to_version

            event = {
                "seq": seq,
                "utc": _utc_now(),
                "cell_id": cell_id,
                "cell_type": cell_type,
                "from_version": from_version,
                "to_version": to_version,
                "op": op,
                "payload": payload or {},
                "cause": cause,
            }
            path = self._segment_for_append(seq)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")

            if (
                self._snapshot_provider is not None
                and seq % self.snapshot_every == 0
            ):
                try:
                    self.write_snapshot(self._snapshot_provider())
                except Exception:
                    # A failed snapshot must never lose the event itself;
                    # reconstruction just replays a longer tail.
                    pass
            return seq

    # ── Reading / replay ─────────────────────────────────────────────

    def events(
        self, start_seq: int = 1, end_seq: int | None = None
    ) -> Iterator[dict]:
        """Iterate events with start_seq <= seq <= end_seq, in order.

        Segments outside [start_seq, end_seq] are skipped wholesale —
        segment n deterministically covers seqs ((n-1)*size, n*size]
        (see _segment_for_append), so a tail replay after a snapshot
        opens a handful of files, not the entire history.
        """
        for seg_no, path in self._list_segments():
            seg_hi = seg_no * self.segment_size
            if seg_hi < start_seq:
                continue
            seg_lo = (seg_no - 1) * self.segment_size + 1
            if end_seq is not None and seg_lo > end_seq:
                return
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    # NUL-strip + parse guard: a process killed mid-append
                    # leaves a torn tail (NUL padding / partial JSON) that
                    # must never poison replay of the valid history.
                    line = line.strip().strip("\x00")
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev["seq"] < start_seq:
                        continue
                    if end_seq is not None and ev["seq"] > end_seq:
                        return
                    yield ev

    def reconstruct(
        self, seq: int | None = None, *, _snaps: list | None = None
    ) -> dict:
        """Rebuild canonical state as of `seq` (None = latest).

        Returns {"seq": s, "state": {cell_type: {cell_id: post_state}}}.

        `_snaps` (internal) restricts snapshot selection to a given
        (seq, path) list — compact() uses it to simulate reconstruction
        AFTER a set of snapshots has been deleted, without touching disk.
        """
        target = seq if seq is not None else self._last_seq
        snap_seq, state = self._load_snapshot_at_or_before(target, snaps=_snaps)
        for ev in self.events(start_seq=snap_seq + 1, end_seq=target):
            self._apply(state, ev)
        state = {k: v for k, v in state.items() if v}
        return {"seq": target, "state": state}

    def record_cell(
        self,
        cell_id: str,
        cell_type: str,
        op: str,
        post_state: dict | None,
        *,
        cause: str | None = None,
        details: dict | None = None,
    ) -> int:
        """Convenience: append a state-based cell event."""
        payload: dict = {"post_state": post_state}
        if details:
            payload["details"] = details
        return self.append(
            cell_id=cell_id,
            cell_type=cell_type,
            op=op,
            payload=payload,
            cause=cause,
        )

    def trajectory(self, cell_id: str) -> list[dict]:
        """Full belief history of one cell.

        Returns [{seq, utc, op, version, confidence, post_state}] --
        confidence extracted from post_state when present, so a decayed
        belief (0.9 -> 0.3 over time) is distinguishable from one that
        was always 0.3.
        """
        out = []
        for ev in self.events():
            if ev.get("cell_id") != cell_id:
                continue
            post = (ev.get("payload") or {}).get("post_state")
            conf = post.get("confidence") if isinstance(post, dict) else None
            out.append({
                "seq": ev["seq"],
                "utc": ev["utc"],
                "op": ev["op"],
                "version": ev["to_version"],
                "confidence": conf,
                "post_state": post,
            })
        return out

    @property
    def last_seq(self) -> int:
        return self._last_seq

    # ── Snapshots ────────────────────────────────────────────────────

    def write_snapshot(self, state: dict) -> str:
        """Persist a canonical-state snapshot at the current seq.

        Written gzip-compressed (``snapshot_XXXXXXXXX.json.gz``) unless
        ``compress_snapshots`` is False. Atomic via tmp + os.replace so a
        crash mid-write never leaves a half-snapshot under the real name.
        """
        with self._lock:
            base = os.path.join(
                self.directory, f"snapshot_{self._last_seq:09d}.json"
            )
            path = base + ".gz" if self.compress_snapshots else base
            blob = {
                "seq": self._last_seq,
                "utc": _utc_now(),
                "state": state,
                "versions": dict(self._versions),
            }
            tmp = path + ".tmp"
            if self.compress_snapshots:
                with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                    json.dump(blob, fh, default=str)
            else:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(blob, fh, default=str)
            os.replace(tmp, path)
            return path

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _apply(state: dict, event: dict) -> None:
        """State-based reducer: upsert post_state, delete on None."""
        cell_type = event.get("cell_type") or "unknown"
        cell_id = event.get("cell_id")
        post = (event.get("payload") or {}).get("post_state", "__absent__")
        if post == "__absent__":
            return  # informational event (graph-level stats etc.)
        bucket = state.setdefault(cell_type, {})
        if post is None:
            bucket.pop(cell_id, None)
        else:
            bucket[cell_id] = post

    def _list_segments(self) -> list[tuple[int, str]]:
        segs = []
        for name in os.listdir(self.directory):
            m = _SEGMENT_RE.match(name)
            if m:
                segs.append((int(m.group(1)), os.path.join(self.directory, name)))
        segs.sort()
        return segs

    def _segment_for_append(self, seq: int) -> str:
        seg_no = (seq - 1) // self.segment_size + 1
        return os.path.join(self.directory, f"events_{seg_no:06d}.jsonl")

    def _read_last_seq(self) -> int:
        segs = self._list_segments()
        if not segs:
            return 0
        self._repair_torn_tail(segs[-1][1])
        last_seq = 0
        with open(segs[-1][1], encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip().strip("\x00")
                if not line:
                    continue
                try:
                    last_seq = int(json.loads(line)["seq"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return last_seq

    @staticmethod
    def _repair_torn_tail(path: str) -> None:
        """Truncate a torn tail write on the newest segment.

        A process killed mid-append (e.g. force-stopped webui server)
        can leave NUL padding or a partial JSON line at the file end.
        Standard WAL recovery: cut the file back to the last valid,
        newline-terminated event. Only trailing garbage is removed —
        valid history is never touched.
        """
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return
        good_end = 0
        pos = 0
        for raw in data.split(b"\n"):
            line_end = pos + len(raw)
            stripped = raw.strip(b"\x00 \t\r")
            if stripped:
                try:
                    json.loads(stripped.decode("utf-8"))
                    good_end = min(line_end + 1, len(data))
                except (ValueError, UnicodeDecodeError):
                    pass
            pos = line_end + 1
        if good_end < len(data):
            try:
                with open(path, "r+b") as fh:
                    fh.truncate(good_end)
            except OSError:
                pass

    def _snapshots(self) -> list[tuple[int, str]]:
        snaps = []
        for name in os.listdir(self.directory):
            m = _SNAPSHOT_RE.match(name)
            if m:
                snaps.append((int(m.group(1)), os.path.join(self.directory, name)))
        snaps.sort()
        return snaps

    def _load_snapshot_at_or_before(
        self, seq: int, snaps: list | None = None
    ) -> tuple[int, dict]:
        # Select the nearest snapshot FIRST, then parse exactly one file.
        # The old loop json.load()ed EVERY snapshot <= seq — with
        # thousands of multi-MB composite snapshots that turned every
        # reconstruct() (webui /state/cells, /state/replay) into
        # gigabytes of dead I/O and an effectively hung endpoint.
        # `snaps` (compact's simulated post-deletion set) overrides the
        # on-disk listing.
        best_seq = 0
        best_path: str | None = None
        for snap_seq, path in (snaps if snaps is not None else self._snapshots()):
            if best_seq < snap_seq <= seq:
                best_seq, best_path = snap_seq, path
        if best_path is None:
            return 0, {}
        try:
            blob = _load_snapshot_json(best_path)
        except (OSError, json.JSONDecodeError, EOFError, gzip.BadGzipFile):
            # A torn/unreadable snapshot must not kill reconstruction —
            # fall back to full replay from the event log.
            return 0, {}
        return best_seq, blob.get("state", {})

    def _restore_versions(self) -> None:
        snap_seq = 0
        snaps = self._snapshots()
        if snaps:
            try:
                blob = _load_snapshot_json(snaps[-1][1])
                self._versions = dict(blob.get("versions", {}))
                snap_seq = blob.get("seq", 0)
            except (OSError, json.JSONDecodeError, EOFError, gzip.BadGzipFile):
                # Unreadable newest snapshot: rebuild versions by full
                # replay rather than crashing on open.
                self._versions = {}
                snap_seq = 0
        for ev in self.events(start_seq=snap_seq + 1):
            self._versions[ev["cell_id"]] = ev["to_version"]

    # ── Compaction / retention ───────────────────────────────────────

    def _compaction_plan(
        self, keep_recent: int, stride: int
    ) -> tuple[list, list]:
        """Split snapshots into (kept, to_delete) — stable across re-runs.

        Retention is keyed on seq, not list position, so running compact()
        twice deletes nothing new. A snapshot is KEPT iff it is one of the
        newest `keep_recent`, OR the single oldest (bounds early-seq
        reconstruct), OR its seq is a multiple of stride*snapshot_every
        (evenly spaced historical anchors). Events are never in scope here.
        """
        snaps = self._snapshots()  # sorted (seq, path)
        if not snaps:
            return [], []
        keep_period = max(1, stride) * max(1, self.snapshot_every)
        keep_recent = max(0, keep_recent)
        n = len(snaps)
        recent_start = n - keep_recent
        oldest_seq = snaps[0][0]
        kept: list = []
        to_delete: list = []
        for idx, (snap_seq, path) in enumerate(snaps):
            keep = (
                idx >= recent_start
                or snap_seq == oldest_seq
                or (snap_seq % keep_period == 0)
            )
            (kept if keep else to_delete).append((snap_seq, path))
        return kept, to_delete

    @staticmethod
    def _bucket_diff_count(a: dict, b: dict) -> int:
        """How many cells differ between two same-typed cell buckets."""
        n = 0
        for k in set(a) | set(b):
            if k not in a or k not in b:
                n += 1
            elif (
                json.dumps(a[k], sort_keys=True, default=str)
                != json.dumps(b[k], sort_keys=True, default=str)
            ):
                n += 1
        return n

    def _gzip_snapshot_inplace(self, path: str) -> int:
        """Replace a plain .json snapshot with .json.gz; return bytes freed.

        Raw-byte gzip (not re-serialized) so the compressed file decodes to
        exactly the original bytes — reconstruct() reads identical content.
        Atomic: write tmp -> os.replace -> remove original, so a reader
        always sees one valid snapshot for this seq.
        """
        if path.endswith(".gz"):
            return 0
        gz_path = path + ".gz"
        tmp = gz_path + ".tmp"
        orig = _safe_size(path)
        try:
            with open(path, "rb") as fin, gzip.open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1 << 20)
            os.replace(tmp, gz_path)
        except OSError:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return 0
        new = _safe_size(gz_path)
        try:
            os.remove(path)
        except OSError:
            pass
        return max(0, orig - new)

    def compact(
        self,
        keep_recent: int = 200,
        stride: int = 20,
        *,
        gzip_survivors: bool = True,
        verify_samples: int = 32,
        dry_run: bool = False,
    ) -> dict:
        """Reclaim snapshot disk without ever deleting an event.

        Deleting an intermediate snapshot only lengthens a tail replay for
        the event-sourced structure of every cell, but the composite
        snapshot also carries values the event log summarizes rather than
        records (bulk graph prune/decay, library-file confidence decay — see
        `_COMPOSITE_BUCKETS`). For those, reconstruct() at a DELETED seq
        coarsens to the nearest KEPT snapshot. That trade was accepted; this
        method measures its magnitude for the report but never blocks on it.
        What it never sacrifices: the events themselves, and byte-identical
        reconstruct() at every KEPT snapshot seq (their bases are untouched).

        Steps: (1) plan kept/to_delete; (2) sample the coarsening at doomed
        seqs (informational); (3) delete the redundant snapshots; (4) gzip
        the survivors. Idempotent — safe to re-run.
        """
        with self._lock:
            kept, to_delete = self._compaction_plan(keep_recent, stride)
        stats: dict = {
            "total_snapshots": len(kept) + len(to_delete),
            "kept": len(kept),
            "deleted": 0,
            "gzipped": 0,
            "already_gz": 0,
            "freed_bytes": 0,
            "sampled_seqs": 0,
            "coarsened": {},  # bucket -> {"seqs": n, "max_diff": m}
            "dry_run": dry_run,
        }

        # ── Measure coarsening (informational, never blocks) ──────────
        # For each sampled doomed snapshot at seq D: compare reconstruct(D)
        # using ALL snapshots (base == snapshot D, zero events) against
        # reconstruct(D) using only the KEPT set (earlier base + replayed
        # tail). Any per-bucket difference is the point-in-time value that
        # will snap to the nearest kept snapshot post-deletion.
        if to_delete and verify_samples > 0:
            all_snaps = sorted(kept + to_delete)
            for d in _evenly_sample([s for s, _ in to_delete], verify_samples):
                full = self.reconstruct(d, _snaps=all_snaps)["state"]
                restr = self.reconstruct(d, _snaps=kept)["state"]
                stats["sampled_seqs"] += 1
                for b in _COMPOSITE_BUCKETS:
                    fb, rb = full.get(b, {}), restr.get(b, {})
                    diff = self._bucket_diff_count(fb, rb)
                    if diff:
                        rec = stats["coarsened"].setdefault(b, {"seqs": 0, "max_diff": 0})
                        rec["seqs"] += 1
                        rec["max_diff"] = max(rec["max_diff"], diff)

        # ── Delete redundant snapshots ────────────────────────────────
        for _seq, path in to_delete:
            sz = _safe_size(path)
            if dry_run:
                stats["freed_bytes"] += sz
                stats["deleted"] += 1
                continue
            try:
                os.remove(path)
                stats["freed_bytes"] += sz
                stats["deleted"] += 1
            except OSError:
                pass

        # ── Gzip survivors ────────────────────────────────────────────
        if gzip_survivors:
            for _seq, path in kept:
                if path.endswith(".gz"):
                    stats["already_gz"] += 1
                    continue
                if dry_run:
                    # Estimate: this composite JSON compresses ~7-8x.
                    stats["freed_bytes"] += int(_safe_size(path) * 0.85)
                    stats["gzipped"] += 1
                    continue
                freed = self._gzip_snapshot_inplace(path)
                if freed >= 0:
                    stats["freed_bytes"] += freed
                    stats["gzipped"] += 1
        return stats


# ── CLI ──────────────────────────────────────────────────────────────


def _dir_bytes(directory: str) -> int:
    total = 0
    try:
        for name in os.listdir(directory):
            total += _safe_size(os.path.join(directory, name))
    except OSError:
        pass
    return total


def _bucket_hashes(state: dict) -> dict:
    buckets = set(state) | set(_COMPOSITE_BUCKETS)
    return {b: _hash_state(state.get(b, {})) for b in buckets}


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m figpso_uil.cognition.version_store",
        description=(
            "VersionStore maintenance. --compact deletes redundant "
            "intermediate snapshots and gzips the survivors; events (the "
            "lossless history) are never touched."
        ),
    )
    parser.add_argument("--compact", action="store_true",
                        help="Run snapshot compaction/retention.")
    parser.add_argument("--dir", default=None,
                        help="events/ directory (default: results/events).")
    parser.add_argument("--keep-recent", type=int, default=200,
                        help="Always keep the newest N snapshots (default 200).")
    parser.add_argument("--stride", type=int, default=20,
                        help="Also keep every Nth older snapshot (default 20).")
    parser.add_argument("--no-gzip", action="store_true",
                        help="Delete redundant snapshots but do NOT gzip survivors.")
    parser.add_argument("--verify", type=int, default=6,
                        help="Historical seqs per class to hash-check before/after "
                             "(default 6 kept + 6 deleted).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the plan and safety gate without deleting.")
    args = parser.parse_args(argv)

    if not args.compact:
        parser.print_help()
        return 0

    store = VersionStore(directory=args.dir)
    kept_plan, del_plan = store._compaction_plan(args.keep_recent, args.stride)
    before_bytes = _dir_bytes(store.directory)
    print(f"[versionstore] dir        : {store.directory}")
    print(f"[versionstore] last_seq   : {store.last_seq}")
    print(f"[versionstore] snapshots  : {len(kept_plan) + len(del_plan)} "
          f"({before_bytes / 1e9:.2f} GB dir)")
    print(f"[versionstore] plan       : keep {len(kept_plan)}, delete "
          f"{len(del_plan)}  (keep_recent={args.keep_recent}, stride={args.stride}, "
          f"gzip={'off' if args.no_gzip else 'on'})")
    if not del_plan and (args.no_gzip or all(p.endswith('.gz') for _, p in kept_plan)):
        print("[versionstore] nothing to do (already compacted).")
        return 0

    # Before/after receipt: sample KEPT seqs (must stay byte-identical for
    # every bucket) and DELETED seqs (reconstruct still works; some buckets
    # coarsen). Hash per bucket now, while all snapshots are still present.
    kept_seqs = _evenly_sample([s for s, _ in kept_plan], args.verify)
    del_seqs = _evenly_sample([s for s, _ in del_plan], args.verify)
    before = {s: _bucket_hashes(store.reconstruct(s)["state"])
              for s in kept_seqs + del_seqs}

    stats = store.compact(
        keep_recent=args.keep_recent, stride=args.stride,
        gzip_survivors=not args.no_gzip, dry_run=args.dry_run,
    )

    verb = "WOULD FREE" if args.dry_run else "FREED"
    print(f"[versionstore] deleted    : {stats['deleted']} snapshots")
    print(f"[versionstore] gzipped    : {stats['gzipped']} survivors "
          f"({stats['already_gz']} already compressed)")
    print(f"[versionstore] {verb:<10}: {stats['freed_bytes'] / 1e9:.2f} GB")
    print(f"[versionstore] coarsening : sampled {stats['sampled_seqs']} deleted "
          f"seqs (historical reconstruct at these snaps snaps to nearest kept)")
    if stats["coarsened"]:
        for b, rec in sorted(stats["coarsened"].items()):
            print(f"[versionstore]   '{b}': up to {rec['max_diff']} cells differ "
                  f"in {rec['seqs']}/{stats['sampled_seqs']} sampled seqs")
    else:
        print("[versionstore]   (no divergence in sample)")

    if args.dry_run:
        print("[versionstore] dry-run — no files changed.")
        return 0

    # Re-open from disk (survivors now gzipped) and re-verify by hashing.
    after_store = VersionStore(directory=args.dir)
    after = {s: _bucket_hashes(after_store.reconstruct(s)["state"])
             for s in kept_seqs + del_seqs}
    after_bytes = _dir_bytes(after_store.directory)

    # HARD guarantee: reconstruct() at every KEPT seq is byte-identical
    # before vs after (all buckets). A mismatch here means gzip corrupted
    # a survivor or a needed base was deleted — a real failure.
    kept_ok = all(before[s] == after[s] for s in kept_seqs)
    print(f"[versionstore] dir now    : {after_bytes / 1e9:.2f} GB "
          f"(reclaimed {(before_bytes - after_bytes) / 1e9:.2f} GB)")
    print(f"[versionstore] verify KEPT : {len(kept_seqs)} seqs — "
          f"{'ALL IDENTICAL (every bucket)' if kept_ok else 'MISMATCH!'}")
    for s in kept_seqs:
        print(f"    kept seq {s:>10}: {'ok' if before[s] == after[s] else 'DIFF'}")
    # Informational: at DELETED seqs, list which buckets coarsened (their
    # historical point-in-time value now resolves to the nearest kept snap).
    print(f"[versionstore] verify DEL  : {len(del_seqs)} seqs — reconstruct "
          f"still succeeds; coarsened buckets shown per seq (by design):")
    for s in del_seqs:
        coarse = [b for b in _COMPOSITE_BUCKETS
                  if before[s].get(b) != after[s].get(b)]
        print(f"    del  seq {s:>10}: coarsened={coarse or 'none'}")

    return 0 if kept_ok else 2


if __name__ == "__main__":
    import sys

    sys.exit(_main())
