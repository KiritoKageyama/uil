"""WsHub -- bridges on_stage() callbacks (fired from a run's worker
thread) into WebSocket broadcasts on FastAPI's asyncio event loop.

Why run_coroutine_threadsafe and not a polled queue: it wakes the event
loop immediately via its own scheduling mechanism, instead of waiting
for the next tick of an arbitrary poll interval -- the closest thing to
true push given fit() runs on a plain thread, not a coroutine.
"""

from __future__ import annotations
import asyncio
from fastapi import WebSocket


class WsHub:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: dict[str, set[WebSocket]] = {}

    def bind_loop(self) -> None:
        """Call once, from an async context (FastAPI startup)."""
        self.loop = asyncio.get_running_loop()

    def subscribe(self, run_id: str, ws: WebSocket) -> None:
        self._subscribers.setdefault(run_id, set()).add(ws)

    def unsubscribe(self, run_id: str, ws: WebSocket) -> None:
        self._subscribers.get(run_id, set()).discard(ws)

    def push(self, run_id: str, payload: dict) -> None:
        """Called from a WORKER THREAD. Must never block -- fire and
        forget; do not call .result() on the returned future here."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast(run_id, payload), self.loop
        )

    async def _broadcast(self, run_id: str, payload: dict) -> None:
        dead = []
        for ws in self._subscribers.get(run_id, ()):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._subscribers.get(run_id, set()).discard(ws)


hub = WsHub()
