from __future__ import annotations
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from figpso_uil.webui.backend.run_manager import manager
from figpso_uil.webui.backend.ws_hub import hub

router = APIRouter(tags=["ws"])


@router.websocket("/ws/runs/{run_id}")
async def run_ws(websocket: WebSocket, run_id: str):
    await websocket.accept()
    record = manager.get(run_id)
    if record is None:
        await websocket.send_json({"type": "error", "error": "run not found"})
        await websocket.close()
        return

    hub.subscribe(run_id, websocket)
    if record.last_snapshot is not None:
        await websocket.send_json({"type": "stage", **record.last_snapshot})
    try:
        while True:
            # Client doesn't need to send anything; just keep the
            # connection open until it disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(run_id, websocket)
