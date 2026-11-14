"""UIL Research Console -- FastAPI entry point.

Run from figpso_uil's PARENT directory (e.g. C:\\Users\\arinb\\Downloads),
matching the `from figpso_uil.X import Y` import convention used
throughout this codebase's own tests:

    uvicorn figpso_uil.webui.backend.main:app --reload

Then open http://localhost:8000
"""

from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from figpso_uil.webui.backend.ws_hub import hub
from figpso_uil.webui.backend.routes import runs, state, retrace, ws
from figpso_uil.cognition.live_feed import enable as enable_live_feed

app = FastAPI(title="UIL Research Console")

app.include_router(runs.router, prefix="/api")
app.include_router(state.router, prefix="/api")
app.include_router(retrace.router, prefix="/api")
app.include_router(ws.router)  # websocket path is already /ws/runs/{id}


@app.on_event("startup")
async def _startup() -> None:
    hub.bind_loop()
    enable_live_feed()  # terminal tests publish to live feed while UI runs


_webui_dir = Path(__file__).resolve().parent.parent
_frontend_dir = _webui_dir / "frontend"
_desktop_dir = _webui_dir / "desktop"
_shared_dir = _webui_dir / "shared"


# Legacy /desktop → /app (browsers keep #hash fragments on 307).
@app.api_route("/desktop", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/desktop/", methods=["GET", "HEAD"], include_in_schema=False)
async def _redirect_desktop_root() -> RedirectResponse:
    return RedirectResponse(url="/app/", status_code=307)


@app.api_route("/desktop/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def _redirect_desktop_path(path: str) -> RedirectResponse:
    return RedirectResponse(url=f"/app/{path}", status_code=307)


# Specific mounts must come BEFORE the "/" catch-all or they are shadowed.
# / = pitch + marketing · /app = Research Console · /shared = tokens + engines
if _shared_dir.is_dir():
    app.mount(
        "/shared", StaticFiles(directory=str(_shared_dir), html=False),
        name="shared",
    )
if _desktop_dir.is_dir():
    app.mount(
        "/app", StaticFiles(directory=str(_desktop_dir), html=True),
        name="app",
    )
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
