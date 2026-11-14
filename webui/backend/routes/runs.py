from __future__ import annotations
from fastapi import APIRouter, HTTPException, UploadFile, File

from figpso_uil.webui.backend.models import StartRunRequest
from figpso_uil.webui.backend.run_manager import manager
from figpso_uil.webui.backend import datasets
from figpso_uil.core.resource_budget import detect_resource_budget

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/system/resources")
def system_resources():
    budget = detect_resource_budget()
    return {"budget": budget.to_dict()}


@router.get("/datasets")
def list_datasets():
    return {"datasets": datasets.list_datasets(), "info": datasets.list_datasets_info()}


@router.get("/datasets/info")
def datasets_info():
    return {"datasets": datasets.list_datasets_info()}


@router.post("/datasets/upload")
async def upload_dataset(name: str, label_column: str = "-1", file: UploadFile = File(...)):
    content = await file.read()
    try:
        col = int(label_column)
    except ValueError:
        col = label_column
    try:
        X, y = datasets.load_csv(content, label_column=col)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not parse CSV: {e}")
    datasets.register_uploaded(name, X, y)
    return {"ok": True, "name": name, "n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}


@router.post("")
def start_run(req: StartRunRequest):
    try:
        record, stopped_previous = manager.start_run(
            dataset_name=req.dataset_name,
            classifier_name=req.classifier,
            seed=req.seed,
            stages=req.stages,
            n_particles=req.n_particles,
            pso_iters=req.pso_iters,
            boosting_rounds=req.boosting_rounds,
            target_accuracy=req.target_accuracy,
            max_stages=req.max_stages,
            symbolic_learning_mode=req.symbolic_learning_mode,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    summary = record.to_summary()
    if stopped_previous:
        summary["stopped_previous"] = stopped_previous
    return summary


@router.get("")
def list_runs():
    return {"runs": manager.list_runs()}


@router.get("/{run_id}")
def get_run(run_id: str):
    record = manager.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    summary = record.to_summary()
    summary["last_snapshot"] = record.last_snapshot
    return summary


@router.post("/{run_id}/stop")
def stop_run(run_id: str):
    try:
        manager.stop_run(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@router.post("/{run_id}/pause")
def pause_run(run_id: str):
    try:
        manager.pause_run(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@router.post("/{run_id}/resume")
def resume_run(run_id: str):
    try:
        manager.resume_run(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}
