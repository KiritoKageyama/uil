from __future__ import annotations
from pydantic import BaseModel


class StartRunRequest(BaseModel):
    dataset_name: str
    classifier: str = "XGBoost"  # "XGBoost" | "RandomForest" | "KNN" | "SymbolicFormula"
    seed: int = 42
    stages: int = 50
    n_particles: int = 15
    pso_iters: int = 15
    boosting_rounds: int = 30
    target_accuracy: float | None = None  # e.g. 0.80 — run until hit (needs max_stages cap)
    max_stages: int = 300  # safety cap when target_accuracy is set
    symbolic_learning_mode: str = "auto"  # auto | single | boosting | bagging | multi


class RunSummary(BaseModel):
    run_id: str
    dataset_name: str
    classifier: str
    status: str  # "running" | "paused" | "stopped" | "completed" | "error"
    stages_total: int
    last_stage: int | None = None
    error: str | None = None


class StateEditRequest(BaseModel):
    item_id: str
    fields: dict


class GoalRequest(BaseModel):
    """Set the active LCA-inspired GoalCell (top-down attention director)."""
    directive: str
    focus_features: list[str] = []
    ignore_features: list[str] = []
    prefer_domains: list[str] = []
    prefer_relations: list[str] = []
    goal_keywords: list[str] = []
    confidence: float = 1.0
