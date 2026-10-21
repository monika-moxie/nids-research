"""FastAPI application for the practical NIDS deployment demonstrator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from deployment_intelligence.schemas import (
    AnalysisResponse,
    FlowRequest,
    IntelligenceResponse,
    MetadataResponse,
    PredictionResponse,
)
from deployment_intelligence.intelligence import IntelligenceOrchestrator
from deployment_intelligence.service import ArtifactBackedNidsService
from shared.config import DatasetConfig


STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ArtifactBackedNidsService()
    service.load()
    app.state.nids_service = service
    yield


app = FastAPI(
    title="NIDS Deployment Intelligence API",
    version="0.1.0",
    description=(
        "Practical Phase 8 demonstrator. It serves the saved research baseline "
        "and reports certificates only in preprocessed feature space."
    ),
    lifespan=lifespan,
)


def get_service(request: Request) -> ArtifactBackedNidsService:
    return request.app.state.nids_service


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    get_service(request)
    return {"status": "ok", "component": "nids-deployment-intelligence"}


@app.get("/metadata", response_model=MetadataResponse)
def metadata(request: Request) -> dict[str, object]:
    return get_service(request).metadata()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FlowRequest, request: Request) -> dict[str, object]:
    try:
        return get_service(request).predict(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(payload: FlowRequest, request: Request) -> dict[str, object]:
    try:
        return get_service(request).analyze(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/intelligence", response_model=IntelligenceResponse)
def intelligence(payload: FlowRequest, request: Request) -> dict[str, object]:
    try:
        return IntelligenceOrchestrator(get_service(request)).analyze_flagged_flow(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/demo-flow")
def demo_flow(request: Request) -> dict[str, object]:
    """Return one local test-set flow for the practical dashboard demonstration."""

    service = get_service(request)
    frame = pd.read_csv(DatasetConfig().test_path, usecols=service.input_fields, nrows=1)
    row = frame.iloc[0].to_dict()
    return {"features": {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}}


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
