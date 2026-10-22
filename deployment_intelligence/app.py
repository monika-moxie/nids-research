"""FastAPI application for the practical NIDS deployment demonstrator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from deployment_intelligence.schemas import (
    AnalysisResponse,
    BatchScoreResponse,
    FlowRequest,
    IntelligenceResponse,
    MetadataResponse,
    PredictionResponse,
)
from deployment_intelligence.intelligence import IntelligenceOrchestrator
from deployment_intelligence.service import ArtifactBackedNidsService
from shared.config import DatasetConfig


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BATCH_ROWS = 500
MAX_BATCH_BYTES = 2 * 1024 * 1024


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


@app.post("/batch-score", response_model=BatchScoreResponse)
async def batch_score(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    """Screen a bounded CSV batch; deep explanation stays analyst-selected."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload a CSV file.")

    content = await file.read(MAX_BATCH_BYTES + 1)
    if len(content) > MAX_BATCH_BYTES:
        raise HTTPException(status_code=413, detail="CSV is limited to 2 MB for this local demonstrator.")

    try:
        frame = pd.read_csv(BytesIO(content), nrows=MAX_BATCH_ROWS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file could not be read as a CSV.") from exc

    if frame.empty:
        raise HTTPException(status_code=422, detail="The CSV contains no flow rows.")
    if len(frame) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=422, detail=f"CSV is limited to {MAX_BATCH_ROWS} flow rows.")

    service = get_service(request)
    expected = set(service.input_fields)
    excluded = {"id", "label", "attack_cat"}
    supplied = set(frame.columns)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected - excluded)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown columns: {', '.join(unknown)}")
        raise HTTPException(status_code=422, detail="Invalid CSV schema; " + "; ".join(details))

    raw_frame = frame.loc[:, service.input_fields].astype(object).where(pd.notna(frame.loc[:, service.input_fields]), None)
    raw_flows = raw_frame.to_dict(orient="records")
    predictions = service.predict_batch(raw_flows)
    alerts = [
        {"row_number": index + 2, **prediction, "features": raw_flows[index]}
        for index, prediction in enumerate(predictions)
        if prediction["predicted_label"] == "attack"
    ]
    return {
        "total_rows": len(raw_flows),
        "flagged_count": len(alerts),
        "decision_threshold": 0.5,
        "alerts": alerts,
    }


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
