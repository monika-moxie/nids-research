# Deployment Intelligence Layer

## Scope

This is Phase 8: a practical applicability demonstration, deliberately
separate from the core research contribution in Phases 1 through 7.

## Backend Foundation

The FastAPI backend loads the actual Phase 1 model and preprocessor artifacts.
It accepts one complete raw UNSW-NB15 flow, rejects schema drift, returns the
attack probability, and can run flow-specific randomized smoothing.

The certificate is an L2 result in preprocessed feature space. It must not be
described as a guarantee over every valid raw network-flow transformation.

`POST /intelligence` is a bounded analyst workflow. It calls the model and
certificate service, calls a LIME explanation tool, applies an explicit
escalate-or-suppress policy, writes a short plain-English summary, and creates
a draft incident report. It cannot take autonomous external security actions.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn deployment_intelligence.app:app --reload
```

Open `http://127.0.0.1:8000/` for the analyst dashboard or
`http://127.0.0.1:8000/docs` for API documentation.

## Dashboard Workflow

1. Load a local demo flow or paste a complete raw-flow JSON object.
2. Run the intelligence workflow.
3. Inspect probability, certificate, LIME evidence, decision, and report.
4. Treat the resulting report as an analyst draft, not an automatic response.

## Batch Triage

The dashboard can screen a CSV containing up to 500 flows. The CSV must contain
the 42 raw model-input columns; the standard UNSW-NB15 `id`, `label`, and
`attack_cat` columns may also be present and are ignored. Batch screening only
uses the clean saved model to build a queue of flows at or above the 0.50
attack threshold. Selecting one queue item then runs the existing detailed
certificate, LIME, decision, and report workflow.

This separation is intentional: LIME and randomized smoothing are relatively
expensive per flow, while the queue is intended to be a fast screening step.
