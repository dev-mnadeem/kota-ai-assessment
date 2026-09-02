from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import ask_agent
from src.datasets import status as dataset_status, build as build_dataset, fingerprint
import os
import hmac
import json
from fastapi.responses import JSONResponse


load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

LOGGER = logging.getLogger("ashe_api")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

app = FastAPI(title="ASHE Pension Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    tool_calls: list[dict[str, object]]
    critic: dict[str, object] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    dataset = dataset_status(os.getenv("ASHE_DB_PATH", "data/ashe.db"))
    dataset["agent_configured"] = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return JSONResponse(status_code=200 if dataset["ready"] else 503, content=dataset)


def require_refresh_access(authorization: str | None):
    expected = os.getenv("ASHE_REFRESH_TOKEN", "")
    if not expected or not authorization or not hmac.compare_digest(authorization, "Bearer " + expected):
        raise HTTPException(status_code=401, detail="Dataset refresh authorization required")


@app.get("/admin/source-status")
def source_status(authorization: str | None = Header(default=None)):
    require_refresh_access(authorization)
    path = Path(os.getenv("ASHE_SOURCE_MANIFEST", "sources/ons.json"))
    manifest = json.loads(path.read_text())
    current = dataset_status(os.getenv("ASHE_DB_PATH", "data/ashe.db"))
    return {"refresh_needed":current.get("manifest_sha256") != fingerprint(path),
            "approved_version":manifest["version"], "current":current}


@app.post("/admin/refresh")
def refresh(authorization: str | None = Header(default=None)):
    require_refresh_access(authorization)
    try:
        return build_dataset(os.getenv("ASHE_SOURCE_MANIFEST", "sources/ons.json"),
                             os.getenv("ASHE_DB_PATH", "data/ashe.db"))
    except Exception:
        LOGGER.exception("dataset_refresh_failed")
        raise HTTPException(status_code=503, detail="Dataset refresh failed; the previous snapshot was preserved.")


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not dataset_status(os.getenv("ASHE_DB_PATH", "data/ashe.db"))["ready"]:
        raise HTTPException(status_code=503, detail="Dataset is not ready; build a verified snapshot first.")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="Dataset is available, but the model provider is not configured.")
    LOGGER.info("api_request | question_length=%s", len(question))

    try:
        result = ask_agent(question)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("api_request_failed | error=%s", exc)
        raise HTTPException(status_code=500, detail="The agent could not complete this request. Please try again.") from exc

    tool_calls = result.get("tool_calls") or []
    LOGGER.info(
        "api_response | tools_called=%s | sources=%s",
        [str(call.get("tool")) for call in tool_calls],
        result.get("sources", []),
    )

    return AskResponse(
        answer=str(result.get("answer", "")),
        sources=[str(item) for item in (result.get("sources") or [])],
        tool_calls=[dict(call) for call in tool_calls],
        critic=dict(result["critic"]) if result.get("critic") is not None else None,
    )
