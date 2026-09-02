from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import ask_agent


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


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    LOGGER.info("api_request | question=%s", question)

    try:
        result = ask_agent(question)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("api_request_failed | error=%s", exc)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}") from exc

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
