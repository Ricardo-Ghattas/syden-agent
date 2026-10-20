import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai.types.responses import Response
from pydantic import SecretStr

from app.agents.llm import LLM, Settings
from app.retrieval import DocumentIndex


@pytest.fixture
def index():
    return DocumentIndex(Path(__file__).parents[1] / "retrieval" / "fixtures")


@pytest.fixture
def client():
    client = MagicMock()
    client.with_options.return_value = client
    return client


@pytest.fixture
def llm(client):
    return LLM(Settings(api_key=SecretStr("test-only-key"), model="test-model"), client)


def reply(*, calls=(), answer=None, status="completed", refusal=False, raw=None):
    output = list(calls)
    if refusal:
        output.append(
            {
                "type": "message",
                "id": "msg_refusal",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "Cannot comply"}],
            }
        )
    elif answer is not None or raw is not None:
        output.append(
            {
                "type": "message",
                "id": "msg_answer",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": raw if raw is not None else json.dumps(answer),
                        "annotations": [],
                    }
                ],
            }
        )
    return Response.model_validate(
        {
            "id": "resp_first",
            "created_at": 0,
            "model": "test-model",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": status,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


def call(name="search_documents", arguments=None, call_id="call_1"):
    return {
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(arguments) if arguments is not None else "{}",
        "call_id": call_id,
    }


def search_args(**updates):
    return {
        "query": "2.3",
        "filters": None,
        "limit": 10,
        "offset": 0,
        "exhaustive": False,
        **updates,
    }


def answer(
    *, sources=(), answer_type="general", text="I can explain construction documents."
):
    return {
        "answer": text,
        "answer_type": answer_type,
        "sources": list(sources),
        "warnings": [],
    }
