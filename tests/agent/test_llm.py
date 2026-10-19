import logging
from unittest.mock import patch

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import ValidationError

from app.agents.llm import LLM, Settings
from app.agents.schemas import AgentError

from .conftest import answer, reply


def test_environment_and_client_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_MODEL", "selected-model")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "512")
    settings = Settings.from_env()
    with patch("app.agents.llm.OpenAI") as constructor:
        llm = LLM(settings)
    constructor.assert_called_once_with(
        api_key="test-secret", timeout=12.5, max_retries=2
    )
    assert llm.client is constructor.return_value
    assert settings.model == "selected-model"
    assert settings.max_output_tokens == 512
    assert "test-secret" not in repr(settings)


@pytest.mark.parametrize("missing", ["OPENAI_API_KEY", "OPENAI_MODEL"])
def test_missing_environment_is_clear(monkeypatch, missing):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.delenv(missing)
    with pytest.raises(AgentError, match=missing):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENAI_TIMEOUT_SECONDS", "nan"),
        ("OPENAI_TIMEOUT_SECONDS", "0"),
        ("OPENAI_MAX_RETRIES", "99"),
        ("OPENAI_MAX_RETRIES", "wrong"),
        ("OPENAI_MAX_OUTPUT_TOKENS", "-1"),
    ],
)
def test_invalid_configuration(monkeypatch, name, value):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv(name, value)
    with pytest.raises(AgentError, match="configuration"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        (AuthenticationError, "authentication"),
        (RateLimitError, "rate limit"),
        (APIStatusError, "rejected"),
        (APITimeoutError, "timed out"),
        (APIConnectionError, "connect"),
    ],
)
def test_provider_errors_are_safe_and_not_retried_again(llm, client, kind, message):
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    if issubclass(kind, APIStatusError):
        error = kind(
            "SECRET-DOCUMENT", response=httpx2.Response(401, request=request), body=None
        )
    elif kind is APITimeoutError:
        error = kind(request=request)
    else:
        error = kind(request=request, message="SECRET-DOCUMENT")
    client.responses.create.side_effect = error
    with pytest.raises(AgentError, match=message) as caught:
        llm.respond([{"role": "user", "content": "Question"}])
    assert "SECRET-DOCUMENT" not in str(caught.value)
    assert client.responses.create.call_count == 1


def test_request_contract_and_safe_observability(llm, client, caplog):
    client.responses.create.return_value = reply(answer=answer(text="CONFIDENTIAL"))
    with caplog.at_level(logging.INFO):
        llm.respond([{"role": "user", "content": "CONFIDENTIAL QUESTION"}])
        llm.respond([], previous_response_id="resp_first", final=True)
    first, second = [item.kwargs for item in client.responses.create.call_args_list]
    assert first["model"] == "test-model"
    assert first["tool_choice"] == "auto"
    assert second["tool_choice"] == "none"
    assert second["previous_response_id"] == "resp_first"
    assert first["instructions"] == second["instructions"]
    assert first["input"][0] == second["input"][0]
    assert first["text"]["format"]["strict"]
    assert "CONFIDENTIAL" not in caplog.text
    assert "test-only-key" not in caplog.text
    assert "input_tokens=100" in caplog.text


def test_direct_settings_validate_key_contract():
    with pytest.raises(ValidationError):
        Settings.model_validate({"api_key": "test", "model": "test", "timeout": -1})


def test_official_sdk_serializes_two_requests_over_mock_transport(index):
    import json

    from openai import OpenAI
    from pydantic import SecretStr

    from app.agents.agent import ask
    from app.tools.registry import reference

    from .conftest import call, search_args

    requests = []
    record = index.get_record("b" * 64 + ":39")
    assert record is not None
    citation = reference(record).model_dump(mode="json")

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            response = reply(calls=[call(arguments=search_args())])
        else:
            result = json.loads(body["input"][1]["output"])
            assert result["data"]["results"][0]["record"]["boq"]["quantity"] == -2.5
            response = reply(
                answer=answer(
                    answer_type="document",
                    sources=[citation],
                    text="Quantity -2.5, boq.xlsx, Structural, row 38.",
                )
            )
        return httpx2.Response(200, json=response.model_dump(mode="json"))

    with OpenAI(
        api_key="offline-only",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
    ) as sdk:
        llm = LLM(Settings(api_key=SecretStr("offline-only"), model="test-model"), sdk)
        response = ask("Item 2.3 quantity?", index, llm)
    assert response.sources[0].row_start == 38
    assert len(requests) == 2
    assert requests[1]["previous_response_id"] == "resp_first"
    assert requests[1]["tool_choice"] == "none"
    assert requests[0]["text"]["format"]["strict"]
    assert requests[0]["tools"][0]["type"] == "function"


@pytest.mark.parametrize(
    ("status", "attempts"), [(400, 1), (401, 1), (429, 2), (500, 2)]
)
def test_sdk_retries_only_transient_failures(status, attempts):
    from openai import OpenAI
    from pydantic import SecretStr

    calls = []

    def transport(request):
        calls.append(request)
        return httpx2.Response(
            status,
            headers={"retry-after-ms": "1"},
            json={
                "error": {"message": "private upstream payload", "type": "test_error"}
            },
        )

    with OpenAI(
        api_key="offline-only",
        max_retries=1,
        http_client=httpx2.Client(transport=httpx2.MockTransport(transport)),
    ) as sdk:
        llm = LLM(Settings(api_key=SecretStr("offline-only"), model="test-model"), sdk)
        with pytest.raises(AgentError):
            llm.respond([{"role": "user", "content": "Question"}])
    assert len(calls) == attempts
