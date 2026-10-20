"""Offline scripted provider and content-free timing of actual provider requests."""

import json
from time import perf_counter

from openai.types.responses import Response

from app.agents.llm import LLM, Settings
from app.agents.observability import argument_summary
from app.agents.schemas import AgentError
from app.tools.registry import reference
from app.tools.schemas import ARGUMENT_MODELS


class ScriptedClient:
    """Fixed transcripts independent of golden answers; never creates a network client."""

    def __init__(self, steps: list[dict], index):
        self.steps = iter(steps)
        self.index = index
        self.responses = self
        self.count = 0

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.count += 1
        try:
            step = next(self.steps)
        except StopIteration as error:
            raise AgentError("Offline transcript exhausted.") from error
        if "error" in step:
            raise AgentError(
                "Injected provider failure.", retryable=step["error"] == "transient"
            )
        if step.get("interrupt"):
            raise KeyboardInterrupt
        output = []
        for number, call in enumerate(step.get("calls", [])):
            output.append(
                {
                    "type": "function_call",
                    "name": call["name"],
                    "arguments": call["arguments"]
                    if isinstance(call["arguments"], str)
                    else json.dumps(call["arguments"]),
                    "call_id": f"call_{self.count}_{number}",
                }
            )
        if "answer" in step or "raw" in step:
            answer = {
                "answer": step.get("answer", ""),
                "answer_type": step.get("answer_type", "document"),
                "sources": [
                    reference(self.index.get_record(rid)).model_dump(mode="json")
                    for rid in step.get("sources", [])
                ],
                "warnings": [],
            }
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{self.count}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": step.get("raw", json.dumps(answer)),
                            "annotations": [],
                        }
                    ],
                }
            )
        return Response.model_validate(
            {
                "id": f"offline_{self.count}",
                "created_at": 0,
                "model": "offline-script",
                "object": "response",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": step.get("status", "completed"),
            }
        )


class MeasuredLLM:
    def __init__(self, llm):
        self.llm = llm
        self.settings = llm.settings
        self.trace = []
        self.latencies = []
        self.usage_complete = True

    def respond(self, inputs, **kwargs):
        iteration = len(self.latencies) + 1
        for item in inputs:
            if item.get("type") == "function_call_output":
                payload = json.loads(item["output"])
                data = payload.get("data", {})
                self.trace.append(
                    {
                        "event": "tool_observed",
                        "iteration": iteration,
                        "success": payload["ok"],
                        "complete": data.get("complete"),
                        "index_complete": data.get("index_complete"),
                    }
                )
        self.trace.append(
            {
                "event": "llm_requested",
                "iteration": iteration,
                "continuation": kwargs.get("previous_response_id") is not None,
            }
        )
        started = perf_counter()
        try:
            response = self.llm.respond(inputs, **kwargs)
            self.usage_complete &= response.usage is not None
            for item in response.output:
                if item.type == "function_call":
                    try:
                        ARGUMENT_MODELS[item.name].model_validate_json(item.arguments)
                        validated = True
                    except (KeyError, ValueError):
                        validated = False
                    self.trace.append(
                        {
                            "event": "tool_requested",
                            "iteration": iteration,
                            "tool": item.name
                            if item.name
                            in {"search_documents", "get_record", "list_documents"}
                            else "unknown",
                            "arguments": argument_summary(item.arguments),
                            "arguments_valid": validated,
                        }
                    )
            return response
        except (AgentError, KeyboardInterrupt):
            self.usage_complete = False
            raise
        finally:
            self.latencies.append(perf_counter() - started)


def offline_llm(steps, index):
    settings = Settings(api_key="offline-unused", model="offline-script", max_retries=0)
    return LLM(settings, ScriptedClient(steps, index))
