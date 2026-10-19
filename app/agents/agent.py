"""One question, at most one batch of read-only calls, then a grounded answer."""

from openai.types.responses import Response, ResponseInputParam
from pydantic import ValidationError

from app.retrieval import DocumentIndex
from app.tools.registry import ToolRegistry

from .llm import LLM
from .schemas import AgentError, AgentRequest, AgentResponse


def check_completion(response: Response) -> None:
    if response.status != "completed":
        raise AgentError(
            "OpenAI response was incomplete or failed; no answer was accepted."
        )
    for item in response.output:
        if item.type == "message" and any(
            part.type == "refusal" for part in item.content
        ):
            raise AgentError("The model declined this request.")


def parse_answer(response: Response, registry: ToolRegistry) -> AgentResponse:
    try:
        answer = AgentResponse.model_validate_json(response.output_text)
    except ValidationError as error:
        raise AgentError(
            "The model returned an empty or invalid structured answer."
        ) from error
    if any(source not in registry.sources for source in answer.sources):
        raise AgentError("The model cited a source that was not supplied by retrieval.")
    if answer.answer_type == "document" and not answer.sources:
        raise AgentError("Document answers require supporting retrieved sources.")
    if answer.answer_type == "general" and (answer.sources or registry.sources):
        raise AgentError(
            "An answer using document evidence must identify itself and cite its sources."
        )
    answer.warnings = list(dict.fromkeys([*answer.warnings, *registry.warnings]))
    return answer


def ask(
    question: str,
    index: DocumentIndex,
    llm: LLM,
    *,
    max_tool_output_bytes: int = 60000,
) -> AgentResponse:
    request = AgentRequest(question=question)
    registry = ToolRegistry(index, max_tool_output_bytes)
    response = llm.respond([{"role": "user", "content": request.question}])
    check_completion(response)
    calls = [item for item in response.output if item.type == "function_call"]
    if calls:
        if len(calls) > 8 or len({call.call_id for call in calls}) != len(calls):
            raise AgentError("The model returned too many or duplicate tool calls.")
        outputs: ResponseInputParam = [
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": registry.execute(call.name, call.arguments),
            }
            for call in calls
        ]
        # The response ID preserves reasoning and every call item in this one request.
        response = llm.respond(outputs, previous_response_id=response.id, final=True)
        check_completion(response)
        if any(item.type == "function_call" for item in response.output):
            raise AgentError(
                "A second tool round was requested; Phase 3 supports one batch only. Narrow the question."
            )
    return parse_answer(response, registry)
