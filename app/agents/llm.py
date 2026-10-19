"""Centralized OpenAI Responses configuration and safe provider failure handling."""

import logging
import os

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from openai.types.responses import Response, ResponseInputParam
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.tools.schemas import TOOL_DEFINITIONS

from .prompts import DEVELOPER_PROMPT, SYSTEM_PROMPT
from .schemas import AgentError, AgentResponse

logger = logging.getLogger(__name__)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    api_key: SecretStr = Field(repr=False)
    model: str = Field(min_length=1, pattern=r"\S")
    timeout: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=1, ge=0, le=2)
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)

    @classmethod
    def from_env(cls):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "").strip()
        if not key:
            raise AgentError("Set OPENAI_API_KEY before asking a question.")
        if not model:
            raise AgentError(
                "Set OPENAI_MODEL to a model supporting tools and structured outputs."
            )
        try:
            return cls(
                api_key=SecretStr(key),
                model=model,
                timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
                max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "1")),
                max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096")),
            )
        except (ValueError, ValidationError) as error:
            raise AgentError(
                "Invalid OpenAI timeout, retry, or token configuration."
            ) from error


class LLM:
    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.settings = settings
        self.client = (
            client
            if client is not None
            else OpenAI(
                api_key=settings.api_key.get_secret_value(),
                timeout=settings.timeout,
                max_retries=settings.max_retries,
            )
        )

    def respond(
        self,
        inputs: ResponseInputParam,
        *,
        previous_response_id: str | None = None,
        final: bool = False,
    ) -> Response:
        try:
            response = self.client.responses.create(
                model=self.settings.model,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "developer", "content": DEVELOPER_PROMPT}, *inputs],
                tools=TOOL_DEFINITIONS,
                tool_choice="none" if final else "auto",
                parallel_tool_calls=True,
                previous_response_id=previous_response_id,
                max_output_tokens=self.settings.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "syden_answer",
                        "strict": True,
                        "schema": AgentResponse.model_json_schema(),
                    }
                },
            )
        except AuthenticationError as error:
            logger.warning("OpenAI authentication failed")
            raise AgentError(
                "OpenAI authentication failed; check OPENAI_API_KEY."
            ) from error
        except RateLimitError as error:
            logger.warning("OpenAI rate limit or quota reached")
            raise AgentError(
                "OpenAI rate limit or quota reached; retry later."
            ) from error
        except APITimeoutError as error:
            logger.warning("OpenAI request timed out")
            raise AgentError("OpenAI request timed out.") from error
        except APIConnectionError as error:
            logger.warning("OpenAI connection failed")
            raise AgentError("Could not connect to OpenAI.") from error
        except APIStatusError as error:
            logger.warning("OpenAI request failed status=%s", error.status_code)
            raise AgentError(
                "OpenAI rejected or failed the request; check model access and configuration."
            ) from error
        except APIError as error:
            logger.warning("OpenAI returned an invalid API response")
            raise AgentError("OpenAI returned an invalid API response.") from error
        logger.info(
            "OpenAI model=%s status=%s input_tokens=%s output_tokens=%s",
            self.settings.model,
            response.status,
            response.usage.input_tokens if response.usage else None,
            response.usage.output_tokens if response.usage else None,
        )
        return response
