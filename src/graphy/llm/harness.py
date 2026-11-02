"""Micro-Agent Harness with Context Pruning, Grammar-Constrained JSON, Self-Verification, and SQLite Caching."""

from __future__ import annotations

import json
import re
from typing import Optional, TypeVar
from pydantic import BaseModel, ValidationError

from graphy.core.logging import setup_logging
from graphy.llm.agents import MicroAgent
from graphy.llm.cache import LLMCache
from graphy.llm.model_manager import ModelLifecycleManager

logger = setup_logging()

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ContextPruner:
    """Prunes text payloads to fit compact local LLM context budgets while keeping math/code intact."""

    MATH_AND_CODE_PATTERN = re.compile(
        r"(\$\$.*?\$\$|\\begin\{.*?\}.*?\\end\{.*?\}|```.*?```|\$[^$\n]+\$)",
        re.DOTALL,
    )

    @classmethod
    def prune_text(cls, text: str, max_chars: int = 3600) -> str:
        """Keep text within max_chars without slicing inside a LaTeX equation or code fence."""
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        if len(cleaned) <= max_chars:
            return cleaned

        # Find atomic spans so we never cut inside an equation or code block
        safe_cut = max_chars
        for match in cls.MATH_AND_CODE_PATTERN.finditer(cleaned):
            start, end = match.span()
            if start < max_chars < end:
                # Include the full equation/code block if reasonably sized, else cut before it
                if end - start <= 800:
                    safe_cut = end
                else:
                    safe_cut = start
                break

        return cleaned[:safe_cut].rstrip() + "\n...[pruned]..."


def strip_thinking_blocks(raw_text: str) -> str:
    """Remove `<think>...</think>` blocks if a reasoning model emits them before JSON."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    # Also strip markdown ```json ... ``` fences if any
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class AgentHarness:
    """Executes MicroAgents with SQLite caching, JSON schema constraints, and deterministic retries."""

    def __init__(
        self,
        model_manager: ModelLifecycleManager,
        cache: Optional[LLMCache] = None,
        max_retries: int = 2,
        max_prompt_chars: int = 4200,
    ) -> None:
        self.model_manager = model_manager
        self.cache = cache or LLMCache()
        self.max_retries = max_retries
        self.max_prompt_chars = max_prompt_chars

    def run(
        self,
        agent: MicroAgent[InputT, OutputT],
        inp: InputT,
        bypass_cache: bool = False,
    ) -> OutputT:
        """Run a micro-agent with caching, grammar-constrained JSON schema, and semantic verification."""
        payload_dict = inp.model_dump(mode="json")
        model_id = self.model_manager.model_id_for_role(agent.role)

        cache_key = self.cache.compute_key(
            agent_name=agent.name,
            schema_version=agent.schema_version,
            model_id=model_id,
            payload=payload_dict,
        )

        if not bypass_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return agent.output_model.model_validate(cached)

        system_prompt = agent.build_system_prompt()
        base_user_prompt = ContextPruner.prune_text(
            agent.build_user_prompt(inp),
            max_chars=self.max_prompt_chars,
        )
        json_schema = agent.output_model.model_json_schema()
        image_path = agent.get_image_path(inp)

        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            if last_error is None:
                user_prompt = base_user_prompt
            else:
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    f"[CORRECTION REQUIRED - PREVIOUS ATTEMPT FAILED VALIDATION]: {last_error}\n"
                    f"Fix the issue and return strictly valid JSON matching the schema."
                )

            raw_response = self.model_manager.chat_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name=agent.name,
                role=agent.role,
                max_tokens=agent.max_tokens,
                image_path=image_path,
            )

            cleaned_json_str = strip_thinking_blocks(raw_response)

            try:
                parsed_dict = json.loads(cleaned_json_str)
                output_obj = agent.output_model.model_validate(parsed_dict)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = f"Schema validation error: {exc}"
                logger.warning(f"{agent.name} attempt {attempt + 1} failed schema validation: {last_error}")
                continue

            semantic_err = agent.validate_semantics(inp, output_obj)
            if semantic_err is not None:
                last_error = f"Semantic constraint violation: {semantic_err}"
                logger.warning(f"{agent.name} attempt {attempt + 1} failed semantic check: {last_error}")
                continue

            # Validated! Store in content-addressable SQLite cache
            self.cache.put(
                cache_key=cache_key,
                agent_name=agent.name,
                model_id=model_id,
                payload=payload_dict,
                output_data=output_obj.model_dump(mode="json"),
            )
            return output_obj

        raise ValueError(
            f"Agent '{agent.name}' failed after {self.max_retries + 1} attempts. Last error: {last_error}"
        )
