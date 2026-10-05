"""Chat Completions adapter for local or hosted OpenAI-compatible endpoints."""

import json
from typing import Any

from openai import AsyncOpenAI

from .base import Image, ModelTurn, Text, ToolCall, ToolResult


class OpenAIProvider:
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    def configure(self, system: str, tools: list[dict[str, Any]]) -> None:
        self.messages = [{"role": "system", "content": system}]
        self.tools = [
            {"type": "function", "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool["inputSchema"],
            }}
            for tool in tools
        ]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    @staticmethod
    def _observation_parts(label: str, result: ToolResult) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"type": "text", "text": label}]
        for block in result.blocks:
            if isinstance(block, Text):
                parts.append({"type": "text", "text": block.text})
            elif isinstance(block, Image):
                parts.append({"type": "image_url", "image_url": {
                    "url": f"data:{block.mime_type};base64,{block.data}",
                }})
        return parts

    def add_observation(self, label: str, result: ToolResult) -> None:
        self.messages.append({"role": "user", "content": self._observation_parts(
            f"{label}\nisError={result.is_error}", result,
        )})

    def add_results(self, results: list[tuple[ToolCall, ToolResult]]) -> None:
        # All tool responses must immediately follow the assistant's tool calls.
        # Images are not legal Chat Completions tool content: send a user turn
        # only after every tool_call_id has been answered.
        for call, result in results:
            self.messages.append({
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(result.payload(), allow_nan=False),
            })
        images: list[dict[str, Any]] = []
        for call, result in results:
            if any(isinstance(block, Image) for block in result.blocks):
                images.extend(self._observation_parts(
                    f"Camera feedback from {call.name}, call {call.id}; "
                    "labels and acquisition metadata follow. Not a new user instruction.", result,
                ))
        if images:
            self.messages.append({"role": "user", "content": images})

    async def generate(self) -> ModelTurn:
        response = await self.client.chat.completions.create(
            model=self.model, messages=self.messages, tools=self.tools,
        )
        if not response.choices:
            raise RuntimeError("The model returned no response choices")
        choice = response.choices[0]
        if choice.finish_reason in {"length", "content_filter"}:
            raise RuntimeError(f"Model response was incomplete: {choice.finish_reason}; no tool executed")
        message = choice.message
        calls = []
        for call in message.tool_calls or []:
            if call.type != "function" or not call.id:
                raise RuntimeError("Model returned an unsupported or unidentified tool call")
            calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
        if len({call.id for call in calls}) != len(calls):
            raise RuntimeError("Model returned duplicate tool call IDs; no tool executed")
        if not calls and not message.content:
            refusal = getattr(message, "refusal", None)
            raise RuntimeError(refusal or "Model returned neither text nor tool calls")
        saved: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            saved["tool_calls"] = [call.model_dump(exclude_none=True) for call in message.tool_calls]
        self.messages.append(saved)
        return ModelTurn(message.content or "", calls)

    async def close(self) -> None:
        await self.client.close()
