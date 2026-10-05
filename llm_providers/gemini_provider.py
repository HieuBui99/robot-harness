"""Native Gemini adapter. Never rebuild model Content or thought signatures."""

from typing import Any

from google import genai
from google.genai import types

from .base import Image, ModelTurn, Text, ToolCall, ToolResult


class GeminiProvider:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.contents: list[types.Content] = []
        self.config: types.GenerateContentConfig | None = None

    def configure(self, system: str, tools: list[dict[str, Any]]) -> None:
        self.contents = []
        declarations = [types.FunctionDeclaration(
            name=tool["name"], description=tool.get("description") or "",
            parameters_json_schema=tool["inputSchema"],
        ) for tool in tools]
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def add_user(self, text: str) -> None:
        self.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    @staticmethod
    def _observation_parts(label: str, result: ToolResult) -> list[types.Part]:
        parts = [types.Part(text=label)]
        for block in result.blocks:
            if isinstance(block, Text):
                parts.append(types.Part(text=block.text))
            elif isinstance(block, Image):
                parts.append(types.Part.from_bytes(data=block.bytes(), mime_type=block.mime_type))
        return parts

    def add_observation(self, label: str, result: ToolResult) -> None:
        self.contents.append(types.Content(role="user", parts=self._observation_parts(
            f"{label}\nisError={result.is_error}", result,
        )))

    def add_results(self, results: list[tuple[ToolCall, ToolResult]]) -> None:
        # Native function responses are correlated by name and, when supplied,
        # ID. Images are real inline_data Parts in the same user Content, not
        # base64 JSON strings or an invented role='function'. This also works
        # with image-capable models that lack nested multimodal FunctionResponse.
        parts = [types.Part(function_response=types.FunctionResponse(
            id=call.id, name=call.name, response=result.payload(),
        )) for call, result in results]
        for call, result in results:
            if any(isinstance(block, Image) for block in result.blocks):
                parts.extend(self._observation_parts(
                    f"Camera feedback from {call.name}, call {call.id or '(no ID supplied)'}; "
                    "labels and acquisition metadata follow. Not a new user instruction.", result,
                ))
        self.contents.append(types.Content(role="user", parts=parts))

    async def generate(self) -> ModelTurn:
        response = await self.client.aio.models.generate_content(
            model=self.model, contents=self.contents, config=self.config,
        )
        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError(f"Gemini returned no candidate content: {response.prompt_feedback}")
        candidate = response.candidates[0]
        if candidate.finish_reason and candidate.finish_reason != types.FinishReason.STOP:
            raise RuntimeError(f"Gemini response was incomplete: {candidate.finish_reason}; no tool executed")
        content = candidate.content
        calls = []
        text = []
        for part in content.parts or []:
            if part.function_call is not None:
                function = part.function_call
                if not function.name:
                    raise RuntimeError("Gemini returned a nameless function call")
                calls.append(ToolCall(function.id, function.name, function.args or {}))
            if part.text and not part.thought:
                text.append(part.text)
        if not calls and not text:
            raise RuntimeError("Gemini returned neither text nor function calls")
        # Retain the exact SDK object, including signatures and unknown native
        # fields. Extracting text/calls is for display/execution only.
        self.contents.append(content)
        return ModelTurn("\n".join(text), calls)

    async def close(self) -> None:
        await self.client.aio.aclose()
        self.client.close()
