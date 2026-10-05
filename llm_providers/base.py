"""Provider-neutral tool results; conversation wire formats stay provider-owned."""

import base64
import json
import math
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Image:
    data: str
    mime_type: str

    def bytes(self) -> bytes:
        return base64.b64decode(self.data, validate=True)


@dataclass
class ToolResult:
    blocks: list[Text | Image] = field(default_factory=list)
    is_error: bool = False

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls([Text(message)], is_error=True)

    def payload(self) -> dict[str, Any]:
        return {
            "isError": self.is_error,
            "text": "\n".join(block.text for block in self.blocks if isinstance(block, Text)),
            "image_count": sum(isinstance(block, Image) for block in self.blocks),
        }

    @classmethod
    def from_mcp(cls, result: Any) -> "ToolResult":
        blocks: list[Text | Image] = []
        is_error = bool(result.isError)
        for block in result.content:
            if block.type == "text":
                blocks.append(Text(block.text))
            elif block.type == "image":
                image = Image(block.data, block.mimeType)
                try:
                    if not image.mime_type.startswith("image/") or not image.bytes():
                        raise ValueError("empty image or non-image MIME type")
                except (ValueError, TypeError) as exc:
                    is_error = True
                    blocks.append(Text(f"Camera image could not be decoded: {exc}"))
                else:
                    blocks.append(image)
            else:
                is_error = True
                blocks.append(Text(f"Unsupported MCP content type: {block.type}"))
        if not blocks and result.structuredContent is not None:
            blocks.append(Text(json.dumps(result.structuredContent, allow_nan=False)))
        return cls(blocks, is_error)


def validate_arguments(value: Any) -> dict[str, Any]:
    """Reject malformed/non-finite arguments, never repair them into a command."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be a JSON object")

    def finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Tool arguments must contain only finite numbers")
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("Tool argument keys must be strings")
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("Tool arguments must be JSON values")

    finite(value)
    return value


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    name: str
    arguments: Any


@dataclass
class ModelTurn:
    text: str
    calls: list[ToolCall] = field(default_factory=list)


class Provider(Protocol):
    def configure(self, system: str, tools: list[dict[str, Any]]) -> None: ...
    def add_user(self, text: str) -> None: ...
    def add_observation(self, label: str, result: ToolResult) -> None: ...
    def add_results(self, results: list[tuple[ToolCall, ToolResult]]) -> None: ...
    async def generate(self) -> ModelTurn: ...
    async def close(self) -> None: ...
