"""Interactive VLM function-calling agent using an OMX MCP server."""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from llm_providers import create_provider
from llm_providers.base import Text, ToolResult, validate_arguments

MOTION_TOOLS = {"move_robot", "move_cartesian", "set_joint_positions", "control_gripper", "apply_named_preset",
                "go_to_initial_position", "go_to_post_grasp_position"}


@asynccontextmanager
async def connect(args):
    if args.mcp_url:
        context = sse_client(args.mcp_url) if args.transport == "sse" else streamablehttp_client(args.mcp_url)
    else:
        server_args = [str(Path(__file__).with_name("mcp_robot_server.py"))]
        if args.dry_run:
            server_args.append("--dry-run")
        if args.config:
            server_args.extend(["--config", str(Path(args.config).resolve())])
        context = stdio_client(StdioServerParameters(command=sys.executable, args=server_args, env=dict(os.environ)))
    async with context as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session


class AIAgent:
    def __init__(self, session, provider, max_steps: int = 20, viewer=None):
        self.session = session
        self.provider = provider
        self.max_steps = max_steps
        self.viewer = viewer
        self.tools = {}

    async def initialize(self):
        listed = await self.session.list_tools()
        self.tools = {tool.name: tool.model_dump(by_alias=True, exclude_none=True) for tool in listed.tools}
        instructions = ToolResult.from_mcp(await self.session.call_tool("get_initial_instructions", {}))
        if instructions.is_error:
            raise RuntimeError(instructions.payload()["text"])
        system = "\n".join(b.text for b in instructions.blocks if isinstance(b, Text))
        self.provider.configure(system, list(self.tools.values()))

    async def observe(self) -> ToolResult:
        result = ToolResult.from_mcp(await self.session.call_tool("get_robot_state", {}))
        self.provider.add_observation("Current robot state and camera views", result)
        if self.viewer:
            self.viewer.update(result)
        return result

    async def execute_calls(self, calls):
        results = []
        batch_motion = sum(call.name in MOTION_TOOLS for call in calls) > 1
        stop_requested = any(call.name == "stop_robot" for call in calls)
        blocked = False
        for call in calls:
            try:
                if call.name not in self.tools:
                    raise ValueError(f"Unknown tool: {call.name}")
                if call.name in MOTION_TOOLS and (batch_motion or blocked or stop_requested):
                    raise ValueError("Motion skipped: use one motion per turn, inspect errors/feedback before continuing")
                arguments = validate_arguments(call.arguments)
                Draft202012Validator(self.tools[call.name]["inputSchema"]).validate(arguments)
                result = ToolResult.from_mcp(await self.session.call_tool(call.name, arguments))
            except Exception as exc:
                result = ToolResult.error(str(exc))
            blocked = blocked or result.is_error
            results.append((call, result))
            print(f"Tool {call.name}: {result.payload()['text']}")
            if self.viewer:
                self.viewer.update(result)
        self.provider.add_results(results)
        return results

    async def run_task(self, prompt: str):
        self.provider.add_user(prompt)
        initial = await self.observe()
        if initial.is_error:
            print("Cannot start task: current robot state/camera observation failed.")
            return
        for _ in range(self.max_steps):
            turn = await self.provider.generate()
            if turn.text:
                print(f"Assistant: {turn.text}")
            if not turn.calls:
                return
            results = await self.execute_calls(turn.calls)
            if any(result.is_error for _, result in results):
                print("Task stopped after a tool error; inspect the result before issuing another task.")
                return
        print(f"Task stopped at the {self.max_steps}-step limit.")


async def run(args):
    provider = create_provider(args.provider, args.model, args.api_key, args.base_url)
    viewer = None
    if args.show_images:
        from agent_utils import ImageViewer
        viewer = ImageViewer()
    try:
        async with connect(args) as session:
            agent = AIAgent(session, provider, args.max_steps, viewer)
            await agent.initialize()
            if args.prompt:
                await agent.run_task(args.prompt)
                return
            print("OMX agent ready. Enter a task; /state, /stop, /reset, /quit.")
            while True:
                prompt = await asyncio.to_thread(input, "You: ")
                if prompt.strip() in {"/quit", "exit", "quit"}:
                    break
                if prompt.strip() == "/stop":
                    print((await session.call_tool("stop_robot", {})).content)
                elif prompt.strip() == "/state":
                    print((await agent.observe()).payload()["text"])
                elif prompt.strip() == "/reset":
                    await agent.initialize()
                    print("Conversation reset.")
                elif prompt.strip():
                    try:
                        await agent.run_task(prompt)
                    except Exception as exc:
                        print(f"Task failed: {exc}. Use /reset to clear conversation.")
    finally:
        await provider.close()
        if viewer:
            viewer.close()


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "gemini"], default=os.environ.get("LLM_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint, including /v1")
    parser.add_argument("--mcp-url", help="Use an existing MCP server instead of launching one via stdio")
    parser.add_argument("--transport", choices=["streamable-http", "sse"], default="streamable-http")
    parser.add_argument("--dry-run", action="store_true", help="Start a hardware-free MCP subprocess")
    parser.add_argument("--config")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--prompt", help="Run a single task then exit")
    parser.add_argument("--show-images", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.mcp_url and (args.dry_run or args.config):
        parser.error("Configure dry-run/config on the existing MCP server, or omit --mcp-url")
    try:
        asyncio.run(run(args))
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
