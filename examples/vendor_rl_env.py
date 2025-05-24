"""Example vendor environment for tool based RL.

This environment exposes tool definitions and reward logic so that the
LLM call can be handled externally.  The step method processes the
model's response, executes any requested tools and returns a reward.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple

import gymnasium as gym

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


class EchoTool(BaseTool):
    """Simple tool that echoes the ``text`` argument."""

    def __init__(self) -> None:
        schema = OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="echo",
                description="Return the input text",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "text": OpenAIFunctionPropertySchema(
                            type="string", description="Text to echo"
                        )
                    },
                    required=["text"],
                ),
            ),
        )
        super().__init__({}, schema)

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
        return parameters.get("text", ""), 0.0, {}


class LLMToolEnv(gym.Env):
    """Gym environment that only manages tools and reward logic."""

    metadata = {"render_modes": []}

    def __init__(self, tools: Dict[str, BaseTool], reward_fn) -> None:
        super().__init__()
        self.tools = tools
        self.reward_fn = reward_fn
        self._messages: List[Dict[str, str]] = []

    def reset(self, prompt: str, **kwargs) -> List[Dict[str, str]]:
        self._messages = [{"role": "user", "content": prompt}]
        return self._messages

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for inclusion in the LLM request."""
        return [t.get_openai_tool_schema().model_dump() for t in self.tools.values()]

    def step(self, llm_message: Dict[str, Any]):
        return asyncio.get_event_loop().run_until_complete(self._astep(llm_message))

    async def _astep(self, llm_message: Dict[str, Any]):
        self._messages.append({"role": "assistant", "content": llm_message.get("content", "")})

        tool_outputs = {}
        for call in llm_message.get("tool_calls", []):
            tool = self.tools.get(call["function"]["name"])
            if tool is None:
                continue
            params = json.loads(call["function"]["arguments"])
            output, _, _ = await tool.execute("session", params)
            self._messages.append({"role": "tool", "name": tool.name, "content": output})
            tool_outputs[tool.name] = output

        reward = self.reward_fn(tool_outputs, llm_message.get("content", ""))
        done = True
        info = {"tool_outputs": tool_outputs}
        return self._messages, reward, done, info

    async def close(self):
        for t in self.tools.values():
            await t.release("session")


def simple_reward(outputs: Dict[str, str], text: str) -> float:
    return float("hello" in text.lower())


async def main() -> None:
    from asyncllm import AsyncLLM

    env = LLMToolEnv({"echo": EchoTool()}, simple_reward)
    llm = AsyncLLM(engine="vllm", endpoint="http://localhost:8000")
    messages = env.reset("Say hello using <tool:echo {\"text\": \"world\"}>")
    schemas = env.get_tool_schemas()
    completion = await llm.acomplete(messages=messages, tools=schemas)
    msg = completion["choices"][0]["message"]
    obs, reward, done, info = env.step(msg)
    print("reward=", reward)
    print("conversation:")
    for m in obs:
        print(m)


if __name__ == "__main__":
    asyncio.run(main())
