import asyncio
import json
import re
from typing import Dict, List, Tuple

import httpx
from asyncllm import AsyncLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

TOOL_PATTERN = re.compile(r"<tool:(\w+)\s*(\{.*?\})?>")


async def parse_tool_calls(text: str) -> List[Tuple[str, Dict]]:
    """Extract tool calls from model output."""
    calls = []
    for match in TOOL_PATTERN.finditer(text):
        name = match.group(1)
        args_str = match.group(2) or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        calls.append((name, args))
    return calls


async def call_tool_api(name: str, params: Dict) -> str:
    """Call an external API asynchronously based on the tool name."""
    url = f"https://api.example.com/{name}"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=params)
        resp.raise_for_status()
        return resp.text


def compute_reward(tool_outputs: Dict[str, str]) -> float:
    """Dummy reward: 1 if calculator returns 4."""
    return float(tool_outputs.get("calculator", "").strip() == "4")


async def generate_with_tools(llm: AsyncLLM, prompt: str) -> Tuple[str, Dict[str, str]]:
    """Run one generation step with tool calls and masking."""
    first = await llm.acomplete(prompt)
    first_text = first["choices"][0]["message"]["content"]

    tool_calls = await parse_tool_calls(first_text)
    tool_outputs = {}
    for name, params in tool_calls:
        tool_outputs[name] = await call_tool_api(name, params)

    masked = first_text
    for i, _ in enumerate(tool_calls):
        masked += f" <tool_output_{i}>"

    second = await llm.acomplete(prompt + masked)
    final_text = second["choices"][0]["message"]["content"]
    return final_text, tool_outputs


async def main() -> None:
    llm = AsyncLLM(engine="vllm", endpoint="http://localhost:8000")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    ppo = PPOTrainer(PPOConfig(), model, tokenizer=tokenizer)

    prompts = ['Calculate 2+2 using <tool:calculator {"expression": "2+2"}>.']

    for prompt in prompts:
        final, tools = await generate_with_tools(llm, prompt)
        reward = compute_reward(tools)
        query = tokenizer(prompt, return_tensors="pt").input_ids[0]
        response = tokenizer(final, return_tensors="pt").input_ids[0]
        ppo.step([query], [response], [reward])
        print(f"Reward: {reward}, output: {final}")

    ppo.save_pretrained("ppo_model")


if __name__ == "__main__":
    asyncio.run(main())
