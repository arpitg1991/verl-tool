AsyncLLM RL Example
===================

This example demonstrates how to pair ``asyncllm`` with ``vllm`` and the
``trl`` library for a minimal reinforcement learning loop. Model output is
parsed for ``<tool:...>`` calls. Each tool is invoked through an API and its
result is masked before continuing the generation. The tool results are then
used to compute a reward, which updates the policy via ``PPOTrainer``.

The full script can be found in ``examples/asyncllm_tool_rl.py``. It uses
``httpx`` for asynchronous HTTP requests and assumes an ``AsyncLLM`` class
provided by the ``asyncllm`` package::

   async def main() -> None:
       llm = AsyncLLM(engine="vllm", endpoint="http://localhost:8000")
       tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
       model = AutoModelForCausalLM.from_pretrained("distilgpt2")
       ppo = PPOTrainer(PPOConfig(), model, tokenizer=tokenizer)

       prompt = "Calculate 2+2 using <tool:calculator {\"expression\": \"2+2\"}>."
       final, tools = await generate_with_tools(llm, prompt)
       reward = compute_reward(tools)
       query = tokenizer(prompt, return_tensors="pt").input_ids[0]
       response = tokenizer(final, return_tensors="pt").input_ids[0]
       ppo.step([query], [response], [reward])

The ``generate_with_tools`` helper parses tool calls, invokes APIs and masks
their outputs. Rewards can be customized according to the task. Running
``examples/run_asyncllm_tool_rl.sh`` trains the model for a few iterations and
saves it for later use.
