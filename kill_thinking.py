"""
kill_thinking.py — find the off switch for reasoning mode

Only two chat models are actually serving on this account, and both are
reasoning models. Rather than hunt for a third, this tries every known
mechanism for disabling the thinking step on each of them.

Run:
    python kill_thinking.py
"""

import os
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TIMEOUT = 90.0

MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "deepseek-ai/deepseek-v4-flash-0731",
]

QUESTION = "Do I need a referral to see a urologist at URO-CARE? Answer in two sentences."

BASE_SYSTEM = "You are a warm, concise patient care assistant for URO-CARE in Nairobi."

# Each attempt: (label, extra system text, extra_body dict)
ATTEMPTS = [
    ("baseline (no flag)", "", {}),
    ("chat_template_kwargs.thinking=False", "", {"chat_template_kwargs": {"thinking": False}}),
    ("chat_template_kwargs.enable_thinking=False", "", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("system: detailed thinking off", "detailed thinking off", {}),
    ("system: /no_think", "/no_think", {}),
    ("reasoning_effort=none", "", {"reasoning_effort": "none"}),
    ("reasoning_effort=low", "", {"reasoning_effort": "low"}),
    ("reasoning.enabled=False", "", {"reasoning": {"enabled": False}}),
]


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("FAIL: OPENAI_API_KEY not found in .env")
        return

    client = OpenAI(
        api_key=api_key,
        base_url=NVIDIA_BASE_URL,
        timeout=TIMEOUT,
        max_retries=0,
    )

    winners = []

    for model in MODELS:
        print("\n" + "=" * 72)
        print(model)
        print("=" * 72)

        for label, sys_extra, extra_body in ATTEMPTS:
            system = BASE_SYSTEM
            if sys_extra:
                system = f"{sys_extra}\n\n{BASE_SYSTEM}"

            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": QUESTION},
                ],
                "temperature": 0.4,
                "max_tokens": 400,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body

            print(f"\n  {label}")
            start = time.perf_counter()

            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                print(f"    FAILED {time.perf_counter() - start:5.2f}s  "
                      f"{type(e).__name__}: {str(e)[:120]}")
                continue

            secs = time.perf_counter() - start
            choice = resp.choices[0] if resp.choices else None

            if not choice:
                print(f"    {secs:5.2f}s  no choices returned")
                continue

            content = (choice.message.content or "").strip()
            reasoning = getattr(choice.message, "reasoning_content", None)
            rlen = len(reasoning) if reasoning else 0

            status = "THINKING OFF" if rlen == 0 else f"thinking {rlen} chars"
            print(f"    {secs:5.2f}s  {status}  finish={choice.finish_reason}")
            print(f"    reply: {content[:150]}")

            if rlen == 0 and content:
                winners.append((secs, model, label))

    print("\n" + "=" * 72)
    print("WORKING COMBINATIONS (no reasoning, real reply)")
    print("=" * 72)

    if not winners:
        print("\nNone. Reasoning could not be disabled on either model.")
        print("Fall back to: keep nemotron-3-super-120b-a12b (6.6s), raise")
        print("max_tokens to 1500 so the visible answer isn't starved, and")
        print("set API_TIMEOUT = 90.")
        return

    for secs, model, label in sorted(winners):
        print(f"\n  {secs:5.2f}s  {model}")
        print(f"          via: {label}")

    best = sorted(winners)[0]
    print("\n" + "=" * 72)
    print(f"Use: {best[1]}")
    print(f"With: {best[2]}")
    print("=" * 72)


if __name__ == "__main__":
    main()