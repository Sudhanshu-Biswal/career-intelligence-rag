import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.config import get_settings

load_dotenv()
log = logging.getLogger(__name__)

settings = get_settings()

MINI = os.getenv("QUERY_MODEL",      "gpt-4o-mini")
FULL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

_client = OpenAI(api_key=settings.openai_api_key)

_token_log: list[dict] = []

PRICING = {
    "gpt-4o-mini": {"input": 0.00015,  "output": 0.0006},
    "gpt-4o":      {"input": 0.0025,   "output": 0.01},
}


def get_total_cost() -> float:
    return round(sum(r["cost_usd"] for r in _token_log), 6)


def _compute_cost(model: str, pt: int, ct: int) -> float:
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return round((pt / 1_000_000 * p["input"] * 1000) +
                 (ct / 1_000_000 * p["output"] * 1000), 6)


def call_llm(
    prompt: str,
    model: str = MINI,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    call_type: str = "unknown",
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or ""
            usage   = response.usage
            pt, ct  = usage.prompt_tokens, usage.completion_tokens
            cost    = _compute_cost(model, pt, ct)
            _token_log.append({
                "call_type": call_type, "model": model,
                "prompt_tokens": pt, "completion_tokens": ct,
                "total_tokens": pt + ct, "cost_usd": cost,
            })
            return content
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"[{call_type}] attempt {attempt+1} failed: {e} — retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"[{call_type}] failed after {max_retries} attempts")


def call_mini(prompt: str, temperature: float = 0.2,
              max_tokens: int = 1500, call_type: str = "unknown") -> str:
    return call_llm(prompt, MINI, temperature, max_tokens, call_type)


def call_full(prompt: str, temperature: float = 0.2,
              max_tokens: int = 1500, call_type: str = "unknown") -> str:
    return call_llm(prompt, FULL, temperature, max_tokens, call_type)


def print_token_summary():
    if not _token_log:
        print("No token usage recorded.")
        return
    model_totals: dict[str, dict] = {}
    for e in _token_log:
        m = e["model"]
        if m not in model_totals:
            model_totals[m] = {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0}
        model_totals[m]["prompt"]     += e["prompt_tokens"]
        model_totals[m]["completion"] += e["completion_tokens"]
        model_totals[m]["total"]      += e["total_tokens"]
        model_totals[m]["cost"]       += e["cost_usd"]
    print("\n" + "=" * 50)
    print("  TOKEN USAGE")
    print("=" * 50)
    for model, t in model_totals.items():
        print(f"  {model:<20} {t['total']:>8,} tokens  ${t['cost']:.6f}")
    print(f"  {'TOTAL':<20} "
          f"{sum(t['total'] for t in model_totals.values()):>8,} tokens  "
          f"${get_total_cost():.6f}")
    print("=" * 50 + "\n")