"""Run the same scenarios on two chat models and compare outcome, tokens, time and estimated cost.

Both models are warmed up first (not counted) so cold-start load time doesn't skew the numbers.
Nothing is downloaded here: if the second model isn't installed we say how to install it.
"""

import time

import ollama

import agent
import evals

SCENARIOS = [c for c in evals.E2E_CASES if c[0].split(" ")[0] in ("refund", "chair", "T-shirt", "change")]


def installed_models():
    try:
        return sorted({m.model for m in ollama.list().models})
    except Exception:
        return []


_cards = None


def model_cards():
    """Model details straight from Ollama (cached)."""
    global _cards
    if _cards is not None:
        return _cards
    cards = []
    try:
        listed = {m.model: m for m in ollama.list().models}
    except Exception:
        listed = {}
    for name, role in ((agent.CHAT_MODEL, "chat · intent routing · tool calling"), (agent.EMBED_MODEL, "embeddings · RAG")):
        try:
            info = ollama.show(name)
            d = info.details
            ctx = next((v for k, v in (info.modelinfo or {}).items() if k.endswith("context_length")), None)
            meta = listed.get(name) or listed.get(name + ":latest")
            cards.append({
                "name": name, "role": role, "family": d.family, "parameters": d.parameter_size,
                "quantization": d.quantization_level, "context_length": ctx, "local": True,
                "digest": meta.digest[:12] if meta else None,
                "size_mb": round(meta.size / 1e6) if meta else None,
            })
        except Exception as exc:
            cards.append({"name": name, "role": role, "error": f"{type(exc).__name__}: {exc}"})
    _cards = cards
    return cards


def _warm(model):
    t0 = time.time()
    try:
        ollama.chat(model=model, messages=[{"role": "user", "content": "ok"}])
        return int((time.time() - t0) * 1000), None
    except Exception as exc:
        return int((time.time() - t0) * 1000), f"{type(exc).__name__}: {exc}"


def run(model_b):
    model_a = agent.CHAT_MODEL
    yield {"type": "compare_start", "models": [model_a, model_b], "scenarios": len(SCENARIOS) + len(evals.ROUTER_CASES)}
    have = installed_models()
    if model_b not in have and (model_b + ":latest") not in have:
        yield {"type": "compare_error",
               "message": f"{model_b} is not installed on this machine.",
               "fix": f"ollama pull {model_b}", "installed": have}
        return

    for model in (model_a, model_b):
        warm_ms, warm_err = _warm(model)
        if warm_err:
            yield {"type": "compare_error", "message": f"{model} failed to load: {warm_err}", "fix": None, "installed": have}
            continue
        passed = total = 0
        agg = {"prompt_tokens": 0, "output_tokens": 0, "llm_ms": 0, "wall_ms": 0, "llm_calls": 0,
               "cost": {p["id"]: 0.0 for p in agent.PRICES.get("profiles", [])}}

        def add(s):
            t = agent.usage_totals(s)
            for k in ("prompt_tokens", "output_tokens", "llm_ms", "llm_calls"):
                agg[k] += t[k]
            for k, v in t["cost"].items():
                agg["cost"][k] = round(agg["cost"][k] + v, 6)

        for case in SCENARIOS:
            name, turns, photo_after, expected = case[:4]
            getter = case[4] if len(case) > 4 else "decision"
            s = agent.Session("compare")
            s.model = model
            t0 = time.time()
            try:
                for t in turns:
                    s.turn_t0 = time.time()
                    for _ in agent.handle_message(s, t):
                        pass
                if photo_after:
                    for _ in agent.handle_photo(s):
                        pass
                if getter == "phase":
                    actual = s.phase
                elif getter == "resolution":
                    actual = s.resolution["type"] if s.resolution else "NONE"
                else:
                    actual = (s.proposal or {}).get("decision", "NO_DECISION")
            except Exception as exc:
                actual = f"{type(exc).__name__}: {str(exc)[:80]}"
            wall = int((time.time() - t0) * 1000)
            agg["wall_ms"] += wall
            add(s)
            total += 1
            passed += int(actual == expected)
            tt = agent.usage_totals(s)
            yield {"type": "compare_row", "model": model, "group": "Conversation outcome", "name": name,
                   "expected": expected, "actual": actual, "ok": actual == expected, "wall_ms": wall,
                   "tokens": tt["prompt_tokens"] + tt["output_tokens"], "llm_calls": tt["llm_calls"], "cost": tt["cost"]}

        for text, expected in evals.ROUTER_CASES:
            s = agent.Session("compare-router")
            s.model = model
            t0 = time.time()
            try:
                actual = agent.route_intent(s, text)
            except Exception as exc:
                actual = f"{type(exc).__name__}"
            wall = int((time.time() - t0) * 1000)
            agg["wall_ms"] += wall
            add(s)
            total += 1
            passed += int(actual == expected)
            tt = agent.usage_totals(s)
            yield {"type": "compare_row", "model": model, "group": "Intent router", "name": f"“{text}”",
                   "expected": expected, "actual": actual, "ok": actual == expected, "wall_ms": wall,
                   "tokens": tt["prompt_tokens"] + tt["output_tokens"], "llm_calls": tt["llm_calls"], "cost": tt["cost"]}

        yield {"type": "compare_model_done", "model": model, "passed": passed, "total": total, "warm_ms": warm_ms, **agg}
    yield {"type": "compare_done"}
