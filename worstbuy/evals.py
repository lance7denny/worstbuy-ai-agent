"""Eval suites for the Worstbuy agent, streamed to the Tests tab.

  fast : deterministic decision rules (no LLM) + router accuracy (LLM)
  full : fast + scripted end-to-end conversations (LLM, slower)

Deterministic cases test the hard-control layer on its own, like Stage 6.
"""

import importlib.util
import json
import time
from pathlib import Path

import agent

ROOT = Path(__file__).resolve().parents[1]


def _load_workshop_decision():
    """06_evals starts with a digit, so import it by path."""
    spec = importlib.util.spec_from_file_location("workshop_decision", ROOT / "06_evals/decision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decide_refund


def _session(order, claim, photo, refunds, policy=True, history=True, inventory=True):
    """A session with synthetic state, so rules can be tested in isolation."""
    s = agent.Session("eval")
    cid = "EVAL-" + str(abs(hash(json.dumps(order, sort_keys=True) + str(refunds))))
    agent.HISTORY[cid] = {"refunds_last_12m": refunds}
    s.order = {"order_id": "ORD-EVAL", "customer_id": cid, "carrier_status": "delivered_front_door",
               "status": "delivered", "days_since_delivery": 5, "item": "EvalItem-in", **order}
    s.order_id = "ORD-EVAL"
    s.claim_type, s.photo = claim, photo
    s.policy_searched, s.history_checked, s.inventory_checked = policy, history, inventory
    return s, cid


def _cleanup(cid):
    agent.HISTORY.pop(cid, None)


LOST = dict(status="in_transit", carrier_status="carrier_confirmed_lost", days_since_delivery=None)

# (name, order overrides, claim type, photo, refunds in 12m, expected summary, session kwargs)
RESOLUTION_CASES = [
    ("damaged + photo, in stock → refund or replacement", dict(amount_paid=49), "DAMAGED_ITEM", True, 0,
     "AUTO_APPROVE: REFUND_FULL $49, REPLACEMENT", {}),
    ("damaged without photo → ask for photo", dict(amount_paid=49), "DAMAGED_ITEM", False, 0, "NEEDS_INFORMATION", {}),
    ("wrong item + photo", dict(amount_paid=49), "WRONG_ITEM", True, 0, "AUTO_APPROVE: REFUND_FULL $49, REPLACEMENT", {}),
    ("wrong item without photo → ask for photo", dict(amount_paid=49), "WRONG_ITEM", False, 0, "NEEDS_INFORMATION", {}),
    ("reported on day 20 → 50% partial refund", dict(amount_paid=49, days_since_delivery=20), "DAMAGED_ITEM", True, 0,
     "AUTO_APPROVE: REFUND_PARTIAL $24.5, REPLACEMENT", {}),
    ("reported on day 14 → still full refund", dict(amount_paid=49, days_since_delivery=14), "DAMAGED_ITEM", True, 0,
     "AUTO_APPROVE: REFUND_FULL $49, REPLACEMENT", {}),
    ("outside 30-day window → denied", dict(amount_paid=49, days_since_delivery=45), "DAMAGED_ITEM", True, 0, "DENIED", {}),
    ("change of mind / does not fit → denied", dict(amount_paid=49), "CHANGE_OF_MIND", False, 0, "DENIED", {}),
    ("“does not fit” on an order still in transit → executive (the screenshot case)",
     dict(amount_paid=35, status="in_transit", days_since_delivery=None), "CHANGE_OF_MIND", False, 0, "HUMAN_REVIEW", {}),
    ("damage claimed on an in-transit order → executive",
     dict(amount_paid=35, status="in_transit", days_since_delivery=None), "DAMAGED_ITEM", True, 0, "HUMAN_REVIEW", {}),
    ("$129 order → executive", dict(amount_paid=129), "DAMAGED_ITEM", True, 0, "HUMAN_REVIEW", {}),
    ("$210 order → executive", dict(amount_paid=210), "DAMAGED_ITEM", True, 0, "HUMAN_REVIEW", {}),
    ("missing parcel but carrier says delivered → executive", dict(amount_paid=49), "MISSING_PARCEL", False, 0, "HUMAN_REVIEW", {}),
    ("missing parcel, carrier confirmed lost, in stock", dict(amount_paid=35, **LOST), "MISSING_PARCEL", False, 0,
     "AUTO_APPROVE: REFUND_FULL $35, REPLACEMENT", {}),
    ("missing parcel, lost, out of stock → refund only", dict(amount_paid=35, item="EvalItem-out", **LOST), "MISSING_PARCEL", False, 0,
     "AUTO_APPROVE: REFUND_FULL $35", {}),
    ("damaged, out of stock → refund only", dict(amount_paid=49, item="EvalItem-out"), "DAMAGED_ITEM", True, 0,
     "AUTO_APPROVE: REFUND_FULL $49", {}),
    ("3 refunds in 12 months → executive", dict(amount_paid=49), "DAMAGED_ITEM", True, 3, "HUMAN_REVIEW", {}),
    ("ambiguous claim (OTHER) → executive", dict(amount_paid=49), "OTHER", True, 0, "HUMAN_REVIEW", {}),
    ("sequencing: stock not checked", dict(amount_paid=49), "DAMAGED_ITEM", True, 0, "HUMAN_REVIEW", dict(inventory=False)),
    ("sequencing: policy not checked", dict(amount_paid=49), "DAMAGED_ITEM", True, 0, "HUMAN_REVIEW", dict(policy=False)),
]

CHOICE_CASES = [("“I'll take the refund”", "I'll take the refund", "CHOOSE_REFUND"),
                ("“refund please”", "refund please", "CHOOSE_REFUND"),
                ("“I'd like the replacement”", "I'd like the replacement", "CHOOSE_REPLACEMENT"),
                ("“send me a new one”", "send me a new one", "CHOOSE_REPLACEMENT"),
                ("“no thanks”", "no thanks", "DECLINE")]

ROUTER_CASES = [
    ("Hi there", "OTHER"),
    ("thanks so much!", "OTHER"),
    ("I want my money back for this order", "REFUND"),
    ("this arrived broken, refund please", "REFUND"),
    ("can you send me a new one?", "REPLACEMENT"),
    ("I'd like to exchange this for a working one", "REPLACEMENT"),
    ("let me speak to a manager", "SUPPORT"),
    ("where is my parcel right now?", "UNSUPPORTED"),
]

TONE_CASES = [
    ("“Fckkk this. I need my money.”", "Fckkk this. I need my money.", "abusive"),
    ("“Hi Scammer. I need my Moneyyy” (accusation, not swearing)", "Hi Scammer. I need my Moneyyy. Give me my money back.", "frustrated"),
    ("“Cant you understand me. Im talking in english only.”", "Cant you understand me. Im talking in english only.", "frustrated"),
    ("“THIS IS RIDICULOUS I WANT MY MONEY”", "THIS IS RIDICULOUS I WANT MY MONEY", "frustrated"),
    ("“I need a refund please”", "I need a refund please", "calm"),
    ("“the glass arrived damaged”", "the glass arrived damaged", "calm"),
]

NO_ID_CASES = [("“I don't have it”", "I don't have it", True), ("“where do I find it?”", "where do I find it?", True),
               ("“I forgot the number”", "I forgot the number", True), ("“ORD-48213”", "ORD-48213", False),
               ("“my jug is cracked”", "my jug is cracked", False)]

REFUSAL_CASES = [("“We can't assist with refunds or money returns.” (the screenshot bug)", "We can't assist with refunds or money returns.", True),
                 ("“I'm unable to help with that.”", "I'm unable to help with that.", True),
                 ("“Could you please provide your order ID?”", "Could you please provide your order ID?", False)]

E2E_CASES = [
    ("refund jug + photo → eligible", ["refund ORD-48213 the jug arrived cracked"], True, "AUTO_APPROVE"),
    ("chair $210 → executive", ["refund ORD-48215 the chair is broken"], False, "HUMAN_REVIEW"),
    ("T-shirt lost, replacement out of stock → refund only", ["I want a replacement for ORD-48216, my parcel never arrived"], False, "REFUND_FULL", "resolution"),
    ("“does not fit” on an in-transit order → executive (screenshot case)", ["I want a refund for ORD-48216, it does not fit me"], False, "HUMAN_REVIEW"),
    ("change of mind on a delivered jug → denied", ["I want a refund for ORD-48213, I changed my mind"], False, "DENIED"),
    ("no order ID, asked 3 times → executive", ["I need my money back", "just give me my money, I need it back", "I need my money"], False, "closed", "phase"),
    ("swearing twice → executive", ["fuck this, give me my money you scammer", "fuck you, scam"], False, "closed", "phase"),
]


def _result(group, name, expected, actual, t0, detail=""):
    return {"type": "eval", "group": group, "name": name, "expected": expected, "actual": actual,
            "status": "pass" if expected == actual else "fail",
            "ms": int((time.time() - t0) * 1000), "detail": detail}


def run(suite="fast"):
    t_all = time.time()
    counts = {"pass": 0, "fail": 0}

    def emit(r):
        counts[r["status"]] += 1
        return r

    yield {"type": "eval_start", "suite": suite}
    try:
        workshop = _load_workshop_decision()
    except Exception as exc:  # keep going without parity
        workshop = None
        yield {"type": "eval", "group": "Workshop golden cases", "name": "load 06_evals/decision.py", "status": "fail",
               "expected": "loads", "actual": f"{type(exc).__name__}: {exc}", "ms": 0, "detail": ""}

    # 1. Workshop golden cases: workshop function AND Worstbuy decide_refund must both match
    group = "Workshop Stage 5/6 golden cases (amount-based rules kept for parity)"
    cases = json.loads((ROOT / "06_evals/golden_cases.json").read_text())
    for c in cases:
        t0 = time.time()
        s, cid = _session({"amount_paid": c["amount_paid"]}, c["claim_type"], c["photo_supplied"], c["refunds_last_12m"])
        reason = {"DAMAGED_ITEM": "arrived damaged", "MISSING_PARCEL": "never arrived", "WRONG_ITEM": "wrong item"}.get(c["claim_type"], "other")
        mine = s.decide_refund("ORD-EVAL", c["requested"], reason, c["photo_supplied"])["decision"]
        _cleanup(cid)
        theirs = workshop(c["amount_paid"], c["requested"], c["claim_type"], c["photo_supplied"], c["refunds_last_12m"]) if workshop else None
        ok = mine == c["expected"] and theirs == c["expected"]
        yield emit({"type": "eval", "group": group, "name": c["name"], "expected": c["expected"],
                    "actual": f"Worstbuy={mine}, workshop={theirs}" if not ok else mine,
                    "status": "pass" if ok else "fail", "ms": int((time.time() - t0) * 1000),
                    "detail": "Both implementations must agree with the golden answer."})

    # 2. Entitlement rules (the new decision engine)
    group = "Entitlement rules: refund / partial / replacement / denied"
    agent.INVENTORY["EvalItem-in"] = {"in_stock": True, "quantity": 5}
    agent.INVENTORY["EvalItem-out"] = {"in_stock": False, "quantity": 0}
    try:
        for name, order, claim, photo, refunds, expected, kw in RESOLUTION_CASES:
            t0 = time.time()
            s, cid = _session(order, claim, photo, refunds, **kw)
            reason = {"DAMAGED_ITEM": "arrived damaged", "MISSING_PARCEL": "never arrived", "WRONG_ITEM": "wrong item",
                      "CHANGE_OF_MIND": "does not fit"}.get(claim, "other")
            r = s.decide_resolution("ORD-EVAL", reason, photo)
            _cleanup(cid)
            actual = r["decision"]
            if r.get("options"):
                actual += ": " + ", ".join(o["type"] + (f" ${o['amount']:g}" if "amount" in o else "") for o in r["options"])
            yield emit(_result(group, name, expected, actual, t0))
    finally:
        agent.INVENTORY.pop("EvalItem-in", None)
        agent.INVENTORY.pop("EvalItem-out", None)

    group = "Customer choice parsing (code, no LLM)"
    for name, text, expected in CHOICE_CASES:
        t0 = time.time()
        s = agent.Session("eval-choice")
        s.awaiting_choice = True
        s.options = [{"type": "REFUND_FULL", "amount": 49, "percent": 100}, {"type": "REPLACEMENT", "item": "x"}]
        yield emit(_result(group, name, expected, agent.route_intent(s, text), t0))

    # 3b. Tone, stuck handling and reply-guard rules (pure code)
    for group, cases, fn in (
        ("Tone detection (code)", TONE_CASES, lambda t: agent.detect_tone(t)),
        ("“I don't have the order ID” detection (code)", NO_ID_CASES, lambda t: bool(agent.NO_ID_RE.search(t))),
        ("Reply guard: unwanted refusals (code)", REFUSAL_CASES, lambda t: bool(agent.REFUSAL_RE.search(t))),
    ):
        for name, text, expected in cases:
            t0 = time.time()
            yield emit(_result(group, name, expected, fn(text), t0))

    # 3c. Usage accounting (tokens, time and cost add up)
    group = "Usage accounting"
    prof = agent.PRICES.get("profiles", [])
    t0 = time.time()
    s = agent.Session("eval-usage")
    s.turn, s.turn_wall = 1, {1: 2000, 2: 3000}
    mk = lambda kind, pt, ot, wall, turn=1: {"model": "m", "kind": kind, "prompt_tokens": pt, "output_tokens": ot, "wall_ms": wall,
                                          "id": 0, "turn": turn, "purpose": "x"}
    s.usage = [mk("llm", 100, 50, 1000), mk("embed", 10, 0, 200), mk("llm", 40, 10, 500, turn=2)]
    t1, t2, ts = agent.usage_totals(s, 1), agent.usage_totals(s, 2), agent.usage_totals(s)
    yield emit(_result(group, "turn totals add up (tokens)", "100 in/50 out/10 embed",
                       f"{t1['prompt_tokens']} in/{t1['output_tokens']} out/{t1['embed_tokens']} embed", t0))
    yield emit(_result(group, "session = sum of turns (tokens + LLM ms)", True,
                       ts["prompt_tokens"] == t1["prompt_tokens"] + t2["prompt_tokens"]
                       and ts["output_tokens"] == t1["output_tokens"] + t2["output_tokens"]
                       and ts["llm_ms"] == t1["llm_ms"] + t2["llm_ms"], t0))
    yield emit(_result(group, "time buckets add up (LLM + embed + other = wall)", True,
                       all(t["llm_ms"] + t["embed_ms"] + t["other_ms"] == t["wall_ms"] for t in (t1, t2, ts)), t0))
    ok_cost = True
    for p in prof:
        want = ((100 * p["input_per_mtok"] + 50 * p["output_per_mtok"]) + 10 * p["embed_per_mtok"]) / 1e6
        ok_cost &= abs(t1["cost"][p["id"]] - round(want, 6)) < 1e-9
    yield emit(_result(group, "cost = tokens × price table (every profile)", True, ok_cost, t0))
    yield emit(_result(group, "local actual cost is $0", 0, agent.PRICES.get("local", {}).get("input_per_mtok", 0)
                       + agent.PRICES.get("local", {}).get("output_per_mtok", 0), t0))

    t0 = time.time()
    s = agent.Session("eval-usage-live")
    s.turn, s.turn_t0 = 1, time.time()
    agent.route_intent(s, "I want a refund please")
    agent.semantic_search("Refund claim for Glass jug: cracked", s=s)
    s.turn_wall[1] = int((time.time() - s.turn_t0) * 1000)
    rows = s.usage
    t = agent.usage_totals(s, 1)
    yield emit(_result(group, "live: router (LLM) + RAG (embedding) both recorded", "1 LLM + 1 embedding",
                       f"{t['llm_calls']} LLM + {t['embed_calls']} embedding", t0))
    yield emit(_result(group, "live: every call has model and non-zero tokens", True,
                       bool(rows) and all(r["model"] and (r["prompt_tokens"] or 0) > 0 for r in rows)
                       and all((r["output_tokens"] or 0) > 0 for r in rows if r["kind"] == "llm"), t0))
    yield emit(_result(group, "live: LLM + embed time never exceeds wall time", True,
                       t["llm_ms"] + t["embed_ms"] <= t["wall_ms"] and t["other_ms"] >= 0, t0))

    # 4. Router accuracy (LLM)
    group = "Intent router (LLM)"
    for text, expected in ROUTER_CASES:
        t0 = time.time()
        s = agent.Session("eval-router")
        actual = agent.route_intent(s, text)
        yield emit(_result(group, f"“{text}”", expected, actual, t0))

    # 5. End-to-end (LLM, slow)
    if suite == "full":
        group = "End-to-end conversations (LLM, slow)"
        for case in E2E_CASES:
            name, turns, photo_after, expected = case[:4]
            getter = case[4] if len(case) > 4 else "decision"
            t0 = time.time()
            s = agent.Session("eval-e2e")
            try:
                for t in turns:
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
                actual = f"{type(exc).__name__}: {exc}"
            yield emit(_result(group, name, expected, actual, t0))

    yield {"type": "eval_done", "passed": counts["pass"], "failed": counts["fail"],
           "total": counts["pass"] + counts["fail"], "ms": int((time.time() - t_all) * 1000)}
