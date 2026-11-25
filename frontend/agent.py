"""Stage 5 guarded agent loop, refactored as a generator of UI events.

Same model, system prompt, tools, and guardrails as 05_agent_loop/run.py, but
state is per-run and every step is yielded as an event instead of printed.
"""

from pathlib import Path
import json
import ollama

ROOT = Path(__file__).resolve().parents[1]
ORDERS = json.loads((ROOT / "shared_data/orders.json").read_text())
HISTORY = json.loads((ROOT / "shared_data/refund_history.json").read_text())
POLICY = (ROOT / "shared_data/refund_policy.md").read_text()

MODEL = "qwen3:4b-instruct"
MAX_STEPS = 8

SYSTEM = """
You are a refund operations agent.

Use tools for facts. Never invent order IDs, customer IDs, policy, history,
carrier status, or refund decisions.

Before proposing a refund:
1. verify the order with get_order
2. search the relevant refund policy
3. check refund history
4. for a missing-parcel claim, also check carrier status
5. call propose_refund for the controlled business decision

IMPORTANT:
- You do not have authority to approve or reject refunds in prose.
- A refund outcome exists only if propose_refund returns one.
- After propose_refund returns, explain that tool result briefly.
""".strip()


def list_orders():
    return ORDERS


def run_agent(order_id: str, reason: str, requested: float, photo_supplied: bool):
    """Yield event dicts: tool_call, guardrail, observation, final, stop, error."""
    state = {
        "trusted_order": None,
        "policy_searched": False,
        "history_checked": False,
        "proposal_result": None,
    }
    notes = []  # guardrail events raised inside tools, flushed after each call

    def get_order(order_id: str):
        """Get trusted order data for an order ID."""
        order = next((x for x in ORDERS if x["order_id"] == order_id), None)
        if order:
            state["trusted_order"] = order
            return order
        return {"error": "order_not_found"}

    def get_refund_history(customer_id: str):
        """Get refund history for a customer ID."""
        trusted = state["trusted_order"]
        if trusted:
            if customer_id != trusted["customer_id"]:
                notes.append({
                    "title": "Orchestrator guardrail",
                    "detail": f"Model supplied customer_id {customer_id}; rebound to trusted {trusted['customer_id']}.",
                })
            customer_id = trusted["customer_id"]
        result = HISTORY.get(customer_id, {"refunds_last_12m": 0})
        state["history_checked"] = True
        return {"customer_id": customer_id, **result}

    def search_policy(query: str):
        """Search refund policy for relevant rules."""
        chunks = [("## " + s).strip() for s in POLICY.split("## ")[1:] if s.strip()]
        q = query.lower()
        terms = {w.strip(".,:;!?()[]").lower() for w in query.split() if len(w) > 2}
        scored = []
        for chunk in chunks:
            text = chunk.lower()
            score = sum(1 for t in terms if t in text)
            if any(x in q for x in ("crack", "broken", "damage", "defect")) and "damaged item" in text:
                score += 5
            if any(x in q for x in ("missing", "lost", "not arrived")) and "missing parcel" in text:
                score += 5
            if any(x in q for x in ("wrong", "incorrect")) and "wrong item" in text:
                score += 5
            scored.append((score, chunk))
        state["policy_searched"] = True
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:2]
        return "\n\n".join(c for _, c in top)

    def check_carrier_scan(order_id: str):
        """Get the trusted carrier status for an order."""
        order = next((x for x in ORDERS if x["order_id"] == order_id), None)
        if not order:
            return {"error": "order_not_found"}
        missing_case = any(x in reason.lower() for x in ("missing", "lost", "not arrived", "never arrived"))
        if not missing_case:
            notes.append({
                "title": "Orchestration note",
                "detail": "Carrier lookup is not required for this damaged-item claim; allowed as a read-only call.",
            })
        return {"order_id": order_id, "carrier_status": order["carrier_status"]}

    def propose_refund(order_id: str, requested_amount: float, reason: str, photo_supplied: bool = False):
        """Run the controlled refund proposal through deterministic guardrails."""
        order = state["trusted_order"]
        if not order or order["order_id"] != order_id:
            result = {"decision": "HUMAN_REVIEW", "reason": "Order not verified"}
        elif not state["policy_searched"]:
            result = {"decision": "HUMAN_REVIEW", "reason": "Policy was not checked"}
        elif not state["history_checked"]:
            result = {"decision": "HUMAN_REVIEW", "reason": "Refund history was not checked"}
        else:
            paid = float(order["amount_paid"])
            amt = float(requested_amount)
            if amt > paid:
                result = {"decision": "BLOCKED", "reason": "Requested refund exceeds amount paid"}
            elif amt > 200:
                result = {"decision": "HUMAN_REVIEW", "reason": "Requested refund exceeds $200 automation cap"}
            elif amt > 75:
                result = {"decision": "HUMAN_REVIEW", "reason": "Requested refund exceeds $75 auto-approval threshold"}
            elif any(x in reason.lower() for x in ("crack", "broken", "damage", "defect")) and not photo_supplied:
                result = {"decision": "NEEDS_INFORMATION", "reason": "Photo evidence is required for damaged item"}
            else:
                refunds = HISTORY.get(order["customer_id"], {}).get("refunds_last_12m", 0)
                if refunds >= 3:
                    result = {"decision": "HUMAN_REVIEW", "reason": "Repeated refund history"}
                else:
                    result = {
                        "decision": "AUTO_APPROVE",
                        "reason": "All deterministic controls passed",
                        "approved_amount": amt,
                        "simulation_only": True,
                    }
        state["proposal_result"] = result
        return result

    tools = [get_order, get_refund_history, search_policy, check_carrier_scan, propose_refund]
    tool_map = {fn.__name__: fn for fn in tools}

    user_message = f"""
Customer requests a refund.

Order ID: {order_id}
Reason: {reason}
Requested amount: {requested}
Photo supplied: {photo_supplied}

Investigate the claim and use the controlled refund decision tool before giving
a final refund outcome.
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message},
    ]

    for step in range(1, MAX_STEPS + 1):
        try:
            response = ollama.chat(model=MODEL, messages=messages, tools=tools)
        except Exception as exc:
            yield {"type": "error", "detail": f"{type(exc).__name__}: {exc}"}
            return
        msg = response.message
        messages.append(msg)
        calls = msg.tool_calls or []

        if calls:
            for call in calls:
                name = call.function.name
                args = dict(call.function.arguments or {})
                yield {"type": "tool_call", "step": step, "name": name, "args": args}

                if name == "propose_refund":
                    rebound = {
                        "order_id": order_id,
                        "requested_amount": requested,
                        "reason": reason,
                        "photo_supplied": photo_supplied,
                    }
                    if rebound != args:
                        yield {
                            "type": "guardrail",
                            "title": "Action gate",
                            "detail": "Consequential-action arguments rebound to trusted session inputs.",
                            "model_args": args,
                            "safe_args": rebound,
                        }
                    args = rebound

                fn = tool_map.get(name)
                if not fn:
                    output = {"error": f"unknown_tool:{name}"}
                else:
                    try:
                        output = fn(**args)
                    except Exception as exc:
                        output = {"error": f"{type(exc).__name__}: {exc}"}

                for n in notes:
                    yield {"type": "guardrail", **n}
                notes.clear()

                yield {"type": "observation", "step": step, "name": name, "output": output}
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": output if isinstance(output, str) else json.dumps(output),
                })
            continue

        if state["proposal_result"] is None:
            yield {
                "type": "guardrail",
                "title": "Action gate",
                "detail": "Model tried to finish before calling propose_refund. Returning control to the agent.",
            }
            messages.append({
                "role": "user",
                "content": (
                    "ACTION GATE: You have not called propose_refund. "
                    "You may not give a refund outcome yet. "
                    "Call propose_refund using the verified facts."
                ),
            })
            continue

        yield {
            "type": "final",
            "answer": msg.content,
            "outcome": state["proposal_result"],
            "simulation_notice": bool(state["proposal_result"].get("simulation_only")),
        }
        return

    yield {
        "type": "stop",
        "detail": f"Agent reached the maximum of {MAX_STEPS} steps. Escalate to human review.",
    }
