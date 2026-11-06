"""Worstbuy conversational refund agent.

Built from the workshop stages:
  * Stage 3  - semantic policy search (nomic-embed-text, cosine similarity)
  * Stage 4  - LLM claim classification + deterministic guardrails
  * Stage 5  - tool-calling loop supervised by an orchestrator (identity
               rebinding, action gate, stop rule, no outcome without the
               controlled propose_refund tool)

The LLM interprets language and phrases replies; code owns every decision.
Simulation only: no real refund is ever processed.
"""

from pathlib import Path
import json
import math
import queue
import re
import threading
import time

import ollama

ROOT = Path(__file__).resolve().parents[1]
ORDERS = json.loads((ROOT / "shared_data/orders.json").read_text())
HISTORY = json.loads((ROOT / "shared_data/refund_history.json").read_text())
INVENTORY = json.loads((ROOT / "shared_data/inventory.json").read_text())
POLICY = (ROOT / "shared_data/refund_policy.md").read_text()

CHAT_MODEL = "qwen3:4b-instruct"
EMBED_MODEL = "nomic-embed-text"
MAX_STEPS = 8
MAX_ORDER_ATTEMPTS = 3

REFUND_INTENT = re.compile(
    r"refund|money back|return|damag|broken|crack|defect|missing|lost|never arrived|"
    r"not arrived|didn.t arrive|wrong item|wrong product|incorrect",
    re.I,
)
REPLACE_RE = re.compile(r"replac|exchange|swap|send (me )?(another|a new)|resend", re.I)
SUPPORT_RE = re.compile(r"human|real person|agent|manager|executive|supervisor|speak to|talk to|complain", re.I)
REFUND_WORD_RE = re.compile(r"refund|money back|cash|credit|reimburs|my money", re.I)
YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|sure|ok|okay|please|refund( me| instead)?|go ahead|do it|that works|sounds good)\b", re.I)
NO_RE = re.compile(r"^\s*(no|nope|nah|not really|no thanks)\b", re.I)
SLOT_ONLY_RE = re.compile(r"^\s*(\$?\s*\d+(\.\d+)?\s*(dollars?|usd|bucks)?|ord-\d+)\s*[.!]?\s*$", re.I)
NO_ID_RE = re.compile(r"(don'?t|do not|dont|didn'?t)\s+(have|know|remember|see)|can'?t\s+(find|remember)|cannot\s+find|no idea|forgot|lost (it|the)|where (is|do i find)", re.I)
ABUSIVE_RE = re.compile(r"\bf+[u*]+c*k+\w*|\bfck+\w*|\bshit\w*|\bbitch\w*|\basshole\w*|\bidiot\w*|\bstupid\b|\bbastard\w*|\bcrap\b|\bdamn\b|\bpiss\w*|\bmoron\w*|\bwtf\b", re.I)
FRUSTRATED_RE = re.compile(r"\bscam(mer|mers)?\b|frustrat|ridiculous|useless|waste of (my )?time|not working|you (don'?t|do not) understand|can'?t you understand|can'?t you|are you (even )?(listening|serious)|i('?m| am) (angry|furious|fed up)|fed up|terrible|worst|unacceptable|again and again|already (told|said)|!!+", re.I)
REFUSAL_RE = re.compile(r"\b(can'?t|cannot|unable to|not able to|won'?t)\b[^.?!]{0,20}\b(assist|help|process|refund|return)", re.I)
FULL_REFUND_RE = re.compile(r"full|entire|all (of )?(my|the) money|everything|whole", re.I)
ORDER_ID_RE = re.compile(r"ORD-\d{3,}", re.I)
DAMAGE_WORDS = ("crack", "broken", "damage", "defect")

# Stage 5 system prompt, verbatim, plus a customer-facing addendum.
STAGE5_SYSTEM = """
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

RESOLUTION_SYSTEM = """
You are a customer resolution agent for refunds and replacements.

Use tools for facts. Never invent order IDs, customer IDs, policy, history,
carrier status, stock levels, amounts, or decisions.

Before proposing a resolution:
1. verify the order with get_order
2. search the relevant policy
3. check refund history
4. for a missing-parcel claim, also check carrier status
5. check stock with check_inventory
6. call propose_resolution for the controlled business decision

IMPORTANT:
- You do not have authority to approve, deny or price anything in prose.
- You never ask the customer how much they want. Entitlement is computed by the tool.
- A resolution exists only if propose_resolution returns one.
- After propose_resolution returns, explain that tool result briefly.
""".strip()

FLOW_CFG = {
    "refund": {"action": "propose_resolution", "system": RESOLUTION_SYSTEM, "noun": "refund"},
    "replacement": {"action": "propose_resolution", "system": RESOLUTION_SYSTEM, "noun": "replacement"},
}

CUSTOMER_ADDENDUM = """
You are speaking directly to a Worstbuy customer. Be warm, brief (2-3 sentences)
and plain-spoken. Never mention tools, functions, guardrails, thresholds or
internal systems. Do not promise anything beyond the tool result.
""".strip()

PERSONA = (
    "You are Worstbuy's friendly customer support assistant, handling refund "
    "and replacement requests only. Never ask the customer how much money they want. Reply in 1-3 short sentences. Never decide, promise or "
    "imply a refund outcome yourself. Never mention tools or internal systems."
)

# ---------------------------------------------------------------- tracing
# Every meaningful step pushes a structured event onto the session queue so the
# UI inspector can show exactly what the backend is doing, as it happens.
def _t(s, kind, title, detail="", data=None, src="", status="ok", ms=None):
    """One-shot trace point."""
    if s is None:
        return None
    s.trace_id += 1
    s.q.put({
        "type": "trace", "id": s.trace_id, "turn": s.turn, "kind": kind, "phase": "point",
        "title": title, "detail": detail, "data": data, "src": src, "status": status, "ms": ms,
    })
    return s.trace_id


def _t_start(s, kind, title, detail="", data=None, src=""):
    """Open a span (shows as 'running' in the UI until _t_end)."""
    if s is None:
        return None
    s.trace_id += 1
    s.q.put({
        "type": "trace", "id": s.trace_id, "turn": s.turn, "kind": kind, "phase": "start",
        "title": title, "detail": detail, "data": data, "src": src, "status": "running", "ms": None,
    })
    return (s.trace_id, time.time())


def _t_end(s, span, detail=None, data=None, status="ok"):
    if s is None or span is None:
        return
    sid, t0 = span
    s.q.put({
        "type": "trace", "id": sid, "turn": s.turn, "phase": "end",
        "detail": detail, "data": data, "status": status, "ms": int((time.time() - t0) * 1000),
    })


def _state(s):
    """Push a snapshot of orchestrator-owned state (the LLM never owns this)."""
    if s is None:
        return
    s.q.put({"type": "state", "state": {
        "phase": s.phase,
        "order_id": s.order_id,
        "customer_id": s.order["customer_id"] if s.order else None,
        "item": s.order["item"] if s.order else None,
        "amount_paid": s.order["amount_paid"] if s.order else None,
        "reason": s.reason,
        "claim_type": s.claim_type,
        "eligible": ", ".join(o["type"] for o in s.options) or None,
        "resolution": s.resolution["type"] if s.resolution else None,
        "photo_supplied": s.photo,
        "policy_searched": s.policy_searched,
        "history_checked": s.history_checked,
        "decision": s.proposal["decision"] if s.proposal else None,
        "order_lookup_failures": s.attempts,
    }})


def _load_prices():
    try:
        return json.loads((Path(__file__).parent / "prices.json").read_text())
    except Exception:
        return {"profiles": [], "local": {}, "currency": "USD", "_note": "prices.json missing"}


PRICES = _load_prices()


def _model(s):
    """The chat model for this session (the model comparison swaps it per session)."""
    return getattr(s, "model", None) or CHAT_MODEL


def _cost_for(row, profile):
    """Estimated USD for one ledger row under a price profile (tokens x price per 1M)."""
    if row["kind"] == "embed":
        return (row["prompt_tokens"] or 0) * profile.get("embed_per_mtok", 0) / 1e6
    return ((row["prompt_tokens"] or 0) * profile.get("input_per_mtok", 0)
            + (row["output_tokens"] or 0) * profile.get("output_per_mtok", 0)) / 1e6


def usage_totals(s, turn=None):
    """Totals for one turn, or the whole session when turn is None."""
    rows = [r for r in s.usage if turn is None or r["turn"] == turn]
    llm = [r for r in rows if r["kind"] == "llm"]
    emb = [r for r in rows if r["kind"] == "embed"]
    wall = sum(s.turn_wall.values()) if turn is None else s.turn_wall.get(turn, 0)
    llm_ms, emb_ms = sum(r["wall_ms"] for r in llm), sum(r["wall_ms"] for r in emb)
    return {
        "llm_calls": len(llm), "embed_calls": len(emb),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in llm),
        "output_tokens": sum(r["output_tokens"] or 0 for r in llm),
        "embed_tokens": sum(r["prompt_tokens"] or 0 for r in emb),
        "llm_ms": llm_ms, "embed_ms": emb_ms, "wall_ms": wall,
        "other_ms": max(0, wall - llm_ms - emb_ms),  # tools, rules, bookkeeping
        "cost": {p["id"]: round(sum(_cost_for(r, p) for r in rows), 6) for p in PRICES.get("profiles", [])},
        "models": sorted({r["model"] for r in rows}),
    }


def usage_event(s):
    return {"type": "usage", "turn": s.turn, "turn_totals": usage_totals(s, s.turn), "session_totals": usage_totals(s)}


def _record(s, purpose, kind, model, resp, t0):
    """Account for one model call. Returns the per-call usage dict that is also shown in the trace."""
    def ms(x):
        return int((x or 0) / 1e6)
    get = (lambda k: resp.get(k) if isinstance(resp, dict) else getattr(resp, k, None))
    ptok, otok = get("prompt_eval_count"), get("eval_count")
    gen_ms, wall_ms = ms(get("eval_duration")), int((time.time() - t0) * 1000)
    u = {
        "model": model, "kind": kind,
        "prompt_tokens": ptok, "output_tokens": otok if kind == "llm" else 0,
        "wall_ms": wall_ms, "model_ms": ms(get("total_duration")), "load_ms": ms(get("load_duration")),
        "prompt_ms": ms(get("prompt_eval_duration")), "gen_ms": gen_ms,
        "tokens_per_s": round(otok / (gen_ms / 1000), 1) if kind == "llm" and otok and gen_ms else None,
    }
    if s is not None:
        row = {**u, "id": len(s.usage) + 1, "turn": s.turn, "purpose": purpose}
        s.usage.append(row)
        s.turn_wall[s.turn] = int((time.time() - s.turn_t0) * 1000)
        s.q.put(usage_event(s))
    return u


# ---------------------------------------------------------------- RAG (stage 3)
_policy_lock = threading.Lock()
_policy_index = None


def split_policy(text):
    return [("## " + s).strip() for s in text.split("## ")[1:] if s.strip()]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def policy_index():
    """Embed policy sections once with nomic-embed-text and cache them."""
    global _policy_index
    with _policy_lock:
        if _policy_index is None:
            chunks = split_policy(POLICY)
            vectors = ollama.embed(model=EMBED_MODEL, input=chunks)["embeddings"]
            _policy_index = list(zip(chunks, vectors))
        return _policy_index


def semantic_search(query, top_k=2, s=None):
    index = policy_index()
    sp = _t_start(s, "RAG", "Semantic policy search",
                  f'Embedding the query with {EMBED_MODEL}, then ranking policy sections by cosine similarity.',
                  {"query": query, "embedding_model": EMBED_MODEL, "policy_sections": len(index)},
                  "agent.py · semantic_search()  (Stage 3)")
    t0 = time.time()
    resp = ollama.embed(model=EMBED_MODEL, input=query)
    u = _record(s, "policy embedding", "embed", EMBED_MODEL, resp, t0)
    qv = resp["embeddings"][0]
    scored = sorted(((cosine_similarity(qv, v), c) for c, v in index), reverse=True)
    top = scored[:top_k]
    _t_end(s, sp, f"Retrieved top {top_k} of {len(index)} policy sections",
           {
               **u,
               "vector_dimensions": len(qv),
               "ranking": [
                   {"section": c.split("\n", 1)[0].lstrip("# ").strip(), "similarity": round(sc, 3),
                    "used": i < top_k}
                   for i, (sc, c) in enumerate(scored)
               ],
               "retrieved_text": [c for _, c in top],
           })
    return top


# ---------------------------------------------------------------- LLM helpers
def chat_text(messages, s=None, purpose="Generate reply", src=""):
    sp = _t_start(s, "LLM", purpose, f"{_model(s)} via Ollama (local)",
                  {"model": _model(s), "messages_sent": [
                      {"role": m["role"], "content": m["content"][:600]} for m in messages]}, src)
    t0 = time.time()
    resp = ollama.chat(model=_model(s), messages=messages)
    u = _record(s, purpose, "llm", _model(s), resp, t0)
    text = resp.message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    _t_end(s, sp, "Model responded", {**u, "output": text})
    return text


CLAIM_TYPES = ["DAMAGED_ITEM", "MISSING_PARCEL", "WRONG_ITEM", "CHANGE_OF_MIND", "OTHER"]


def classify_claim(reason, s=None):
    """Stage 4: the LLM interprets language only; it does not decide."""
    prompt = f"""
Classify this refund reason into exactly one category:

DAMAGED_ITEM   - arrived broken, cracked, defective or not working
MISSING_PARCEL - never arrived, lost, not delivered
WRONG_ITEM     - received a different product than ordered
CHANGE_OF_MIND - does not fit, wrong size, don't want it, no longer needed, changed mind
OTHER          - anything else or unclear

Refund reason: {reason}

Return only the category name. No explanation.
""".strip()
    out = chat_text([{"role": "user", "content": prompt}], s,
                    "LLM · classify the claim type", "agent.py · classify_claim()  (Stage 4)").upper()
    ok = out in CLAIM_TYPES
    _t(s, "CODE", "Validate classification against allow-list",
       f"Model said “{out[:40]}” → " + ("accepted" if ok else "not in allow-list, forced to OTHER"),
       {"allowed": CLAIM_TYPES, "model_output": out[:80]},
       "agent.py · classify_claim()", "ok" if ok else "warn")
    return out if ok else "OTHER"


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["order_id", "reason"],
}


def extract_slots(text, known, s=None):
    """LLM reads the customer's words; code validates what it returns."""
    prompt = f"""
Extract details from the customer's LATEST message only.

Already known: {json.dumps(known)}

Latest message: "{text}"

Fields:
- order_id: an order ID such as ORD-12345 if written in this message, else null
- reason: what is wrong with the order, in the customer's words, ONLY if stated
  in this message, else null

Return JSON only.
""".strip()
    sp = _t_start(s, "LLM", "LLM · extract details (structured output)",
                  f"{_model(s)} is forced to return JSON matching a schema. It only reads language; it decides nothing.",
                  {"model": _model(s), "prompt": prompt, "json_schema": EXTRACT_SCHEMA},
                  "agent.py · extract_slots()  (Stage 4)")
    try:
        t0 = time.time()
        resp = ollama.chat(
            model=_model(s),
            messages=[{"role": "user", "content": prompt}],
            format=EXTRACT_SCHEMA,
        )
        u = _record(s, "extract details", "llm", _model(s), resp, t0)
        data = json.loads(resp.message.content)
        _t_end(s, sp, "Model returned JSON", {**u, "raw_json": data})
    except Exception as exc:
        data = {}
        _t_end(s, sp, f"Extraction failed: {type(exc).__name__}", {"error": str(exc)}, "error")
    out = {"order_id": None, "reason": None}
    checks = []

    # Code-side validation: never accept values that are not in the customer's text.
    m = ORDER_ID_RE.search(text)
    if m:
        out["order_id"] = m.group(0).upper()
    checks.append({"field": "order_id", "model_said": data.get("order_id"),
                   "regex_found_in_text": out["order_id"], "accepted": out["order_id"]})
    reason = data.get("reason")
    if isinstance(reason, str) and len(reason.strip()) >= 3:
        out["reason"] = reason.strip()[:300]
    checks.append({"field": "reason", "model_said": reason, "accepted": out["reason"]})
    _t(s, "CODE", "Validate LLM output with code (never trust the model blindly)",
       "Each extracted value is cross-checked against the customer's actual words.",
       {"customer_text": text, "checks": checks}, "agent.py · extract_slots()  (Stage 4/5 principle)")
    return out


# ---------------------------------------------------------------- session
class _LogQueue(queue.Queue):
    """Event queue that also appends every event to the session's audit log."""

    def __init__(self, session):
        super().__init__()
        self._session = session

    def put(self, item, *a, **kw):
        if item is not None:
            self._session.log.append({"t": round(time.time(), 3), **item})
        super().put(item, *a, **kw)


class _Ladder:
    """Evaluates deterministic rules top to bottom and records every one for the UI."""

    def __init__(self, s, title, inputs, src):
        self.s, self.rows = s, []
        self.sp = _t_start(
            s, "DECISION", title,
            "Plain Python rules decide the outcome, top to bottom. The first rule that fires wins. The LLM has no vote.",
            {"inputs": inputs}, src)

    def rule(self, name, fired, why):
        self.rows.append({"rule": name, "fired": bool(fired), "detail": why})
        return bool(fired)

    def done(self, result):
        self.s.proposal = result
        _t_end(self.s, self.sp, f"Outcome: {result['decision']} — {result['reason']}",
               {"rules_evaluated": self.rows, "result": result},
               "ok" if result["decision"] in ("AUTO_APPROVE", "NEEDS_INFORMATION") else "warn")
        return result


class Session:
    def __init__(self, sid):
        self.id = sid
        self.lock = threading.Lock()
        self.phase = "menu"  # menu -> flow -> closed
        self.flow = None  # "refund" | "replacement" | None
        self.history = []  # (role, text) customer-visible transcript
        self.order = None
        self.order_id = None
        self.reason = None
        self.claim_type = None
        self.amount = None
        self.photo = False
        self.policy_searched = False
        self.history_checked = False
        self.inventory_checked = False
        self.proposal = None
        self.attempts = 0
        self.awaiting_photo = False
        self.offer_refund = False  # replacement unavailable; refund was offered
        self.options = []  # eligible resolutions computed by code
        self.awaiting_choice = False
        self.resolution = None  # what the customer chose (validated against options)
        self.tone = "calm"  # calm | frustrated | abusive (set by the router each turn)
        self.frustration = 0
        self.abuse_strikes = 0
        self.stuck = 0  # same question asked repeatedly without progress
        self.last_ask = None
        self.pending = False  # inputs changed since last decision
        self.model = CHAT_MODEL  # chat model for this session (comparison runs override it)
        self.usage = []  # ledger: one row per model call
        self.turn_wall = {}  # turn -> wall-clock ms so far
        self.turn_t0 = time.time()
        self.log = []  # full audit trail of every event
        self.created = time.time()
        self.q = _LogQueue(self)  # live events for the UI (chat + trace + state)
        self.trace_id = 0
        self.turn = 0

    def new_queue(self):
        self.q = _LogQueue(self)
        return self.q

    # -- tools (bound to this session; same names/contracts as stage 5) --------
    def tools(self, flow="refund"):
        s = self

        def get_order(order_id: str):
            """Get trusted order data for an order ID."""
            order = next((x for x in ORDERS if x["order_id"] == order_id), None)
            _t(s, "TOOL", "Tool · get_order  → orders.json",
               "System-of-record lookup. The result is trusted; model text is not.",
               {"input": {"order_id": order_id}, "result": order or {"error": "order_not_found"}},
               "agent.py · tools()  (Stage 2/5)", "ok" if order else "warn")
            return order or {"error": "order_not_found"}

        def get_refund_history(customer_id: str):
            """Get refund history for a customer ID."""
            if s.order and customer_id != s.order["customer_id"]:  # never trust a model-supplied identifier
                _t(s, "GUARDRAIL", "Identity guardrail · customer ID rebound",
                   "The model supplied a customer ID that does not match the verified order. Replaced with the trusted one.",
                   {"model_supplied": customer_id, "trusted_from_order": s.order["customer_id"]},
                   "agent.py · get_refund_history()  (Stage 5)", "warn")
            if s.order:
                customer_id = s.order["customer_id"]
            s.history_checked = True
            result = {"customer_id": customer_id, **HISTORY.get(customer_id, {"refunds_last_12m": 0})}
            _t(s, "TOOL", "Tool · get_refund_history  → refund_history.json",
               "Refunds in the last 12 months for the verified customer.",
               {"input": {"customer_id": customer_id}, "result": result},
               "agent.py · tools()  (Stage 2/5)")
            return result

        def search_policy(query: str):
            """Search refund policy for relevant rules."""
            item = s.order["item"] if s.order else "item"
            noun = "Replacement" if flow == "replacement" else "Refund"  # only steers the retrieval query
            _t(s, "TOOL", "Tool · search_policy  → RAG",
               "The model asked for policy. Query is enriched with the trusted item name, as in Stage 3.",
               {"model_query": query, "enriched_query": f"{noun} claim for {item}: {query}"},
               "agent.py · tools()  (Stage 3/5)")
            results = semantic_search(f"{noun} claim for {item}: {query}", s=s)
            s.policy_searched = True
            return "\n\n".join(chunk for _, chunk in results)

        def check_carrier_scan(order_id: str):
            """Get the trusted carrier status for an order."""
            order = next((x for x in ORDERS if x["order_id"] == order_id), None)
            if not order:
                return {"error": "order_not_found"}
            _t(s, "TOOL", "Tool · check_carrier_scan  → carrier data",
               "Trusted carrier status for the order.",
               {"input": {"order_id": order_id}, "result": {"carrier_status": order["carrier_status"]}},
               "agent.py · tools()  (Stage 2/5)")
            return {"order_id": order_id, "carrier_status": order["carrier_status"]}

        def check_inventory(item: str):
            """Check whether a replacement for an item is in stock."""
            if s.order:  # stock is looked up for the verified order's item, not model text
                item = s.order["item"]
            stock = INVENTORY.get(item, {"in_stock": False, "quantity": 0})
            s.inventory_checked = True
            _t(s, "TOOL", "Tool · check_inventory  → inventory.json",
               "Stock lookup for the verified order's item.",
               {"input": {"item": item}, "result": stock}, "agent.py · tools()  (new in phase 2)")
            return {"item": item, **stock}

        def propose_resolution(order_id: str, reason: str, photo_supplied: bool = False):
            """Work out what the customer is entitled to (full refund, partial refund, replacement, or none) through deterministic guardrails."""
            return s.decide_resolution(order_id, reason, photo_supplied)

        return [get_order, get_refund_history, search_policy, check_carrier_scan, check_inventory, propose_resolution]

    def decide(self, *a, **kw):  # backwards-compatible name
        return self.decide_refund(*a, **kw)

    def decide_refund(self, order_id, requested_amount, reason, photo_supplied):
        """Deterministic guardrails (stage 4 + stage 5) - the only source of outcomes."""
        order = self.order
        L = _Ladder(self, "Deterministic decision · propose_refund",
                    {"order_id": order_id, "requested_amount": requested_amount,
                     "reason": reason, "photo_supplied": photo_supplied},
                    "agent.py · Session.decide_refund()  (Stage 4 + 5, same rules as 06_evals/decision.py)")
        rule = L.rule
        if rule("Order verified?", not order or order["order_id"] != order_id, "order must match the trusted record"):
            result = {"decision": "HUMAN_REVIEW", "reason": "Order not verified"}
        elif rule("Policy checked first?", not self.policy_searched, "sequencing control"):
            result = {"decision": "HUMAN_REVIEW", "reason": "Policy was not checked"}
        elif rule("History checked first?", not self.history_checked, "sequencing control"):
            result = {"decision": "HUMAN_REVIEW", "reason": "Refund history was not checked"}
        else:
            paid = float(order["amount_paid"])
            amt = float(requested_amount)
            damaged = self.claim_type == "DAMAGED_ITEM" or any(w in reason.lower() for w in DAMAGE_WORDS)
            refunds = HISTORY.get(order["customer_id"], {}).get("refunds_last_12m", 0)
            if rule("Amount > 0?", amt <= 0, f"requested ${amt:g}"):
                result = {"decision": "BLOCKED", "reason": "Requested refund must be greater than zero"}
            elif rule("Requested > amount paid?", amt > paid, f"requested ${amt:g} vs paid ${paid:g}"):
                result = {"decision": "BLOCKED", "reason": "Requested refund exceeds amount paid"}
            elif rule("Requested > $200 hard cap?", amt > 200, f"requested ${amt:g}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Requested refund exceeds $200 automation cap"}
            elif rule("Requested > $75 auto-approval limit?", amt > 75, f"requested ${amt:g}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Requested refund exceeds $75 auto-approval threshold"}
            elif rule("Claim type ambiguous (OTHER)?", self.claim_type == "OTHER", f"claim type {self.claim_type}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Claim reason is ambiguous"}
            elif rule("Parcel 'missing' but carrier says delivered?",
                      self.claim_type == "MISSING_PARCEL" and "delivered" in order["carrier_status"],
                      f"carrier: {order['carrier_status']}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Carrier shows delivered but customer disputes"}
            elif rule("Damaged item without photo?", damaged and not photo_supplied,
                      f"damaged={damaged}, photo={photo_supplied}"):
                result = {"decision": "NEEDS_INFORMATION", "reason": "Photo evidence is required for damaged item"}
            elif rule("3+ refunds in last 12 months?", refunds >= 3, f"{refunds} refunds"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Repeated refund history"}
            else:
                result = {
                    "decision": "AUTO_APPROVE",
                    "reason": "All deterministic controls passed",
                    "approved_amount": amt,
                    "simulation_only": True,
                }
        return L.done(result)

    def decide_resolution(self, order_id, reason, photo_supplied):
        """What is this customer entitled to? Deterministic rules decide; nobody is asked for an amount.

        Outcomes: AUTO_APPROVE (with the list of eligible options), NEEDS_INFORMATION,
        DENIED (policy says no), HUMAN_REVIEW (ambiguous / contradictory / high value).
        """
        order = self.order
        L = _Ladder(self, "Deterministic decision · propose_resolution",
                    {"order_id": order_id, "reason": reason, "claim_type": self.claim_type, "photo_supplied": photo_supplied},
                    "agent.py · Session.decide_resolution()  (entitlement rules)")
        rule = L.rule
        if rule("Order verified?", not order or order["order_id"] != order_id, "order must match the trusted record"):
            result = {"decision": "HUMAN_REVIEW", "reason": "Order not verified"}
        elif rule("Policy checked first?", not self.policy_searched, "sequencing control"):
            result = {"decision": "HUMAN_REVIEW", "reason": "Policy was not checked"}
        elif rule("History checked first?", not self.history_checked, "sequencing control"):
            result = {"decision": "HUMAN_REVIEW", "reason": "Refund history was not checked"}
        elif rule("Stock checked first?", not self.inventory_checked, "sequencing control"):
            result = {"decision": "HUMAN_REVIEW", "reason": "Inventory was not checked"}
        else:
            value = float(order["amount_paid"])
            days = order.get("days_since_delivery")
            delivered = order.get("status") == "delivered"
            claim = self.claim_type
            refunds = HISTORY.get(order["customer_id"], {}).get("refunds_last_12m", 0)
            stock = INVENTORY.get(order["item"], {"in_stock": False})
            if rule("Claim contradicts the order record?",
                    claim in ("DAMAGED_ITEM", "WRONG_ITEM", "CHANGE_OF_MIND") and not delivered,
                    f"claim {claim} but order status is “{order.get('status')}”"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Claim contradicts the order record (item not delivered yet)"}
            elif rule("Parcel 'missing' but carrier says delivered?",
                      claim == "MISSING_PARCEL" and "delivered" in order["carrier_status"],
                      f"carrier: {order['carrier_status']}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Carrier shows delivered but customer disputes"}
            elif rule("Outside 30-day window?", days is not None and days > 30, f"days since delivery: {days}"):
                result = {"decision": "DENIED", "reason": "The 30-day window has passed"}
            elif rule("Change of mind / does not fit?", claim == "CHANGE_OF_MIND", f"claim type {claim}"):
                result = {"decision": "DENIED", "reason": "Change of mind and fit issues aren't covered by the policy"}
            elif rule("Claim type ambiguous (OTHER)?", claim == "OTHER", f"claim type {claim}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Claim reason is ambiguous"}
            elif rule("Order value > $200 hard cap?", value > 200, f"value ${value:g}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Order value exceeds $200 automation cap"}
            elif rule("Order value > $75 auto-approval limit?", value > 75, f"value ${value:g}"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Order value exceeds $75 auto-approval threshold"}
            elif rule("Photo proof missing (damaged / wrong item)?",
                      claim in ("DAMAGED_ITEM", "WRONG_ITEM") and not photo_supplied,
                      f"claim {claim}, photo={photo_supplied}"):
                result = {"decision": "NEEDS_INFORMATION", "reason": "Photo evidence is required for damaged or wrong items"}
            elif rule("3+ refunds in last 12 months?", refunds >= 3, f"{refunds} refunds"):
                result = {"decision": "HUMAN_REVIEW", "reason": "Repeated refund history"}
            else:
                # Eligible. Work out exactly what they are entitled to.
                late = days is not None and days > 14
                pct = 50 if late else 100
                options = [{
                    "type": "REFUND_PARTIAL" if late else "REFUND_FULL",
                    "percent": pct, "amount": round(value * pct / 100, 2),
                    "why": (f"Because this was reported {days} days after delivery, the refund is 50% (our 15-30 day rate)." if late else
                            "That's the full amount because it's within our full-refund period."),
                }]
                rule("Late report (15-30 days) → partial refund?", late, f"days since delivery: {days}")
                can_replace = claim in ("DAMAGED_ITEM", "WRONG_ITEM", "MISSING_PARCEL") and stock.get("in_stock")
                rule("Replacement available? (eligible claim + in stock)", can_replace,
                     f"claim {claim}, {order['item']}: {stock}")
                if can_replace:
                    options.append({"type": "REPLACEMENT", "item": order["item"],
                                    "why": "eligible claim and the item is in stock"})
                result = {"decision": "AUTO_APPROVE", "reason": "Eligible: " + " or ".join(o["type"] for o in options),
                          "options": options, "simulation_only": True}
        return L.done(result)


# ---------------------------------------------------------------- orchestration
def _say(s, text, how="LLM-written"):
    s.history.append(("assistant", text))
    _t(s, "OUTPUT", "Reply sent to customer", f"Wording source: {how}", {"message": text},
       "agent.py · _say()")
    return {"type": "assistant", "text": text}


DRIFT_WORDS = ("couldn't find", "could not find", "can't find", "unable to find", "not found", "double-check", "photo", "upload", "receipt", "document", "screenshot", "confirmation", "business day", "within")


def _reply(s, instruction, fallback=None):
    """Plain conversational reply (no tools) steered by the orchestrator."""
    facts = {
        "order_verified": bool(s.order),
        "item": s.order["item"] if s.order else None,
        "reason_given": s.reason,
        "claim_type": s.claim_type,
        "photo_uploaded": s.photo,
    }
    system = f"{PERSONA}\n\nKnown facts: {json.dumps(facts)}\n\nINSTRUCTION: {instruction}"
    msgs = [{"role": "system", "content": system}]
    # Only the customer's recent words: replaying earlier bot turns makes the
    # small model imitate them (e.g. repeat "couldn't find order") instead of
    # following the current instruction.
    for role, text in [h for h in s.history if h[0] == "user"][-2:]:
        msgs.append({"role": role, "content": text})
    out = chat_text(msgs, s, "LLM · write the customer reply",
                    "agent.py · _reply()  (orchestrator instructs, model only phrases)")
    # Guard: the model may only ask for / promise what the orchestrator said.
    hits = [w for w in DRIFT_WORDS if w in out.lower() and w not in instruction.lower()]
    if REFUSAL_RE.search(out) and not REFUSAL_RE.search(instruction):
        hits.append("refusal the orchestrator did not ask for")
    if not out or hits:
        _t(s, "GUARDRAIL", "Reply guard · model drifted, using safe wording",
           "The model mentioned something the orchestrator did not ask for, so the reply is replaced.",
           {"model_reply": out, "off_script_words": hits, "fallback_used": fallback or instruction},
           "agent.py · _reply()", "warn")
        return fallback or instruction
    _t(s, "GUARDRAIL", "Reply guard · wording accepted",
       "No off-script claims detected in the model's reply.", {"checked_for": list(DRIFT_WORDS)},
       "agent.py · _reply()")
    return out


# ---------------------------------------------------------------- support handoff
def case_summary(s, reason):
    """What a human executive would see. Built by code from verified state, not by the LLM."""
    return {
        "escalation_reason": reason,
        "flow": s.flow,
        "order_id": s.order_id,
        "customer_id": s.order["customer_id"] if s.order else None,
        "item": s.order["item"] if s.order else None,
        "amount_paid": s.order["amount_paid"] if s.order else None,
        "claim_type": s.claim_type,
        "customer_said": s.reason,
        "eligible_options": s.options,
        "photo_supplied": s.photo,
        "checks_run": {
            "order_verified": bool(s.order),
            "policy_searched": s.policy_searched,
            "history_checked": s.history_checked,
            "inventory_checked": s.inventory_checked,
        },
        "system_decision": s.proposal,
        "last_messages": [{"role": r, "text": t[:200]} for r, t in s.history[-6:]],
    }


def _escalate(s, reason):
    s.phase = "closed"
    summary = case_summary(s, reason)
    _t(s, "DECISION", "Escalate to human executive",
       "Any HUMAN_REVIEW, support request, stop rule or failure ends the automated flow. A case summary is built from verified state.",
       {"reason": reason, "case_summary": summary},
       "agent.py · _escalate()  (Stage 5 stop rule)", "warn")
    _state(s)
    text = (
        "Thanks for your patience. " + reason + " I'm connecting you to a Worstbuy executive "
        "who can take it from here. They'll have your order details so you won't need to repeat yourself."
    )
    yield _say(s, text, "fixed template (no LLM)")
    yield {"type": "ui", "action": "connect_executive", "summary": summary}


FRIENDLY = {
    "exceeds $200": "A request this size needs a specialist to review it personally.",
    "exceeds $75": "Requests at this amount need a specialist to review them.",
    "Repeated refund": "Your account needs a quick specialist review before we can go further.",
    "ambiguous": "I want to make sure we get this exactly right, so a specialist should take a look.",
    "disputes": "The carrier shows this parcel as delivered, so a specialist needs to look into it.",
    "30-day": "This purchase is outside our 30-day window, so a specialist needs to review it.",
    "Wrong-item": "Wrong-item claims at this value need a specialist's review.",
    "contradicts": "That doesn't match what our records show for this order (it hasn't been delivered yet), so a specialist needs to look into it.",
}


def _friendly(reason):
    for key, msg in FRIENDLY.items():
        if key in reason:
            return msg
    return "This one needs a specialist's review."


# ---------------------------------------------------------------- guarded agent loop
def _task_for(s):
    pref = s.flow if s.flow in ("refund", "replacement") else "refund"
    return f"""
Customer is asking about a {pref} for an order.

Order ID: {s.order_id}
What the customer said: {s.reason}
Photo supplied: {s.photo}

Investigate the claim and use the controlled propose_resolution tool before
giving any outcome. Do not ask the customer for an amount.
""".strip()


def _run_decision(s):
    """Stage 5 guarded tool loop for one decision attempt (refund or replacement)."""
    cfg = FLOW_CFG[s.flow]
    action = cfg["action"]
    s.stuck, s.last_ask = 0, None  # progress was made
    s.proposal = None
    tools = s.tools(s.flow)
    tool_map = {fn.__name__: fn for fn in tools}
    context = _task_for(s)
    messages = [
        {"role": "system", "content": cfg["system"] + "\n\n" + CUSTOMER_ADDENDUM},
        {"role": "user", "content": context},
    ]
    labels = {
        "get_order": "Verifying your order…",
        "search_policy": "Reviewing our policy…",
        "get_refund_history": "Checking your account history…",
        "check_carrier_scan": "Checking the carrier's tracking…",
        "check_inventory": "Checking stock…",
        "propose_resolution": "Working out what you're entitled to…",
    }
    _t(s, "ROUTER", f"Enter guarded agent loop · {s.flow}",
       f"All inputs are collected. The model may now choose tools, up to {MAX_STEPS} steps (stop rule).",
       {"flow": s.flow, "system_prompt": cfg["system"], "task_given_to_model": context,
        "tools_offered": [fn.__name__ for fn in tools], "max_steps": MAX_STEPS},
       "agent.py · _run_decision()  (Stage 5)")

    for step in range(1, MAX_STEPS + 1):
        sp = _t_start(s, "LLM", f"LLM · agent step {step}: choose next tool",
                      f"{_model(s)} sees the task and the tool results so far, and picks tools (native tool calling).",
                      {"model": _model(s), "messages_in_context": len(messages)},
                      "agent.py · _run_decision()  (Stage 5)")
        try:
            t0 = time.time()
            resp = ollama.chat(model=_model(s), messages=messages, tools=tools)
            u = _record(s, f"agent step {step}", "llm", _model(s), resp, t0)
            msg = resp.message
        except Exception as exc:
            _t_end(s, sp, f"LLM call failed: {type(exc).__name__}", {"error": str(exc)}, "error")
            yield from _escalate(s, "I ran into a technical problem checking this.")
            return
        messages.append(msg)
        calls = msg.tool_calls or []
        _t_end(s, sp,
               ("Model requested: " + ", ".join(c.function.name for c in calls)) if calls
               else "Model produced a final answer (no tool call)",
               {**u,
                "tool_calls": [{"name": c.function.name, "arguments": dict(c.function.arguments or {})} for c in calls],
                "text": (msg.content or "")[:500]})

        if calls:
            for call in calls:
                name = call.function.name
                args = dict(call.function.arguments or {})
                yield {"type": "status", "text": labels.get(name, "Working on it…")}

                if name == "get_order" and args.get("order_id") != s.order_id:
                    _t(s, "GUARDRAIL", "Argument guardrail · order ID rebound",
                       "The model's order ID differs from the one the customer gave. Trusted session value used.",
                       {"model_args": args, "safe_args": {"order_id": s.order_id}},
                       "agent.py · _run_decision()  (Stage 5)", "warn")
                    args["order_id"] = s.order_id  # trusted binding
                if name == action:
                    # Action gate: rebind to orchestrator-held inputs; run any
                    # prerequisite checks the model skipped.
                    safe = {"order_id": s.order_id, "reason": s.reason, "photo_supplied": s.photo}
                    _t(s, "GUARDRAIL", "Action gate · consequential call rebound to trusted inputs",
                       f"{action} is the only way to get an outcome. Its arguments always come from verified "
                       "session state, never from model text." +
                       (" (Model's arguments differed.)" if safe != args else " (Model's arguments already matched.)"),
                       {"model_args": args, "safe_args": safe},
                       "agent.py · _run_decision()  (Stage 5)", "warn" if safe != args else "ok")
                    args = safe
                    prereqs = [("policy", s.policy_searched, "search_policy", {"query": s.reason}),
                               ("history", s.history_checked, "get_refund_history", {"customer_id": s.order["customer_id"]}),
                               ("stock", s.inventory_checked, "check_inventory", {"item": s.order["item"]})]
                    for what, done, tool, targs in prereqs:
                        if not done:
                            _t(s, "GUARDRAIL", f"Sequencing gate · {what} not yet checked",
                               "Model skipped a required step; the orchestrator runs it before allowing a decision.",
                               {}, "agent.py · _run_decision()", "warn")
                            yield {"type": "status", "text": labels[tool]}
                            tool_map[tool](**targs)

                fn = tool_map.get(name)
                if not fn:
                    output = {"error": f"unknown_tool:{name}"}
                    _t(s, "ERROR", f"Unknown tool requested: {name}", "Rejected; not in the allow-list.",
                       {"tool_map": list(tool_map)}, "agent.py · _run_decision()", "error")
                else:
                    try:
                        output = fn(**args)
                    except Exception as exc:
                        output = {"error": f"{type(exc).__name__}: {exc}"}
                        _t(s, "ERROR", f"Tool {name} failed", str(exc), {"args": args},
                           "agent.py · _run_decision()", "error")
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": output if isinstance(output, str) else json.dumps(output),
                })
            _state(s)
            continue

        if s.proposal is None:
            # Action gate: no outcome without the controlled tool.
            _t(s, "GUARDRAIL", "Action gate · model tried to finish early",
               f"No outcome is allowed unless {action} has run. Control is returned to the model.",
               {"model_text": (msg.content or "")[:300]}, "agent.py · _run_decision()  (Stage 5)", "warn")
            messages.append({
                "role": "user",
                "content": (
                    f"ACTION GATE: You have not called {action}. "
                    "You may not give an outcome yet. "
                    f"Call {action} using the verified facts."
                ),
            })
            continue

        yield from _handle_outcome(s, (msg.content or "").strip())
        return

    # Stop rule: never loop forever - hand over to a human.
    _t(s, "GUARDRAIL", "Stop rule · step limit reached",
       f"The agent used all {MAX_STEPS} steps without an outcome. Escalating instead of looping.",
       {"max_steps": MAX_STEPS}, "agent.py · _run_decision()  (Stage 5)", "warn")
    yield from _escalate(s, "I couldn't complete the checks automatically.")


def _describe(o):
    if o["type"] == "REPLACEMENT":
        return f"a replacement {o['item']}"
    word = "a full refund" if o["type"] == "REFUND_FULL" else f"a {o['percent']}% partial refund"
    return f"{word} of ${o['amount']:g}"


def _finalize(s, choice):
    """Customer picked one of the options the rules produced. Validate, then close."""
    o = next((x for x in s.options if x["type"] == choice), None)
    if not o:  # defensive: only options computed by code can ever be granted
        _t(s, "GUARDRAIL", "Choice rejected · not an eligible option",
           "The customer asked for something the rules did not offer.", {"choice": choice, "eligible": s.options},
           "agent.py · _finalize()", "warn")
        return None
    s.resolution = o
    s.awaiting_choice = False
    s.phase = "closed"
    _t(s, "DECISION", f"Customer chose: {o['type']}",
       "The choice is validated against the options the deterministic rules computed. Nothing else can be granted.",
       {"chosen": o, "eligible": s.options}, "agent.py · _finalize()")
    _state(s)
    if o["type"] == "REPLACEMENT":
        text = (f"Done - a replacement {o['item']} for order {s.order_id} has been approved. "
                "(Demo store: nothing is actually shipped.)")
        summary = {"kind": "replacement", "order_id": s.order_id, "item": o["item"], "simulation": True}
    else:
        text = (f"Done - {_describe(o)} for your {s.order['item']} (order {s.order_id}) has been approved. "
                "(Demo store: no real refund is processed.)")
        summary = {"kind": "refund", "order_id": s.order_id, "item": s.order["item"], "amount": o["amount"],
                   "percent": o["percent"], "simulation": True}
    return text, summary


def _handle_outcome(s, model_text):
    p = s.proposal
    model_text = re.sub(r"<think>.*?</think>", "", model_text, flags=re.S).strip()
    decision = p["decision"]
    _state(s)
    _t(s, "DECISION", f"Route outcome: {decision}",
       "The decision came from code. The model's own wording is not used for the verdict.",
       {"tool_result": p, "model_closing_text_ignored": model_text[:300]},
       "agent.py · _handle_outcome()")

    if decision == "HUMAN_REVIEW":
        yield from _escalate(s, _friendly(p["reason"]))
        return

    if decision == "DENIED":
        s.phase = "flow"
        _state(s)
        text = (f"I'm sorry - I can't offer a refund or replacement for this one. {p['reason']}. "
                "If you think that's wrong, I can connect you to an executive.")
        yield _say(s, text, "fixed template built from the tool result (no LLM)")
        yield {"type": "ui", "action": "offer_executive"}
        return

    if decision == "AUTO_APPROVE":
        s.options = p["options"]
        # Put the customer's preferred option first (their menu choice), but offer everything eligible.
        pref = "REPLACEMENT" if s.flow == "replacement" else "REFUND"
        s.options.sort(key=lambda o: 0 if o["type"].startswith(pref) else 1)
        note = ""
        if s.flow == "replacement" and not any(o["type"] == "REPLACEMENT" for o in s.options):
            note = f" A replacement isn't available (the {s.order['item']} is out of stock or not eligible), so here is what I can do instead."
        if len(s.options) == 1:
            done = _finalize(s, s.options[0]["type"])
            text, summary = done
            if note:
                text = note.strip() + " " + text
            yield _say(s, text, "fixed template built from the tool result (no LLM, so no invented promises)")
            yield {"type": "ui", "action": "approved", "summary": summary}
            return
        s.awaiting_choice = True
        why = s.options[0]["why"]
        text = ("Good news - you're eligible for " + " or ".join(_describe(o) for o in s.options) +
                f". {why} Which would you prefer?")
        _t(s, "DECISION", "Eligible for more than one resolution → customer chooses",
           "Both options came from the rules; the customer picks.", {"options": s.options}, "agent.py · _handle_outcome()")
        yield _say(s, text, "fixed template built from the tool result (no LLM)")
        yield {"type": "ui", "action": "choose_resolution",
               "options": [{"type": o["type"], "label": ("Take the " if o["type"] != "REPLACEMENT" else "Send a ") +
                            ("replacement" if o["type"] == "REPLACEMENT" else
                             ("full refund" if o["type"] == "REFUND_FULL" else "partial refund")),
                            "say": "I'd like the replacement" if o["type"] == "REPLACEMENT" else "I'll take the refund"}
                           for o in s.options]}
        return

    # NEEDS_INFORMATION
    s.awaiting_photo = True
    text = _reply(s, "Tell the customer a photo of the item is needed to continue and ask them to use the Upload photo button below.",
                  "To continue I need a photo of the item - please use the Upload photo button.")
    yield _say(s, text)
    yield {"type": "ui", "action": "request_photo"}


# ---------------------------------------------------------------- intent router
INTENTS = ["REFUND", "REPLACEMENT", "SUPPORT", "UNSUPPORTED", "OTHER"]
TONES = ["calm", "frustrated", "abusive"]
ROUTER_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": INTENTS}, "tone": {"type": "string", "enum": TONES}},
    "required": ["intent", "tone"],
}


def detect_tone(text):
    """Code-side tone check. Swearing is only ever flagged by this regex, never by the model alone."""
    if ABUSIVE_RE.search(text):
        return "abusive"
    letters = [c for c in text if c.isalpha()]
    shouting = len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.7
    if FRUSTRATED_RE.search(text) or shouting:
        return "frustrated"
    return "calm"


def route_intent(s, text):
    """Decide what the customer wants. The LLM only classifies; code decides what to do with it."""
    last_bot = next((t for r, t in reversed(s.history) if r == "assistant"), "")
    s.tone = detect_tone(text)  # code rules decide tone (the model's opinion is logged only)

    # Cheap code rules first: answers to a question we just asked are not new intents.
    if s.awaiting_choice:
        wants_rep, wants_ref = bool(REPLACE_RE.search(text)), bool(REFUND_WORD_RE.search(text))
        has = {o["type"] for o in s.options}
        if wants_rep and not wants_ref and "REPLACEMENT" in has:
            _t(s, "ROUTER", "Intent routing → CHOOSE_REPLACEMENT (code rule, no LLM)",
               "The customer was asked to choose and said replacement.", {"text": text}, "agent.py · route_intent()")
            return "CHOOSE_REPLACEMENT"
        if wants_ref and not wants_rep and any(t.startswith("REFUND") for t in has):
            _t(s, "ROUTER", "Intent routing → CHOOSE_REFUND (code rule, no LLM)",
               "The customer was asked to choose and said refund.", {"text": text}, "agent.py · route_intent()")
            return "CHOOSE_REFUND"
        if NO_RE.search(text):
            _t(s, "ROUTER", "Intent routing → DECLINE (code rule, no LLM)",
               "The customer declined the options.", {"text": text}, "agent.py · route_intent()")
            return "DECLINE"
    if s.flow and SLOT_ONLY_RE.match(text):
        _t(s, "ROUTER", "Intent routing → stay in flow (code rule, no LLM)",
           "The message is just an order ID or an amount, so it answers the current question.",
           {"text": text, "flow": s.flow}, "agent.py · route_intent()")
        return "OTHER"

    prompt = f"""
Classify the customer's LATEST message into exactly one intent.

REFUND      - wants money back for an order (damaged, missing, wrong, unwanted)
REPLACEMENT - wants the item replaced, exchanged, or sent again
SUPPORT     - wants a human, agent, manager, or wants to complain
UNSUPPORTED - asks to track a parcel, asks for order status/details, or asks a general policy question
OTHER       - greeting, thanks, small talk, or an answer to the assistant's last question

TONE: calm, frustrated (annoyed, repeating themselves, complaining the bot doesn't understand), or abusive (insults, swearing).

Current conversation topic: {s.flow or "none"}
Assistant's last message: "{last_bot[:200]}"
Customer's latest message: "{text}"

Return JSON with "intent" and "tone" only.
""".strip()
    sp = _t_start(s, "LLM", "LLM · intent router (structured output)",
                  f"{_model(s)} classifies the message into one allowed intent. It routes; it does not decide anything.",
                  {"model": _model(s), "prompt": prompt, "json_schema": ROUTER_SCHEMA},
                  "agent.py · route_intent()  (phase 2)")
    intent, raw = None, None
    try:
        t0 = time.time()
        resp = ollama.chat(model=_model(s), messages=[{"role": "user", "content": prompt}], format=ROUTER_SCHEMA)
        u = _record(s, "intent router", "llm", _model(s), resp, t0)
        raw = json.loads(resp.message.content)
        intent = raw.get("intent")
        model_tone = raw.get("tone")
        _t_end(s, sp, f"Model said: {intent} · tone {model_tone}", {**u, "raw_json": raw})
        # The model's tone opinion is logged for teaching, but only the code rules decide tone:
        # a 4B model over-flags calm messages, and swearing/frustration must be testable.
    except Exception as exc:
        _t_end(s, sp, f"Router failed: {type(exc).__name__}", {"error": str(exc)}, "error")

    if intent not in INTENTS:
        # Fallback: keyword rules (same words the phase-1 bot used).
        if SUPPORT_RE.search(text):
            intent = "SUPPORT"
        elif REPLACE_RE.search(text):
            intent = "REPLACEMENT"
        elif REFUND_INTENT.search(text):
            intent = "REFUND"
        else:
            intent = "OTHER"
        _t(s, "CODE", "Router fallback · keyword rules used",
           "The LLM result was missing or not in the allow-list.", {"model_raw": raw, "fallback_intent": intent},
           "agent.py · route_intent()", "warn")
    else:
        _t(s, "CODE", f"Validate intent against allow-list → {intent}",
           "Only these intents can change what the bot does.", {"allowed": INTENTS, "accepted": intent},
           "agent.py · route_intent()")
    _t(s, "CODE", f"Tone check → {s.tone}",
       "Code rules (regex + shouting) decide tone; the model's opinion is only logged. Tone changes how the bot responds, never the refund rules.",
       {"tone": s.tone, "frustration_count": s.frustration, "abuse_strikes": s.abuse_strikes},
       "agent.py · detect_tone()", "warn" if s.tone != "calm" else "ok")
    return intent


def _switch_flow(s, target, why):
    """Change flow but keep the verified order. Per-flow checks are reset."""
    prev = s.flow
    s.flow = target
    s.phase = target
    s.proposal = None
    s.options, s.awaiting_choice = [], False
    s.awaiting_photo = False
    s.policy_searched = s.history_checked = s.inventory_checked = False
    s.pending = bool(s.order and s.reason)
    _t(s, "ROUTER", f"Flow: {prev or 'none'} → {target}",
       why + (" Verified order and what the customer said are kept; policy/history/stock checks are redone for the new flow." if s.order else ""),
       {"order_kept": s.order_id, "reason_kept": s.reason}, "agent.py · _switch_flow()")
    _state(s)


# ---------------------------------------------------------------- slot handling + dialog
def _update_slots(s, text):
    """Extract + validate slots from the customer's message. Returns notice or None."""
    known = {"order_id": s.order_id, "reason": s.reason}
    got = extract_slots(text, known, s)
    notice = None

    if got["order_id"] and got["order_id"] != s.order_id:
        sp = _t_start(s, "TOOL", "Tool · look up order in orders.json",
                      "Verifying the customer's order ID against the system of record.",
                      {"order_id": got["order_id"]}, "agent.py · _update_slots()  (Stage 2)")
        order = next((x for x in ORDERS if x["order_id"] == got["order_id"]), None)
        if order:
            s.order, s.order_id = order, order["order_id"]
            s.policy_searched = s.history_checked = s.inventory_checked = False
            s.photo = s.awaiting_photo = False
            s.pending = True
            _t_end(s, sp, f"Order verified: {order['item']}, ${order['amount_paid']}", {"order": order})
        else:
            s.attempts += 1
            notice = f"not_found:{got['order_id']}"
            _t_end(s, sp, f"Order not found (failure {s.attempts} of {MAX_ORDER_ATTEMPTS})",
                   {"order_id": got["order_id"], "attempts": s.attempts}, "warn")

    if got["reason"] and got["reason"] != s.reason:
        s.reason = got["reason"]
        s.claim_type = classify_claim(s.reason, s)
        s.pending = True

    _state(s)
    return notice


def _last_user(s):
    return next((t for r, t in reversed(s.history) if r == "user"), "")


def _ask(s, key, instruction, fallback, help_text):
    """Ask for a missing detail. If the customer is stuck (same ask again and again),
    help once, then offer an executive instead of looping forever."""
    s.stuck = s.stuck + 1 if s.last_ask == key else 1
    s.last_ask = key
    if key == "order" and NO_ID_RE.search(_last_user(s)):
        s.stuck = max(s.stuck, 2)  # they told us they don't have it: skip straight to help
    _t(s, "ROUTER", f"Stuck counter: asked for {key} · attempt {s.stuck}",
       "The same question without progress. Attempt 1 asks normally, attempt 2 gives help and offers an executive, attempt 3 hands off.",
       {"key": key, "attempt": s.stuck}, "agent.py · _ask()  (stuck handling)", "warn" if s.stuck > 1 else "ok")
    if s.stuck >= 3:
        yield from _escalate(s, "I haven't been able to get the details I need to help you here.")
        return
    if s.stuck == 2:
        yield _say(s, help_text, "fixed template (no LLM)")
        yield {"type": "ui", "action": "offer_executive"}
        return
    yield _say(s, _reply(s, instruction, fallback))


def _dialog(s, title, detail, data=None):
    _t(s, "ROUTER", title, detail, data, "agent.py · _continue_flow()  (dialog policy: code decides what to ask)")


def _continue_flow(s, notice=None):
    noun = FLOW_CFG[s.flow]["noun"]
    missing = [k for k, v in (("order", s.order), ("reason", s.reason)) if not v]
    ID_HELP = ("No problem if you don't have it handy. Your order ID (it looks like ORD-12345) is in your order confirmation "
               "email and under “My orders” in your account. If you can't find it, I can connect you to an executive.")
    if notice and notice.startswith("not_found:"):
        if s.attempts >= MAX_ORDER_ATTEMPTS:
            _dialog(s, "Dialog policy → escalate", f"{s.attempts} failed order lookups, so a human takes over.")
            yield from _escalate(s, "I couldn't find that order after a few tries.")
            return
        bad = notice.split(":", 1)[1]
        _dialog(s, "Dialog policy → ask for order ID again", f"Lookup of {bad} failed.", {"missing": missing})
        yield from _ask(s, "order",
                        f"Say you couldn't find order {bad}, and ask them to double-check the order ID (format ORD-12345).",
                        f"I couldn't find order {bad}. Could you double-check the order ID? (It looks like ORD-12345.)", ID_HELP)
        return

    if not s.order:
        _dialog(s, "Dialog policy → ask for order ID", "No verified order yet.", {"missing": missing})
        yield from _ask(s, "order", f"Ask for their order ID (format ORD-12345) so you can look into the {noun}.",
                        "Sure - what's your order ID? (It looks like ORD-12345.)", ID_HELP)
        return
    if not s.reason:
        _dialog(s, "Dialog policy → ask what went wrong", "Order verified, reason still missing.", {"missing": missing})
        yield from _ask(s, "reason", f"You found their {s.order['item']} order. Ask what went wrong with it.",
                        f"Thanks - I found your {s.order['item']} order. What went wrong with it?",
                        "I still need to know what went wrong with the order (for example: damaged, never arrived, or the wrong item). If it's easier, I can connect you to an executive.")
        return
    if s.pending:
        s.pending = False
        _dialog(s, "Dialog policy → all details collected, run the decision",
                f"Everything needed for a {noun} is known and something changed, so the guarded agent loop runs.",
                {"flow": s.flow, "order": s.order_id, "reason": s.reason, "claim_type": s.claim_type,
                 "photo": s.photo})
        yield from _run_decision(s)
        return

    if s.awaiting_photo and not s.photo:
        _dialog(s, "Dialog policy → still waiting for the photo", "Decision is NEEDS_INFORMATION until an image arrives.")
        yield _say(s, _reply(s, "Remind the customer to use the Upload photo button so you can continue.", "Please use the Upload photo button so I can continue."))
        yield {"type": "ui", "action": "request_photo"}
        return
    _dialog(s, "Dialog policy → free-form question", "Nothing new to decide; the model answers briefly.")
    yield _say(s, _reply(s, f"Answer their question helpfully and briefly, staying on their {noun} request.",
                         f"Thanks - could you tell me a bit more about what you need for your {noun}?"))


# ---------------------------------------------------------------- entry points
def handle_message(s, text):
    """Generator of UI events for one customer message."""
    s.turn += 1
    s.turn_t0 = time.time()
    s.history.append(("user", text))
    _t(s, "INPUT", "Customer message received", f"“{text[:120]}”",
       {"text": text, "session": s.id, "phase": s.phase, "flow": s.flow}, "server.py · POST /api/message")
    if s.phase == "closed":
        _t(s, "ROUTER", "Conversation already closed", "No further automated handling.", {}, "agent.py · handle_message()")
        yield _say(s, "This conversation has ended. Please start a new chat if you need anything else.", "fixed template (no LLM)")
        yield {"type": "ui", "action": "closed"}
        return

    intent = route_intent(s, text)

    if intent == "SUPPORT":
        yield from _escalate(s, "You asked to speak with someone.")
        return

    if s.tone == "abusive":
        s.abuse_strikes += 1
        _t(s, "GUARDRAIL", f"Tone guardrail · abusive language (strike {s.abuse_strikes} of 2)",
           "The bot never mirrors abuse. First strike: one calm boundary and an executive offer. Second strike: hand off.",
           {"text": text, "strikes": s.abuse_strikes}, "agent.py · handle_message()  (tone handling)", "warn")
        if s.abuse_strikes >= 2:
            yield from _escalate(s, "I'm sorry this has been so frustrating.")
            return
        yield _say(s, "I'm sorry this has been frustrating, and I do want to get it sorted for you. I can't carry on if the conversation "
                      "becomes abusive - would you like to keep going, or shall I connect you to an executive?",
                   "fixed template (no LLM, never mirrors the language)")
        yield {"type": "ui", "action": "offer_executive"}
        return

    if s.tone == "frustrated":
        s.frustration += 1
        _t(s, "GUARDRAIL", f"Tone guardrail · frustrated customer (count {s.frustration} of 3)",
           "Acknowledge once, keep helping, offer an executive early, and hand off if it keeps happening.",
           {"text": text, "frustration": s.frustration}, "agent.py · handle_message()  (tone handling)", "warn")
        if s.frustration >= 3:
            yield from _escalate(s, "I'm sorry I haven't been able to sort this out for you.")
            return
        if s.frustration == 1:
            yield _say(s, "I'm really sorry for the hassle - I want to get this fixed for you. If you'd rather talk to a person, I can connect you.",
                       "fixed template (no LLM)")
            yield {"type": "ui", "action": "offer_executive"}

    if intent == "DECLINE":
        s.awaiting_choice = False
        yield from _escalate(s, "No problem - an executive can talk through other options with you.")
        return

    if intent in ("CHOOSE_REFUND", "CHOOSE_REPLACEMENT"):
        want = "REPLACEMENT" if intent == "CHOOSE_REPLACEMENT" else next(
            (o["type"] for o in s.options if o["type"].startswith("REFUND")), None)
        done = _finalize(s, want)
        if done:
            text, summary = done
            yield _say(s, text, "fixed template built from the validated choice (no LLM)")
            yield {"type": "ui", "action": "approved", "summary": summary}
        return

    if s.awaiting_choice and intent in ("OTHER", "REFUND", "REPLACEMENT"):
        _t(s, "ROUTER", "Waiting for a choice", "The customer hasn't picked yet; re-offer the options.",
           {"options": s.options}, "agent.py · handle_message()")
        yield _say(s, "Just so I get it right - which would you prefer: " + " or ".join(_describe(o) for o in s.options) + "?",
                   "fixed template (no LLM)")
        yield {"type": "ui", "action": "choose_resolution",
               "options": [{"type": o["type"], "label": ("Send a replacement" if o["type"] == "REPLACEMENT" else "Take the refund"),
                            "say": "I'd like the replacement" if o["type"] == "REPLACEMENT" else "I'll take the refund"} for o in s.options]}
        return

    if intent == "UNSUPPORTED":
        yield _say(s, "That isn't something I can help with here just yet. I can help with refunds and replacements - or connect you to an executive.",
                   "fixed template (no LLM)")
        yield {"type": "ui", "action": "show_menu"}
        _state(s)
        return

    if intent in ("REFUND", "REPLACEMENT"):
        target = intent.lower()
        if s.flow != target:
            _switch_flow(s, target, "The router saw a new intent.")
    elif not s.flow:
        _t(s, "ROUTER", "No intent yet → greet and show the menu", "Greetings and small talk never start a flow.",
           {"intent": intent}, "agent.py · handle_message()")
        yield _say(s, "Hi, welcome to Worstbuy! 👋 I can help with the options below - how can I help today?", "fixed template (no LLM)")
        yield {"type": "ui", "action": "show_menu"}
        _state(s)
        return

    yield {"type": "status", "text": "Reading your message…"}
    notice = _update_slots(s, text)
    yield from _continue_flow(s, notice)


def handle_photo(s):
    """Customer uploaded an image. Any image counts; it is not analysed."""
    s.turn += 1
    s.turn_t0 = time.time()
    _t(s, "INPUT", "Photo uploaded", "Any image counts as evidence in this demo; its pixels are not analysed.",
       {"phase": s.phase, "flow": s.flow}, "server.py · POST /api/photo")
    if not s.flow or not s.order:
        yield _say(s, "Thanks for the photo! Let's start with your request first - what's your order ID?", "fixed template (no LLM)")
        return
    s.history.append(("user", "(uploaded a photo)"))
    s.photo = True
    s.awaiting_photo = False
    s.pending = True
    _t(s, "CODE", "State updated: photo_supplied = true", "Orchestrator state changes; the model is not asked.",
       {}, "agent.py · handle_photo()")
    _state(s)
    yield from _continue_flow(s)
