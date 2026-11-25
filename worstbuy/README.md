# Worstbuy: a refund and replacement chatbot you can see inside

A customer-support agent for a made-up store. It is built to **teach how a production-style AI agent works**:
a chat on the left, a live trace of everything the backend does in the middle, and usage and state on the right.

Everything runs **on your own computer**. No API keys, no cloud, no cost.
It is a simulation: no real money moves, no real orders exist.

## Quick start

You need **Python 3.10+**, **[Ollama](https://ollama.com/download)** (running), and about **3 GB** of disk space for two models.

```bash
./worstbuy/run.sh
```

The script creates a virtual environment, installs the one dependency, offers to download the models if they are missing, and starts the app at http://127.0.0.1:8000.

Prefer to do it by hand?

```bash
ollama pull qwen3:4b-instruct      # chat / routing / tool calling (about 2.5 GB)
ollama pull nomic-embed-text       # embeddings for RAG (about 0.3 GB)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python worstbuy/server.py
```

Then open http://127.0.0.1:8000. A reply takes 30 to 60 seconds on a laptop because a decision makes about nine model calls. That is part of the lesson.

## What to try

| Say | What you will see |
|---|---|
| `Hey, the jug I received is cracked, I need a refund ORD-48213` | Photo requested, then a full refund or replacement offered |
| a refund for `ORD-48214` (headphones, $129) | Handed to an executive (over the $75 limit) |
| a replacement for `ORD-48216` (T-shirt, lost) | Replacement out of stock, so refund only |
| `ORD-48213` and "I changed my mind" | Denied by policy |
| swearing at the bot | One calm boundary, then a handoff |

Demo orders: `ORD-48213` (glass jug, $49), `ORD-48214` (headphones, $129), `ORD-48215` (chair, $210), `ORD-48216` (T-shirt, $35, lost in transit).

## Where things are

- **Live trace** (top nav): every step, LLM call, RAG search, tool call, guardrail and decision, with tokens, seconds and estimated hosted cost.
- **Tests**: evals for the decision rules, attack demos (prompt injection and friends), and a two-model comparison.
- **FAQ**: how it is built, how to read the trace, running and sharing.

| File | Purpose |
|---|---|
| `worstbuy/agent.py` | The orchestrator: router, extraction, dialog, guarded agent loop, rules engine |
| `worstbuy/server.py` | Tiny standard-library web server that streams events |
| `worstbuy/index.html` | The whole UI (chat, trace, usage, tests, FAQ) |
| `worstbuy/evals.py`, `attacks.py`, `compare.py` | The Tests tab |
| `worstbuy/prices.json` | **Placeholder** prices for the "if hosted" estimate. Edit before quoting numbers |
| `shared_data/` | Fake orders, history, inventory and the policy text that RAG searches |

## Sharing and cost

- **Teammates cloning the repo and running it locally cost you nothing**; they use their own machines.
- It **cannot be deployed to Vercel as-is**: the server needs Ollama on the same machine. Hosting would mean swapping in a hosted model API (billed to whoever owns the key), making sessions stateless, and adding login, rate limits and a spend cap. See the FAQ tab.

## Not production-ready

No login, in-memory sessions, fake data, placeholder prices, and a 4B model that can mislabel a claim (the safe failure is a handoff to a human). The workshop README lists what a real deployment also needs.
