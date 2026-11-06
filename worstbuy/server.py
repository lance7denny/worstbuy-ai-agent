"""Worstbuy server.  Run: python worstbuy/server.py  ->  http://127.0.0.1:8000"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import json
import queue
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent))
import agent
import attacks
import compare
import evals

INDEX_PATH = Path(__file__).parent / "index.html"
ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = "127.0.0.1", 8000
MAX_PHOTO = 8 * 1024 * 1024
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _session(self, sid):
        with SESSIONS_LOCK:
            return SESSIONS.get(sid)

    def _ndjson(self, my_q, work, transform=None):
        """Run work(my_q) in a thread; stream everything put on my_q as NDJSON until None."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        threading.Thread(target=work, daemon=True).start()
        try:
            while True:
                e = my_q.get()
                if e is None:
                    break
                if transform:
                    e = transform(e)
                    if e is None:
                        continue
                self.wfile.write((json.dumps(e) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream(self, session, events):
        """One chat turn: trace steps, state and chat events for this session."""
        my_q = session.new_queue()  # per-turn queue: a dropped connection can't leak events into the next turn

        def worker():
            with session.lock:  # one turn at a time per session
                try:
                    for e in events():
                        my_q.put(e)
                except Exception as exc:
                    print("error:", type(exc).__name__, exc)
                    my_q.put({"type": "trace", "id": 0, "turn": session.turn, "kind": "ERROR",
                              "phase": "point", "title": f"Server error: {type(exc).__name__}",
                              "detail": str(exc), "data": None, "src": "server.py", "status": "error", "ms": None})
                    my_q.put({"type": "assistant", "text": "Sorry, something went wrong on our side. Please try again."})
                finally:
                    session.turn_wall[session.turn] = int((time.time() - session.turn_t0) * 1000)
                    my_q.put(agent.usage_event(session))  # final totals for the turn
                    my_q.put(None)

        self._ndjson(my_q, worker)

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/prices":
            self._send(200, json.dumps(agent.PRICES).encode())
        elif path == "/api/models":
            self._send(200, json.dumps({"cards": compare.model_cards(), "installed": compare.installed_models()}).encode())
        elif path == "/api/attacks":
            self._send(200, json.dumps(attacks.public_list()).encode())
        elif path == "/api/audit":
            sid = self.path.split("session=")[-1] if "session=" in self.path else ""
            s = self._session(sid)
            if not s:
                return self._send(404, b'{"error":"unknown session"}')
            doc = {
                "session": s.id,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.created)),
                "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "metadata": {
                    "chat_model": agent.CHAT_MODEL, "embedding_model": agent.EMBED_MODEL,
                    "policy_sha256_12": _sha(ROOT / "shared_data/refund_policy.md"),
                    "orders_sha256_12": _sha(ROOT / "shared_data/orders.json"),
                    "inventory_sha256_12": _sha(ROOT / "shared_data/inventory.json"),
                    "stage5_refund_system_prompt": agent.STAGE5_SYSTEM,
                    "resolution_system_prompt": agent.RESOLUTION_SYSTEM,
                    "max_steps": agent.MAX_STEPS,
                    "note": "Demo audit log: in-memory only. Production needs durable, access-controlled, PII-aware storage.",
                },
                "usage": {"totals": agent.usage_totals(s), "per_turn": {str(t): agent.usage_totals(s, t) for t in sorted(s.turn_wall)},
                          "ledger": s.usage, "price_table": agent.PRICES, "models": compare.model_cards()},
                "transcript": [{"role": r, "text": t} for r, t in s.history],
                "events": s.log,
            }
            self._send(200, json.dumps(doc, indent=2).encode(), "application/json",
                       {"Content-Disposition": f'attachment; filename="worstbuy-audit-{s.id[:8]}.json"'})
        else:
            self._send(404, b"not found", "text/plain")

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if self.path == "/api/session":
            sid = uuid.uuid4().hex
            with SESSIONS_LOCK:
                SESSIONS[sid] = agent.Session(sid)
            return self._send(200, json.dumps({"session": sid}).encode())

        if self.path == "/api/message":
            try:
                data = json.loads(self.rfile.read(length))
                text = str(data["text"]).strip()[:1000]
                s = self._session(data["session"])
            except (KeyError, ValueError, TypeError):
                return self._send(400, b'{"error":"bad request"}')
            if not s or not text:
                return self._send(400, b'{"error":"bad request"}')
            return self._stream(s, lambda: agent.handle_message(s, text))

        if self.path == "/api/photo":
            s = self._session(self.headers.get("X-Session", ""))
            ctype = self.headers.get("Content-Type", "")
            if not s or not ctype.startswith("image/") or length > MAX_PHOTO:
                return self._send(400, b'{"error":"bad request"}')
            self.rfile.read(length)  # image is discarded: any image counts
            return self._stream(s, lambda: agent.handle_photo(s))

        if self.path == "/api/evals":
            try:
                suite = json.loads(self.rfile.read(length) or b"{}").get("suite", "fast")
            except ValueError:
                suite = "fast"
            q = queue.Queue()

            def work():
                try:
                    for e in evals.run("full" if suite == "full" else "fast"):
                        q.put(e)
                except Exception as exc:
                    q.put({"type": "eval", "group": "Runner", "name": "eval runner crashed", "status": "fail",
                           "expected": "no crash", "actual": f"{type(exc).__name__}: {exc}", "ms": 0, "detail": ""})
                    q.put({"type": "eval_done", "passed": 0, "failed": 1, "total": 1, "ms": 0})
                finally:
                    q.put(None)

            return self._ndjson(q, work)

        if self.path == "/api/compare":
            try:
                model_b = str(json.loads(self.rfile.read(length) or b"{}").get("model", "qwen3:1.7b")).strip()[:80]
            except ValueError:
                model_b = "qwen3:1.7b"
            q = queue.Queue()

            def work():
                try:
                    for e in compare.run(model_b):
                        q.put(e)
                except Exception as exc:
                    q.put({"type": "compare_error", "message": f"{type(exc).__name__}: {exc}", "fix": None, "installed": []})
                finally:
                    q.put(None)

            return self._ndjson(q, work)

        if self.path == "/api/attack":
            try:
                a = attacks.find(json.loads(self.rfile.read(length))["id"])
            except (KeyError, ValueError, TypeError):
                a = None
            if not a:
                return self._send(400, b'{"error":"unknown attack"}')
            s = agent.Session("attack-" + uuid.uuid4().hex[:6])  # throwaway session, never listed
            q = s.new_queue()  # logging queue: the verdict reads the session log
            label = "⚔ " + a["title"]

            def remap(e):
                """Keep attack trace ids/turns from colliding with the live chat's; only trace + verdict go to the client."""
                if e.get("type") not in ("trace", "attack_result"):
                    return None  # attack chat/ui/state events must never reach the live chat window
                if e.get("type") == "trace":
                    e = dict(e, id=f'{a["id"]}:{e["id"]}', turn=label)
                return e

            def work():
                try:
                    q.put(attacks.run(a, s))
                except Exception as exc:
                    q.put({"type": "attack_result", "id": a["id"], "title": a["title"], "status": "fail",
                           "goal": a["goal"], "decision": "error", "stopped_by": f"{type(exc).__name__}: {exc}",
                           "assistant_said": ""})
                finally:
                    q.put(None)

            return self._ndjson(q, work, remap)

        self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("Warming policy embeddings (nomic-embed-text)…")
    agent.policy_index()
    print(f"Worstbuy → http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
