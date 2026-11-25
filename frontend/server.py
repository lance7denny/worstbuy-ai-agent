"""Tiny stdlib web server for the refund agent UI.  Run: python frontend/server.py"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
import agent

INDEX = (Path(__file__).parent / "index.html").read_bytes()
HOST, PORT = "127.0.0.1", 8000


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, INDEX, "text/html; charset=utf-8")
        elif self.path == "/api/orders":
            self._send(200, json.dumps(agent.list_orders()).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, b"not found", "text/plain")
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            order_id = str(data["order_id"]).strip()
            reason = str(data["reason"]).strip()
            requested = float(data["requested"])
            photo = bool(data.get("photo", False))
        except (KeyError, ValueError, TypeError):
            return self._send(400, b'{"error":"bad request"}', "application/json")

        # Stream newline-delimited JSON events as the agent works.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for event in agent.run_agent(order_id, reason, requested, photo):
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Refund agent UI → http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
