#!/usr/bin/env python3
"""Throwaway upstream-capture server for the Axiom compression test.

The test proxy is pointed at this server as its "Anthropic upstream". When the
proxy forwards a (compressed) request, we save the exact outbound body to disk so
we can measure what actually crossed the wire, then return a minimal valid
Anthropic Messages response so the proxy completes cleanly.

Run: python scripts/compress_test_capture.py   (listens on 127.0.0.1:3002)
"""
import json
import http.server
import socketserver

PORT = 3002
CAPTURE = r"C:\Users\garza\AXIOM-AETHER\logs\compress_outbound.json"


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        with open(CAPTURE, "wb") as f:
            f.write(body)
        # Minimal valid Anthropic Messages API response.
        resp = json.dumps({
            "id": "msg_capture", "type": "message", "role": "assistant",
            "model": "capture", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"capture-up")


if __name__ == "__main__":
    print(f"capture server on 127.0.0.1:{PORT} -> {CAPTURE}")
    socketserver.TCPServer(("127.0.0.1", PORT), H).serve_forever()
