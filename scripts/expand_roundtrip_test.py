#!/usr/bin/env python3
"""End-to-end test of the skeleton expand round-trip.

1. Send a heavy file through the test proxy (:3001) with a pinned session id —
   the proxy compresses it (dropping bodies) AND stores the source.
2. Call POST /v1/expand with that session + a symbol name — the proxy returns the
   full body the digest dropped.
"""
import json
import urllib.request

PROXY = "http://127.0.0.1:3001"
SESSION = "expand-roundtrip-demo"
TARGET = r"C:\Users\garza\AXIOM-AETHER\axiom_engine_rs\src\server.rs"
SYMBOL = "unix_now"  # a fn within the first 12KB of server.rs


def post(path, body, headers=None):
    h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01",
         "x-api-key": "test"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(PROXY + path, data=json.dumps(body).encode(),
                                 method="POST", headers=h)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)


def main():
    with open(TARGET, "r", encoding="utf-8", errors="ignore") as f:
        big = f.read()[:12000]

    print("STEP 1 — compress a file (drops bodies, stores source)")
    st, _ = post("/v1/messages", {
        "model": "claude-haiku-4-5", "max_tokens": 16,
        "messages": [
            {"role": "user", "content": big},
            {"role": "user", "content": "what does this file do?"},
        ],
    }, headers={"X-Axiom-Session-Id": SESSION})
    print(f"  /v1/messages -> HTTP {st}")

    print(f"\nSTEP 2 — expand the dropped symbol '{SYMBOL}'")
    st, resp = post("/v1/expand", {"session_id": SESSION, "symbol": SYMBOL})
    print(f"  /v1/expand -> HTTP {st}")
    try:
        d = json.loads(resp)
    except Exception:
        print("  raw:", resp[:300]); return
    if d.get("found"):
        body = d["body"]
        print(f"  found=True, body is {len(body)} chars. First lines:\n")
        for line in body.splitlines()[:8]:
            print("    " + line)
        print("\n  [OK] ROUND-TRIP WORKS: the dropped body was recovered on demand.")
    else:
        print("  found=False:", d.get("error"))


if __name__ == "__main__":
    main()
