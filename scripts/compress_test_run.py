#!/usr/bin/env python3
"""Drive the Axiom compression test and report before/after numbers.

Sends a real source file as 'heavy' context through the test proxy (:3001),
which compresses it and forwards to the capture server. We then compare the
original outbound payload vs the compressed one the proxy actually sent.
"""
import json
import os
import urllib.request

PROXY = "http://127.0.0.1:3001/v1/messages"
CAPTURE = r"C:\Users\garza\AXIOM-AETHER\logs\compress_outbound.json"
TARGET_FILE = r"C:\Users\garza\AXIOM-AETHER\axiom_engine_rs\src\server.rs"


def approx_tokens(s: str) -> int:
    # Consistent rough proxy: ~4 chars/token (matches typical BPE for code).
    return max(1, round(len(s) / 4))


def main():
    with open(TARGET_FILE, "r", encoding="utf-8", errors="ignore") as f:
        big = f.read()
    # Keep the test bounded but clearly "heavy".
    big = big[:12000]

    # Heavy context message (gets compressed) + a small surviving query.
    request = {
        "model": "claude-haiku-4-5",
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": big},
            {"role": "user", "content": "In one sentence, what does this file implement?"},
        ],
    }
    original_body = json.dumps(request)

    # Clear any stale capture
    if os.path.exists(CAPTURE):
        os.remove(CAPTURE)

    req = urllib.request.Request(
        PROXY, data=original_body.encode(), method="POST",
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                 "x-api-key": "test-key-not-used-upstream"},
    )
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        print("request error:", e)

    if not os.path.exists(CAPTURE):
        print("NO CAPTURE — proxy did not forward (compression may have errored).")
        return
    with open(CAPTURE, "r", encoding="utf-8", errors="ignore") as f:
        forwarded = f.read()

    # Measure the heavy message specifically (what compression targets).
    heavy_chars = len(big)
    orig_chars = len(original_body)
    fwd_chars = len(forwarded)

    print("=" * 56)
    print("AXIOM COMPRESSION TEST — real file: server.rs (first 12 KB)")
    print("=" * 56)
    print(f"Heavy context message: {heavy_chars:>8} chars  (~{approx_tokens(big):>6} tokens)")
    print("-" * 56)
    print(f"Full request SENT to proxy : {orig_chars:>8} chars  (~{approx_tokens(original_body):>6} tokens)")
    print(f"What proxy FORWARDED up    : {fwd_chars:>8} chars  (~{approx_tokens(forwarded):>6} tokens)")
    saved = orig_chars - fwd_chars
    pct = (saved / orig_chars * 100) if orig_chars else 0
    print("-" * 56)
    print(f"SAVED ON THE WIRE          : {saved:>8} chars  (~{approx_tokens(' '*saved):>6} tokens)  =  {pct:.1f}% smaller")
    print("=" * 56)

    # Show what replaced the file (the fingerprint), trimmed.
    try:
        fwd = json.loads(forwarded)
        msgs = fwd.get("messages", [])
        print(f"\nForwarded message count: {len(msgs)} (was 2)")
        for m in msgs:
            c = m.get("content")
            txt = c if isinstance(c, str) else json.dumps(c)
            print(f"  [{m.get('role')}] {txt[:160].replace(chr(10),' ')}{'...' if len(txt)>160 else ''}")
    except Exception as e:
        print("(could not parse forwarded body:", e, ")")


if __name__ == "__main__":
    main()
