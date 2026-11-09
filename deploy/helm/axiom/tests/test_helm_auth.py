#!/usr/bin/env python3
"""Regression test: authenticated Helm gossip (audit Finding 6, P1).

Renders the axiom chart with `helm template` in three configurations and
asserts on the rendered output (behavior, not template text):

* apiKey set: the Secret carries AXIOM_API_KEY, the gossip CronJob and the
  `helm test` pod both receive it from the Secret, and both send
  `X-Axiom-Key` on data-plane probes (/v1/immunity, /v1/immunity/merge,
  /v1/models). /readyz stays unauthenticated.
* apiKey unset (defaults): no AXIOM_API_KEY anywhere in the rendered
  output's env, so the key-disabled config keeps working unchanged.
* existingSecret set: gossip and test pods reference that Secret by name
  instead of the chart-created one.

Usage: python3 deploy/helm/axiom/tests/test_helm_auth.py
Requires: helm (v3) on PATH, PyYAML.
Exit code is 0 only when every assertion passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

CHART = Path(__file__).resolve().parents[1]
RELEASE = "test-release"
TEST_KEY = "testkey123"


def render(extra_args: list[str]) -> list[dict]:
    cmd = ["helm", "template", RELEASE, str(CHART), *extra_args]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stdout}\n{out.stderr}")
    return [d for d in yaml.safe_load_all(out.stdout) if d]


def by_kind(docs: list[dict], kind: str, name_sub: str = "") -> dict:
    for d in docs:
        if d.get("kind") == kind and name_sub in d.get("metadata", {}).get("name", ""):
            return d
    raise AssertionError(f"no {kind} matching {name_sub!r} in rendered output")


def containers(doc: dict) -> list[dict]:
    spec = doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return spec["containers"]


def env_entries(container: dict) -> list[dict]:
    return container.get("env", []) or []


def env_names(container: dict) -> list[str]:
    return [e["name"] for e in env_entries(container)]


def secret_ref(container: dict, name: str) -> dict | None:
    for e in env_entries(container):
        if e["name"] == name:
            return e.get("valueFrom", {}).get("secretKeyRef")
    return None


def check_api_key_set(docs: list[dict]) -> list[str]:
    failures = []
    secret = by_kind(docs, "Secret")
    if (secret.get("stringData") or {}).get("AXIOM_API_KEY") != TEST_KEY:
        failures.append("Secret does not carry AXIOM_API_KEY=testkey123")

    gossip = by_kind(docs, "CronJob", "gossip")
    gossip_ctr = containers(gossip)[0]
    ref = secret_ref(gossip_ctr, "AXIOM_API_KEY")
    if not ref or ref.get("key") != "AXIOM_API_KEY" or not ref.get("name"):
        failures.append("gossip CronJob env lacks AXIOM_API_KEY secretKeyRef")
    if ref and ref.get("optional") is not True:
        failures.append("gossip AXIOM_API_KEY secretKeyRef must be optional:true")

    cm = by_kind(docs, "ConfigMap", "gossip")
    script = cm["data"]["gossip.sh"]
    if "X-Axiom-Key" not in script:
        failures.append("gossip.sh never sends X-Axiom-Key")
    if "AXIOM_API_KEY:-" not in script:
        failures.append("gossip.sh does not guard on AXIOM_API_KEY presence")
    joined = script.replace("\\\n", " ")
    for line in joined.splitlines():
        if "curl" in line and "/v1/immunity" in line and "curl_auth" not in line:
            failures.append(f"gossip.sh calls an immunity endpoint without auth wrapper: {line.strip()}")

    test_pod = by_kind(docs, "Pod", "test-connection")
    test_ctr = test_pod["spec"]["containers"][0]
    ref = secret_ref(test_ctr, "AXIOM_API_KEY")
    if not ref or ref.get("key") != "AXIOM_API_KEY" or not ref.get("name"):
        failures.append("test pod env lacks AXIOM_API_KEY secretKeyRef")
    test_script = test_ctr["args"][0]
    if "X-Axiom-Key" not in test_script:
        failures.append("test pod never sends X-Axiom-Key on /v1/models")
    readyz_lines = [ln for ln in test_script.splitlines() if "/readyz" in ln]
    if any("X-Axiom-Key" in ln or "curl_auth" in ln for ln in readyz_lines):
        failures.append("/readyz probe must stay unauthenticated")

    deploy = by_kind(docs, "Deployment")
    secret_refs = deploy["spec"]["template"]["spec"]["containers"][0].get("envFrom", [])
    if not any(s.get("secretRef", {}).get("name") for s in secret_refs):
        failures.append("Deployment lost its Secret envFrom (app pods need AXIOM_API_KEY)")
    return failures


def check_api_key_unset(docs: list[dict]) -> list[str]:
    failures = []
    secret = by_kind(docs, "Secret")
    if "AXIOM_API_KEY" in (secret.get("stringData") or {}):
        failures.append("Secret carries AXIOM_API_KEY although apiKey is unset")

    gossip = by_kind(docs, "CronJob", "gossip")
    if "AXIOM_API_KEY" in env_names(containers(gossip)[0]):
        failures.append("gossip CronJob injects AXIOM_API_KEY although apiKey is unset")

    test_pod = by_kind(docs, "Pod", "test-connection")
    if "AXIOM_API_KEY" in env_names(test_pod["spec"]["containers"][0]):
        failures.append("test pod injects AXIOM_API_KEY although apiKey is unset")

    # The script is key-aware (conditional) but must not send a header
    # unconditionally: every curl call goes through the wrapper.
    cm = by_kind(docs, "ConfigMap", "gossip")
    script = cm["data"]["gossip.sh"]
    if "AXIOM_API_KEY:-" not in script:
        failures.append("gossip.sh lost its key-presence guard")
    return failures


def check_existing_secret(docs: list[dict]) -> list[str]:
    failures = []
    if any(d.get("kind") == "Secret" for d in docs):
        failures.append("chart created a Secret although existingSecret is set")
    gossip = by_kind(docs, "CronJob", "gossip")
    ref = secret_ref(containers(gossip)[0], "AXIOM_API_KEY")
    if not ref or ref.get("name") != "my-existing-secret":
        failures.append("gossip CronJob does not reference existingSecret for AXIOM_API_KEY")
    test_pod = by_kind(docs, "Pod", "test-connection")
    ref = secret_ref(test_pod["spec"]["containers"][0], "AXIOM_API_KEY")
    if not ref or ref.get("name") != "my-existing-secret":
        failures.append("test pod does not reference existingSecret for AXIOM_API_KEY")
    return failures


def main() -> int:
    cases = [
        ("apiKey set", ["--set", f"secrets.apiKey={TEST_KEY}"], check_api_key_set),
        ("apiKey unset (defaults)", [], check_api_key_unset),
        ("existingSecret set",
         ["--set", "secrets.existingSecret=my-existing-secret"],
         check_existing_secret),
    ]
    failed = 0
    for name, args, check in cases:
        try:
            failures = check(render(args))
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures = [f"render/check crashed: {exc}"]
        if failures:
            failed += 1
            print(f"FAIL [{name}]")
            for f in failures:
                print(f"  - {f}")
        else:
            print(f"PASS [{name}]")
    print(f"{len(cases) - failed}/{len(cases)} configurations passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
