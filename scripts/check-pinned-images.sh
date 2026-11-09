#!/usr/bin/env bash
# check-pinned-images.sh — fail on any :latest or digest-less image reference
# in the Helm chart values, chart templates, and Dockerfile.
#
# Invariant: every deployable image reference must carry a sha256 digest pin.
# A `latest` tag is only acceptable when a digest is set alongside it, because
# the rendered reference is then `repo:tag@digest` and the digest is what the
# runtime actually pulls.
#
# Usage: ./scripts/check-pinned-images.sh [repo-root]
# Exits 0 when every check passes, 1 otherwise. No network access needed.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VALUES="$ROOT/deploy/helm/axiom/values.yaml"
DEPLOYMENT="$ROOT/deploy/helm/axiom/templates/deployment.yaml"
GOSSIP="$ROOT/deploy/helm/axiom/templates/gossip.yaml"
TEST_CONN="$ROOT/deploy/helm/axiom/templates/tests/test-connection.yaml"
DOCKERFILE="$ROOT/Dockerfile"

FAILURES=0
fail() { echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); }
pass() { echo "PASS: $1"; }

# --- values.yaml: each of the three images needs a sha256 digest -----------
# python exits with the failure count, so the summary matches the FAIL lines.
PY_FAILS=0
python3 - "$VALUES" <<'PYEOF' || PY_FAILS=$?
import re, sys

path = sys.argv[1]
digest_re = re.compile(r"^sha256:[0-9a-f]{64}$")
# Section-aware scan: top-level keys start at column 0; digest lines sit at
# two-space indent directly under their section.
sections = {}
current = None
with open(path) as f:
    for line in f:
        if re.match(r"^[A-Za-z]", line):
            current = line.split(":")[0].strip()
            continue
        m = re.match(r"^  digest:\s*\"?([^\"\s]+)\"?\s*$", line)
        if m and current:
            sections[current] = m.group(1)

failures = 0
for section in ("image", "gossip", "tests"):
    value = sections.get(section)
    if value is None:
        print(f"FAIL: values.yaml [{section}] has no digest field")
        failures += 1
    elif not digest_re.match(value):
        print(f"FAIL: values.yaml [{section}] digest is not a sha256 pin: {value}")
        failures += 1
    else:
        print(f"PASS: values.yaml [{section}] digest pinned ({value[:19]}...)")
sys.exit(failures)
PYEOF
FAILURES=$((FAILURES + PY_FAILS))

# --- values.yaml: no :latest image ref may stand without its digest --------
# (Covered by the digest requirement above, but reported explicitly so a
# bare `tag: latest` or `image: ...:latest` reads as the finding it is.)
if grep -qE '^\s+tag:\s*latest\s*$' "$VALUES" && ! grep -qE '^\s+digest:\s*"sha256:[0-9a-f]{64}"' "$VALUES"; then
  fail "values.yaml sets image.tag latest with no digest pinned"
else
  pass "values.yaml has no unpinned :latest tag"
fi
if grep -qE '^\s+image:\s*[^ ]*:latest\s*$' "$VALUES" && ! grep -qE '^\s+digest:\s*"sha256:[0-9a-f]{64}"' "$VALUES"; then
  fail "values.yaml sets an image to :latest with no digest pinned"
else
  pass "values.yaml has no unpinned :latest image string"
fi

# --- templates: the rendered image must include the digest -----------------
check_template() {
  local file="$1" needle="$2" label="$3"
  if grep -qF "$needle" "$file"; then
    pass "$label renders with digest"
  else
    fail "$label does not reference its .Values digest"
  fi
  if grep -qE 'image:.*:latest' "$file"; then
    fail "$label hardcodes a :latest image"
  fi
}
check_template "$DEPLOYMENT" '.Values.image.digest' "deployment.yaml main image"
check_template "$GOSSIP" '.Values.gossip.digest' "gossip.yaml gossip image"
check_template "$TEST_CONN" '.Values.tests.digest' "test-connection.yaml test image"

# --- Dockerfile: every FROM line must pin a digest -------------------------
while IFS= read -r line; do
  if ! echo "$line" | grep -qE '@sha256:[0-9a-f]{64}'; then
    fail "Dockerfile FROM without digest pin: $line"
  else
    pass "Dockerfile FROM pinned: $(echo "$line" | awk '{print $2}' | cut -c1-40)..."
  fi
done < <(grep -E '^FROM ' "$DOCKERFILE")

if [ "$FAILURES" -ne 0 ]; then
  echo "check-pinned-images: $FAILURES failure(s)"
  exit 1
fi
echo "check-pinned-images: all image references pinned"
