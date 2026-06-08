#!/usr/bin/env python3
"""Author synthetic (developer-question -> technical-content) pairs for the
contrastive embedder (Phase 2.0.1), WITHOUT calling any LLM/API.

These teach the natural-language-question -> content mapping that mined
doc<->body pairs don't. Topics are broad and phrasings varied so the embedder
GENERALIZES; the eval_recall seeds (auth/db/style/retry/cache) are deliberately
NOT reproduced verbatim here (no test contamination) — different wording, so the
held-out eval remains a real test.

Appends source="synthetic" lines to checkpoints/pairs.jsonl.
"""
import json
import os

# (question, content) — content is a substantive technical statement a developer
# would accept as the answer. Hand-authored, deliberately diverse.
PAIRS = [
    # authentication / sessions / tokens (worded differently from eval seeds)
    ("How do users stay signed in across requests?", "Sessions are kept with a signed bearer token sent in the Authorization header; the server validates its signature and expiry on every request."),
    ("What stops an expired credential from being accepted?", "Each access token carries an exp claim; the middleware rejects any token whose expiry has passed and returns 401."),
    ("How do we let a logged-in user get a fresh token without re-entering a password?", "A long-lived refresh token is exchanged at the token endpoint for a new short-lived access token; refresh tokens rotate on each use."),
    ("Where is the password actually verified?", "On sign-in the submitted password is hashed with Argon2id and compared in constant time against the stored hash; the plaintext is never persisted."),
    ("How is a single-sign-on identity trusted?", "The OIDC provider returns an ID token whose signature is verified against the provider's JWKS before the user is provisioned."),
    ("What scopes can a service account use?", "Service accounts receive a restricted token whose scope claim lists only the APIs they may call; the gateway enforces scope per route."),

    # authorization / access control
    ("How do we decide if a user may perform an action?", "Authorization is role-based: each request's roles are checked against the permission required by the route before the handler runs."),
    ("How is multi-tenant data kept isolated?", "Every row carries a tenant_id and all queries are scoped to the caller's tenant; a missing tenant scope is treated as a hard error."),
    ("Can a user edit another user's records?", "Ownership is enforced server-side: mutations verify the resource's owner_id equals the authenticated user id, returning 403 otherwise."),

    # databases / SQL / migrations
    ("Which datastore backs the service and how is it accessed?", "A relational database is accessed through a typed query layer; raw string concatenation is never used, only parameterized statements."),
    ("How are schema changes rolled out?", "Schema changes are versioned migration files applied in order at deploy time; each migration is forward-only and idempotent."),
    ("How do we avoid the N+1 query problem?", "Related rows are fetched with a single join or a batched IN query rather than one query per parent record."),
    ("What keeps a large result set from exhausting memory?", "List endpoints paginate with LIMIT/OFFSET or keyset cursors; an unbounded query is never issued."),
    ("How are concurrent writes to the same row handled?", "Optimistic concurrency: each row has a version column, and an update fails if the version changed since it was read."),
    ("How do we run a set of writes all-or-nothing?", "The writes are wrapped in a single transaction that commits only if every statement succeeds, rolling back on any error."),

    # caching
    ("How is repeated expensive work avoided?", "Results are memoized in an in-process cache keyed by the inputs, with a short time-to-live so stale values expire."),
    ("What must never be cached?", "Per-user and security-sensitive responses are never cached; only shared, non-personalized data is eligible."),
    ("How is a distributed cache kept consistent on writes?", "Writes invalidate or update the cache key immediately after the database commit, so readers don't see stale data."),
    ("Why use a read-through cache?", "A read-through cache returns hot data without touching the database and repopulates itself on a miss, cutting load and latency."),

    # retries / resilience
    ("How should a transient network failure be handled?", "The call is retried a bounded number of times with exponential backoff and jitter, and gives up with an error after the cap."),
    ("How do we stop hammering a failing dependency?", "A circuit breaker opens after repeated failures, short-circuiting calls for a cooldown before probing the dependency again."),
    ("How are timeouts chosen for outbound calls?", "Every outbound request has an explicit deadline; a slow dependency fails fast rather than blocking the caller indefinitely."),
    ("What makes a retry safe to perform?", "Only idempotent operations are retried; non-idempotent calls carry an idempotency key so duplicates are ignored upstream."),

    # logging / observability
    ("How do we trace one request across services?", "A correlation id is generated at the edge and propagated in headers, so all logs and spans for a request can be joined."),
    ("What format are logs emitted in?", "Logs are structured JSON with a level, timestamp, message, and contextual fields, so they can be queried and aggregated."),
    ("How is application health measured?", "Key metrics — request rate, error rate, and latency percentiles — are exported and alerted on against SLO thresholds."),
    ("How do we avoid leaking secrets into logs?", "Sensitive fields are redacted by the logger before emission; tokens, passwords, and keys are never written out."),

    # error handling
    ("How are errors surfaced to API clients?", "Handlers return a consistent error envelope with a code and human-readable message; internal details are not exposed."),
    ("How are unexpected panics contained?", "A top-level recovery middleware catches panics, logs the stack, and returns a 500 without crashing the process."),
    ("Where should input be validated?", "All external input is validated at the system boundary against a schema before any business logic runs."),

    # testing
    ("How is new functionality verified?", "Each feature ships with unit tests for logic and integration tests for the endpoint, run in CI before merge."),
    ("How are external dependencies handled in tests?", "Dependencies are mocked or stubbed behind an interface so tests are deterministic and don't hit the network."),
    ("What coverage bar must changes meet?", "Changes must keep line coverage above the project threshold; uncovered new code blocks the merge."),
    ("How do we guard against UI regressions?", "Critical screens have screenshot/visual-regression tests at key breakpoints, compared against approved baselines."),

    # concurrency / async
    ("How is shared state protected across threads?", "Shared mutable state is guarded by a mutex or replaced with immutable message passing so there are no data races."),
    ("How are many I/O operations run efficiently?", "Independent I/O is issued concurrently and awaited together rather than sequentially, avoiding request waterfalls."),
    ("How is a long task kept off the request path?", "Long work is enqueued to a background worker and the request returns immediately with a job id to poll."),
    ("What bounds resource use under load?", "A semaphore or worker pool caps in-flight work so a spike can't exhaust connections or memory."),

    # memory / performance
    ("How is memory kept bounded while processing a huge file?", "The file is streamed in fixed-size chunks rather than read fully into memory."),
    ("How was a hot code path made faster?", "Profiling identified the bottleneck, which was replaced with a lower-complexity algorithm and an added index."),
    ("How do we prevent unbounded growth of an in-memory structure?", "The structure is an LRU with a max size that evicts the least-recently-used entries."),

    # networking / HTTP / APIs
    ("How is an API versioned without breaking clients?", "Breaking changes go behind a new version prefix; old versions keep working until clients migrate."),
    ("How are large uploads handled?", "Uploads are streamed to object storage with a size limit and content-type check, not buffered in the app."),
    ("How is a public endpoint protected from abuse?", "Each client is rate-limited with a token bucket; exceeding the limit returns 429 with a Retry-After header."),
    ("What makes responses consistent across endpoints?", "All endpoints return a uniform envelope with status, data, and error fields plus pagination metadata."),

    # serialization / data
    ("How is data exchanged between services?", "Payloads are serialized as JSON with explicit schemas; unknown fields are ignored for forward compatibility."),
    ("How are dates represented on the wire?", "Timestamps are ISO-8601 in UTC, parsed to the local zone only at the presentation layer."),
    ("How is a numeric token amount kept exact?", "Money and token amounts use integer base units, never floating point, to avoid rounding errors."),

    # configuration / secrets
    ("Where does runtime configuration come from?", "Configuration is read from environment variables with validated defaults; required secrets are checked at startup."),
    ("How are secrets kept out of source control?", "Secrets live in a secret manager or untracked env files; the repo ships only an example template."),
    ("What happens if a required secret is missing at boot?", "Startup fails fast with a clear error naming the missing variable rather than running half-configured."),

    # build / deploy / CI
    ("How is a release built reproducibly?", "CI builds an immutable artifact from a pinned dependency lockfile, tagged with the commit hash."),
    ("How are migrations applied during deploy?", "Migrations run as a gated step before the new version takes traffic, and the deploy aborts if they fail."),
    ("How do we roll back a bad deploy?", "Deploys are versioned and immutable, so rollback is repointing traffic to the previous known-good artifact."),

    # security
    ("How is user-supplied HTML rendered safely?", "User content is escaped by default; any allowed HTML is sanitized with a vetted allowlist before rendering."),
    ("How are SQL injection attacks prevented?", "All queries use bound parameters; user input is never interpolated into SQL text."),
    ("How is cross-site request forgery mitigated?", "State-changing requests require a per-session CSRF token validated server-side, plus SameSite cookies."),
    ("How are dependencies kept free of known vulnerabilities?", "A scanner runs in CI against the lockfile and fails the build on high-severity advisories."),

    # queues / messaging
    ("How is a message processed exactly once?", "Consumers are idempotent and dedupe on a message id, so redelivery has no extra effect."),
    ("What happens to a message that keeps failing?", "After a max retry count it is moved to a dead-letter queue for inspection instead of blocking the stream."),

    # frontend / rendering
    ("How is layout shift avoided while images load?", "Images declare explicit width and height (or an aspect ratio) so space is reserved before they load."),
    ("How is a heavy library kept off the initial load?", "It is code-split and dynamically imported only when the feature that needs it is used."),
    ("How does the UI stay responsive during data fetches?", "Cached data renders immediately while a background revalidation updates it when it returns."),

    # general engineering practice
    ("How big should a function or file get before splitting?", "Functions stay small and single-purpose; a file that grows past a few hundred lines is split by responsibility."),
    ("Why prefer immutable updates?", "Returning new values instead of mutating in place avoids hidden side effects and makes state easier to reason about."),
    ("How are feature flags used?", "New behavior ships behind a flag defaulted off, enabling gradual rollout and instant disable without a redeploy."),
]


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "checkpoints", "pairs.jsonl")
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for q, c in PAIRS:
            f.write(json.dumps({"anchor": q, "positive": c, "source": "synthetic"}) + "\n")
            n += 1
    print(f"appended {n} synthetic pairs -> {out}")


if __name__ == "__main__":
    main()
