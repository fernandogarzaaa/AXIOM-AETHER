# Cognitive Protocol, version 1 (CP/1)

CP/1 is the stable wire contract shared by the three repositories that together
form one cognitive organism:

| Repository       | Role in the organism                                             |
| ---------------- | ---------------------------------------------------------------- |
| **AXIOM-AETHER** | Context engineering, compression, grounding, fast-weight adaptation, provenance, epistemic validation, MCP infrastructure. **Normative owner of CP/1.** |
| **EVE**          | Environments, simulations, deterministic replay, evaluation, fitness measurement, benchmark scenarios. |
| **ADAM**         | Genome, identity, beliefs, long-term memory, skills, evolution, reflection, self-model. |

This document is the single normative source. `schema/` holds machine-readable
JSON Schemas; `fixtures/` holds golden documents that every implementation must
round-trip byte-identically. `MANIFEST.sha256` pins the exact bytes of both so a
vendored copy can prove it has not drifted.

## 1. Why a protocol at all

The three repositories are written in Rust, TypeScript and Python, live in
separate version-control histories, and are released on independent cadences.
Any design in which one imports another's code creates a build-time dependency
edge that cannot survive that reality: ADAM cannot link a TypeScript simulation
engine, and EVE cannot link a `candle`-backed inference crate.

CP/1 replaces the dependency edge with a data edge. Each repository owns a
**vendored binding** — a hand-written implementation of these schemas in its own
language — plus a conformance test that reads the *shared* fixtures. Nothing is
generated at build time, nothing is fetched at build time, and no repository
needs the others present to compile or test. Drift is caught by the conformance
test, not by a linker.

### 1.1 Rationale: why not a shared package?

Rejected alternatives, and why:

- **A fourth "common types" repository consumed as a dependency.** Reintroduces
  the build edge in a worse form: a release of the common repo would have to be
  coordinated across three package ecosystems (crates.io, npm, PyPI) before any
  consumer could move. The organism's evolution rate would be capped by its
  slowest packaging pipeline.
- **Code generation from the schemas at build time.** Requires a generator
  toolchain (and its transitive dependencies) in all three CI environments, and
  produces types that are idiomatic in none of the three languages. The types
  here are small and stable enough that hand-written bindings are cheaper to
  read, cheaper to review, and free at build time.
- **gRPC/protobuf.** Solves a problem CP/1 does not have (high-frequency binary
  RPC) at the cost of one it does have (human-auditable, hash-stable documents
  that can be committed to a provenance log and diffed in review).

The chosen design costs one thing: three hand-written bindings must be kept in
sync. `MANIFEST.sha256` plus per-repository conformance tests make that cost
visible and enforced rather than silent.

## 2. Encoding and canonical form

CP/1 documents are JSON. Every document has exactly one **canonical form**, used
for hashing and signing:

1. Object members are sorted by key, ascending, comparing **UTF-8 bytes**.
   For valid UTF-8 this is identical to Unicode code-point order. It is *not*
   identical to UTF-16 code-unit order, which is what a bare JavaScript
   `Array.prototype.sort()` gives: beyond the BMP the two disagree, because a
   leading surrogate (`D800`–`DBFF`) sorts below `FFFD` in UTF-16 while the
   corresponding UTF-8 lead byte (`F0`) sorts above `EF`. Rust's `str: Ord` and
   Python's `sorted` already produce the required order; a JavaScript binding
   must compare UTF-8 bytes explicitly. The corpus carries a `Genome` whose
   `preferences` include `U+FFFD` and `U+1D11E` so any binding that gets this
   wrong fails the round-trip check rather than shipping divergent hashes.
2. No insignificant whitespace: no spaces after `:` or `,`, no newlines.
3. Strings use the shortest valid escaping: only `"`, `\`, and `U+0000`–`U+001F`
   are escaped, the latter as `\u00XX` except for the standard short forms
   `\b \f \n \r \t`.
4. Arrays preserve order. Order is semantic everywhere in CP/1.
5. Numbers are **integers only** (see §2.1). They are rendered with no sign for
   non-negative values, no leading zeros, no exponent, and no fraction. Every
   integer member is additionally bounded by the schema at 2^53−1
   (`9007199254740991`) or lower. JavaScript numbers are IEEE-754 doubles, so an
   unbounded integer is one a Rust binding can emit losslessly and a JavaScript
   binding silently rounds — and a rounded value re-serializes to different bytes
   and a different hash.
6. `null` is never written. An absent value is an absent key.

Canonical form is what `content_hash` commits to and what a signature covers.

### 2.1 Rationale: no floating point on the wire

Every quantity that would naturally be a fraction — confidence, fitness, risk,
similarity, decay — is transmitted as an integer in **basis points**: an
`integer` in `[0, 10000]` where `10000` means 1.0.

This is not stylistic. A protocol shared by Rust (`f32` in ADAM's memory
records, `f64` elsewhere) and JavaScript (IEEE-754 double, the only numeric
type) has no interoperable canonical rendering for floats. `0.1 + 0.2`,
shortest-round-trip formatting, `-0.0`, and `f32`→`f64` widening all produce
values that compare equal in one language and hash differently in another. A
protocol whose hashes depend on the reader's numeric type cannot support
reproducible provenance, which is the entire point of §4.

Basis points make the wire exact and force every implementation to be explicit
about where it rounds. Bindings convert at the boundary and only at the
boundary; internal representations stay whatever each language prefers.

Rounding on encode is **round-half-away-from-zero**, so `0.00005` → `1` bp in
every binding. Values outside `[0, 1]` are clamped, not wrapped.

## 3. Canonical types

Twelve types are canonical. Every concept in the organism is expressed in terms
of them, and each has exactly one owning repository — the only component allowed
to author (as opposed to read) documents of that type.

| Type            | Owner | Meaning |
| --------------- | ----- | ------- |
| `Identity`      | ADAM  | Who the organism is. Stable across model replacement. |
| `Genome`        | ADAM  | The full evolvable state: identity, values, goals, capabilities, skills, preferences, policies. |
| `Capability`    | ADAM  | A declared ability, and the provider that satisfies it. |
| `Belief`        | ADAM  | A proposition the organism holds, with confidence and evidence. |
| `Memory`        | ADAM  | A durable, provenanced record consolidated from experience. |
| `Skill`         | ADAM  | A named, testable procedure with a fitness history. |
| `Mutation`      | ADAM  | A proposed change to genome, skills or beliefs. |
| `Reflection`    | ADAM  | A point-in-time self-assessment across subsystems. |
| `Observation`   | EVE   | One perceived fact about an environment, at a point in time. |
| `Experience`    | EVE   | An observation situated in intent: goal, action, outcome, affect. |
| `FitnessResult` | EVE   | Evidence-backed measurement of a mutation against scenarios. |
| `Context`       | AXIOM | A bounded, compressed, grounded working set handed to a model. |

Ownership is exclusive for **authorship**, not for reading. ADAM reads
`FitnessResult` to gate a mutation; EVE reads `Mutation` to know what to
measure; AXIOM reads `Memory` to ground a `Context`. None of them may mint a
document of a type they do not own — a rule the conformance suite cannot check,
but code review and the `provenance.authored_by` field make auditable.

### 3.0 Protocol messages, which are not canonical types

Two shapes travel the wire without being canonical types, because they describe
an *exchange* rather than a fact about the organism:

| Message             | Owner | Meaning |
| ------------------- | ----- | ------- |
| `ValidationRequest` | ADAM  | Asks EVE to measure a mutation (§7). |
| `SignedEnvelope`    | —     | Transport wrapper (§6). Carries no provenance of its own. |

`ValidationRequest` is schema'd, sealed, provenance-bearing and covered by the
conformance corpus exactly like a canonical type — the distinction is that it is
not a durable record of what the organism is or has experienced, so it never
appears as an event `subject_type`, and it is not one of the twelve. Adding a
protocol message is backward-compatible in the sense of §8; adding a canonical
type is too, but the two lists are versioned separately.

### 3.1 Rationale: why these twelve, and why ownership is exclusive

The pre-CP/1 state of the three repositories contained, by actual count:

- **Three unrelated memory implementations.** ADAM's SQLite store with
  embeddings and decay, EVE's four-subsystem human memory model (working,
  episodic, semantic, spatial), and AXIOM's append-only JSONL scope store —
  plus four more partial stores in AXIOM alone (`heal_memory`, `patch_memory`,
  `vibe_memory`, `memory_recall`). Each used the word "memory" for a different
  thing.
- **Two unrelated belief systems.** ADAM's evidence-and-confidence
  `BeliefRegistry` and AXIOM's `BetaBelief` (Beta-distributed with
  Dempster–Shafer combination). Both are correct; they model different things
  under one name.
- **A component named for an integration that did not exist.** ADAM's
  `adam-eve` crate scored mutations by calling a caller-supplied Rust closure.
  It shared nothing with the EVE repository but the letters. The developmental
  lifecycle's "validate inside EVE" step therefore had no implementation at all.

CP/1 does not merge those implementations — merging them would destroy real,
justified differences. It assigns each concept one owner and makes the others
express their version *as* the canonical type at the boundary. AXIOM's
`BetaBelief` remains a Beta distribution internally; when it crosses the wire it
becomes a `Belief` whose `confidence_bp` is the distribution mean and whose
`uncertainty_bp` carries the variance that a scalar would have discarded. That
is a lossless-enough projection with a named owner, which is what the system
lacked.

### 3.2 Common members

Every canonical document carries:

- `cp` — always `"cp1"`. Present so a reader can reject a foreign document
  before parsing anything else.
- `type` — the canonical type name, exactly as spelled in the table above.
- `id` — a UUID, lowercase, hyphenated.
- `provenance` — see §4.

Timestamps are RFC 3339 with a `Z` offset and exactly millisecond precision
(`1970-01-01T00:00:00.000Z`). Fixed precision is required: a hash over a
timestamp that one binding renders with microseconds and another with seconds is
not reproducible.

## 4. Provenance

`Provenance` is the organism's chain of custody. It is mandatory on every
document, and it is what makes evolution auditable rather than merely logged.

```
{
  "authored_by": "adam",              // "adam" | "eve" | "axiom"
  "produced_at": "...Z",              // RFC 3339, ms precision
  "origin": "reflection:consolidation",
  "evidence": ["<uri or quoted excerpt>", ...],
  "derived_from": ["<uuid>", ...],    // ids of documents this was computed from
  "content_hash": "<64 lowercase hex>"
}
```

`content_hash` is SHA-256 over the canonical form (§2) of the document with the
`provenance.content_hash` member **removed** — a document cannot commit to its
own hash. `derived_from` makes the organism's knowledge a DAG: any belief can be
walked back through the memories that formed it, to the experiences those were
consolidated from, to the observations that produced them.

A `SignedEnvelope` (§6) may additionally carry an HMAC-SHA256 over the
`content_hash`, keyed by a shared fleet secret, for transport across a trust
boundary. AXIOM's existing `provenance` module implements exactly this
construction and is the reference implementation.

### 4.1 Rationale: hash excludes the hash, includes everything else

An alternative is to hash only a document's "semantic" fields and exclude
timestamps and provenance. That makes hashes stable under re-emission, which
sounds desirable and is in fact the bug: two documents produced by different
components from different evidence at different times would collide. Since
`derived_from` and `evidence` are precisely what distinguishes a well-grounded
belief from a fabricated one, excluding them from the hash would make the
provenance chain unforgeable in the wrong direction — anyone could substitute
evidence without changing the hash.

### 4.2 Events are addressable in `derived_from`

An entry in `derived_from` is the `id` of any CP/1 document — a canonical type
**or an event**. Events carry an `id` of the same shape (§5) and are documents
in their own right, so no schema distinction is needed; what follows is the
semantics, which a schema cannot express.

Referencing an event says *this document was computed from what that event
announced*. That is the only way to chain a conclusion back to the run that
produced it, because the run itself is not a canonical type — it is an event.

**Run parity.** `baseline.runs` MUST equal `candidate.runs`. This is not a new
constraint so much as a restatement of what §7.1 already requires: a
counterfactual is valid only because baseline and candidate differ by nothing
but the mutation, and a different number of runs is itself a difference. A
`FitnessResult` is therefore either fully measured (`runs > 0` on both sides,
equal) or fully declined (`runs = 0` on both sides) — there is no state in
between where "measured" is a matter of degree on one side and not the other.

**Required edge.** A `FitnessResult` whose `baseline.runs` is greater than zero
MUST list, in its `derived_from`, the id of the `SimulationCompleted` event
whose runs it summarizes, alongside the id of the `Mutation` it scores. Because
of run parity, this condition is equivalently `candidate.runs > 0`; nothing can
satisfy one side without the other.

The edge is more than a reference to the right document type — it is a claim
that can be checked. A conforming binding MUST verify:

- The referenced document's `subject_id` equals the `FitnessResult`'s
  `mutation_id`. A `SimulationCompleted` for a different mutation is a valid
  event that says nothing about this result.
- The referenced document's `payload.baseline_runs` and `payload.candidate_runs`
  equal the `FitnessResult`'s `baseline.runs` and `candidate.runs`. Otherwise a
  result claiming 90 runs could cite a real event that ran once.

Without these checks the edge is a reference in name only: any
`SimulationCompleted` for any mutation would satisfy it, which is not
meaningfully different from requiring no edge at all. A fabricated result and a
measured one are structurally identical unless the edge is bound to the
specific claim it is supposed to back — this is the one place in the protocol
where a component reports on work only it can see, which is exactly where the
chain has to be checkable rather than merely conventional.

The condition is `runs > 0` rather than unconditional because a result that
reports no runs is the honest encoding of *EVE declined to measure this* (§3,
`FitnessResult`; and `runs` in the schema). There is no simulation for it to
name, and requiring one would force it to invent the very reference this rule
exists to make meaningful. A result claiming runs it cannot chain, and a result
claiming none, are the two consistent states; the gap between them is what check
5 closes.

Conformance check 5 (§9) enforces all of this over the corpus: it resolves
every `derived_from` id against the documents present, rejects unequal
`baseline.runs`/`candidate.runs`, and fails a `FitnessResult` that reports runs
without naming a matching `SimulationCompleted`.

## 5. Events

Everything important emits an event. Events are the organism's nervous system:
subsystems announce facts and never call each other directly.

Event `type` values are exactly:

| Event                | Emitter | Meaning |
| -------------------- | ------- | ------- |
| `ObservationRecorded`| EVE     | An environment was perceived. |
| `ExperienceCreated`  | EVE     | An observation was situated in goal + action + outcome. |
| `ContextCompressed`  | AXIOM   | A working set was compressed; carries before/after token counts. |
| `GroundingFailed`    | AXIOM   | A claim could not be supported by supplied evidence. |
| `MemoryConsolidated` | ADAM    | Experiences were distilled into a durable memory. |
| `BeliefUpdated`      | ADAM    | A belief was formed, reinforced, weakened or retracted. |
| `SkillLearned`       | ADAM    | A skill reached the promoted stage. |
| `ReflectionCompleted`| ADAM    | A self-assessment across subsystems was produced. |
| `MutationProposed`   | ADAM    | A change to genome, skills or beliefs was proposed. |
| `SimulationCompleted`| EVE     | A deterministic scenario run finished. |
| `FitnessMeasured`    | EVE     | A mutation was scored against baseline and candidate runs. |
| `MutationAccepted`   | ADAM    | A proposal passed governance and was applied. |
| `MutationRejected`   | ADAM    | A proposal was refused; carries the reason. |
| `GenomeCommitted`    | ADAM    | A new immutable genome version was appended. |

The list is closed. Adding an event is a CP/1 version change, because consumers
switch exhaustively on it — an open enumeration would make every consumer's
handling of unknown events an untested code path.

Envelope shape:

```
{
  "cp": "cp1",
  "type": "MutationAccepted",
  "id": "<uuid>",                  // this event's identity
  "occurred_at": "...Z",
  "actor": "adam",
  "subject_id": "<uuid>",          // the canonical document this is about
  "subject_type": "Mutation",
  "correlation_id": "<uuid>",      // the lifecycle turn this belongs to
  "causation_id": "<uuid>",        // the event that caused this one, if any
  "payload": { ... },              // event-specific, schema'd per event
  "provenance": { ... }
}
```

`correlation_id` is what makes a full developmental turn (§7) reconstructible:
every event emitted while processing one observation shares it. `causation_id`
gives the finer-grained edge — which specific event triggered this one — so the
turn is a tree, not just a bag.

## 6. Transport

CP/1 is transport-agnostic. Two transports are normative because the organism
uses both:

**Line-delimited JSON over stdio.** One canonical-form document per line, `\n`
terminated, UTF-8, no BOM. This is how ADAM invokes EVE for fitness measurement:
ADAM spawns EVE's validator as a subprocess, writes one `ValidationRequest`
line, and reads `SignedEnvelope` lines until the response. Chosen because it
needs no network listener, no port allocation, no service discovery and no
authentication story for the common single-host case — and because a subprocess
boundary is a real isolation boundary for a component whose job is to run
untrusted scenarios.

**HTTP POST of a `SignedEnvelope`.** For the distributed case, where EVE runs as
a service. Over HTTP the HMAC is **required**: the request crosses a trust
boundary the receiver does not control, and an unauthenticated envelope proves
only that its payload is internally consistent — anyone can produce that. A
receiver reachable over HTTP MUST refuse an envelope with no `hmac`, and MUST
refuse one whose `hmac` does not verify, before parsing the payload.

Over stdio the HMAC is optional, because the parent process spawned the child
and controls its argv, environment and file descriptors; requiring a shared
secret there is ceremony without a threat.

The HMAC is HMAC-SHA256 (RFC 2104) computed over the **64 lowercase-hex ASCII
characters of the `sha256` member**, not over the payload bytes and not over the
raw digest — so a receiver verifies it without re-hashing the payload. The key
is the fleet secret as raw bytes; a secret configured as text is UTF-8 encoded
with no trailing newline. Comparison MUST be constant-time.

A `SignedEnvelope` wraps any CP/1 document:

```
{
  "cp": "cp1",
  "schema": "cp1_signed_envelope",
  "payload": "<canonical-form JSON of the document, as a string>",
  "sha256": "<64 lowercase hex of payload bytes>",
  "hmac": "<64 lowercase hex; REQUIRED over HTTP, optional only over trusted stdio>"
}
```

The payload is a *string*, not a nested object, so the bytes that were hashed
are the exact bytes transmitted. Re-serializing a nested object would let a
receiver's JSON writer alter the bytes the hash committed to.

## 7. The developmental lifecycle

CP/1 exists to make one loop executable across three repositories. Each arrow is
a document crossing a boundary; each stage emits its event.

```
              EVE                        ADAM                       AXIOM
               │                          │                           │
   Observe ────┤ Observation              │                           │
               │  ObservationRecorded     │                           │
   Experience ─┤ Experience ─────────────►│                           │
               │  ExperienceCreated       │                           │
               │                          ├── Reflect                 │
               │                          │    ReflectionCompleted    │
               │                          ├── Consolidate ──► Memory  │
               │                          │    MemoryConsolidated     │
               │                          ├── Update beliefs          │
               │                          │    BeliefUpdated          │
               │                          ├── Generate mutations      │
               │                          │    MutationProposed       │
               │◄─ Mutation ──────────────┤                           │
   Validate ───┤ (deterministic scenarios)│                           │
               │  SimulationCompleted     │                           │
   Measure ────┤ FitnessResult ──────────►│                           │
               │  FitnessMeasured         │                           │
               │                          ├── Govern: accept / reject │
               │                          │    MutationAccepted|Rejected
               │                          ├── Commit genome           │
               │                          │    GenomeCommitted        │
               │                          │                           │
               │                          └── Context request ───────►│
               │                                       Context ◄──────┤
               │                                        ContextCompressed
```

There is no shortcut edge. In particular a mutation that touches the genome
beyond low-stakes preferences **cannot** be accepted without a `FitnessResult`
whose `recommendation` is `approve` and whose provenance chains back to a real
simulation run. ADAM enforces this; the enforcement is meaningless unless the
`FitnessResult` was authored by EVE, which is why `provenance.authored_by` is
mandatory and why ownership (§3) is exclusive.

### 7.1 Rationale: fitness is counterfactual, not absolute

A `FitnessResult` reports two measurements — `baseline` (the organism as it is)
and `candidate` (the organism with the mutation applied) — over the *same*
scenario set with the *same* seed, plus their delta. An absolute score would be
uninterpretable: "this mutation scored 72" says nothing without knowing what the
organism scored before.

Determinism is what makes the comparison valid. EVE's simulation is seeded
(`seed` is a required member of `ValidationRequest`), so baseline and candidate
differ only by the mutation. A `FitnessResult` whose two runs used different
seeds is malformed, and bindings must reject it.

## 8. Versioning

`cp` is `"cp1"` for the lifetime of this major version. Within it:

- Adding an **optional** member to an existing type is backward-compatible.
- Adding a new canonical type is backward-compatible.
- Adding an event, removing anything, renaming anything, changing a member's
  type, or changing the canonical-form rules is **not**, and requires `cp2`.

`VERSION` holds the exact revision (e.g. `1.0.0`) for diagnostics.
`MANIFEST.sha256` pins every schema and fixture file. A vendored binding's
conformance test verifies its copy against the manifest, so a change to the
normative source that has not been propagated fails CI in the repositories that
have not yet caught up — loudly, and before anything ships.

## 9. Conformance

Every binding, in every repository, runs the same checks against the same
vendored corpus. That is the whole mechanism keeping three hand-written
bindings in three languages agreeing about the wire — there is no code
generation and no shared library, so this suite is load-bearing.

| # | Check | The failure it catches |
| - | ----- | ---------------------- |
| 1 | **Round trip.** Parsing a fixture and re-encoding it in canonical form reproduces the exact bytes. | A binding whose key ordering, number rendering or string escaping differs from the normative source — the class of bug that silently produces documents other components reject. |
| 2 | **Seal.** Each fixture's `provenance.content_hash` is the true hash of the document with that member removed. | A binding that hashes a different byte sequence than it transmits. |
| 3 | **Structure.** Required members are present with the right shapes, and basis-point members are integers in range. | A binding that would accept a float where the protocol forbids one, or a magnitude where only a delta may be signed. |
| 4 | **Manifest.** The vendored copy hashes to what the normative source recorded. | A binding running against a stale corpus, which would make checks 1–3 pass against the wrong contract. |
| 5 | **Provenance edges.** `derived_from` ids resolve within the corpus; `baseline.runs` equals `candidate.runs`; and a measured `FitnessResult` references a `SimulationCompleted` whose `subject_id` and reported run counts match (§4.2). | A measurement that cannot be chained back to the specific run that produced it — including one that cites a real run for the wrong mutation, or the wrong count — which is indistinguishable from a fabricated one. |

Check 5 is the only one that reads across documents rather than within one. It
is deliberately scoped to the corpus: a binding cannot resolve an id it has
never been given, so the check enforces the edges among documents actually
present and says nothing about ids that point outside the set.
