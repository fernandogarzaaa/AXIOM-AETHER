"""Monte-Carlo: Prolonged-Session Stack (PSS) in SUBSCRIPTION QUOTA UNITS.

Unit = "quota units" normalized so 1 unit == 1 Sonnet-5 uncached input token.
Anthropic usage windows weight tokens ~ like price, so we reuse price ratios
as quota weights. Baseline = CVM v1 shipped (v0.4.0). PSS levers measured on top.

Load-bearing question: R1 model routing's DUAL-CACHE economics. Haiku keeps its
OWN prompt cache; alternating Haiku/top-tier => repeated writes on both sides
=> possibly net-negative. Router uses hysteresis (run-length gating).
"""

import random
import statistics as st


def w(per_mtok):
    return per_mtok / 2.00


S_IN, S_W5, S_W1H, S_RD, S_OUT = w(2.00), w(2.50), w(4.00), w(0.20), w(10.00)
H_IN, H_W5, H_RD, H_OUT = w(1.00), w(1.25), w(0.10), w(5.00)
O_IN, O_W5, O_RD, O_OUT = w(5.00), w(6.25), w(0.50), w(25.00)
# Fable 5 (corrected from Anthropic pricing docs, 2026-07): 10/12.50/20/1/50
F_IN, F_W5, F_W1H, F_RD, F_OUT = w(10.00), w(12.50), w(20.00), w(1.00), w(50.00)

PREFIX = 80_000
OUT_T_HARD = 1_500
OUT_T_MECH = 400
USER_T = 250
LIGHT_TOOL_T = 800
HEAVY_TOOL_T = 16_000
HEAVY_RATE = 0.45
DIGEST_KEEP = 0.15
COMPACT_AT = 180_000
SUMMARY_T = 5_000
TTL_S = 300
GAP_LONG_P = 0.15
MECH_RATE = 0.55
MECH_RUN_MEAN = 4.0
COOLDOWN = 3


def turns_in_session(rng):
    return int(rng.triangular(100, 300, 160))


def new_session_turn_types(rng, n):
    types, heavy, err = [], [], []
    i = 0
    while i < n:
        if rng.random() < MECH_RATE:
            run = max(1, int(rng.expovariate(1.0 / MECH_RUN_MEAN)))
            for _ in range(run):
                if i >= n:
                    break
                types.append('mech'); i += 1
        else:
            types.append('hard'); i += 1
    for t in types:
        heavy.append(rng.random() < HEAVY_RATE)
        err.append(rng.random() < (0.06 if t == 'mech' else 0.02))
    return types, heavy, err


def cache_cost(ph, new_in, warm, wr_write, wr_read, wr_in):
    return ph * (wr_read if warm else wr_write) + new_in * wr_in


def simulate_session(rng, levers):
    n = turns_in_session(rng)
    types, heavy, err = new_session_turn_types(rng, n)
    T_IN, T_W5, T_W1H, T_RD, T_OUT = levers['top']

    quota = 0.0
    transcript = 0.0
    top_warm = False
    haiku_warm = False
    ttl_budget = TTL_S
    cooldown = 0
    routed_turns = 0
    local_turns = 0
    use_1h = levers['R3'] and (rng.random() < 0.5)

    # L-A tool elision: shrink the static prefix at its root (tool schemas).
    # ~70% of the 80K prefix is tool/MCP defs; keep the recently-used subset
    # (~2-3 tools) + a small "load more" affordance. Deterministic per session
    # => cache-safe. Conservative: keep 30% of prefix.
    prefix = PREFIX * (0.30 if levers.get('LA') else 1.0)

    for i in range(n):
        broke = False
        if transcript + prefix > COMPACT_AT:
            transcript = SUMMARY_T
            top_warm = haiku_warm = False
            broke = True
        if (rng.random() < GAP_LONG_P) and not use_1h:
            top_warm = haiku_warm = False
            broke = True
        if levers['R2'] and broke:
            transcript *= 0.55

        ph = prefix + transcript
        new_in = USER_T + (HEAVY_TOOL_T if heavy[i] else LIGHT_TOOL_T)

        # L-B local short-circuit: confidently-trivial turns (a subset of
        # mechanical turns with NO heavy tool result and no error) are answered
        # by the local TTT engine -> zero API quota, turn leaves the ledger.
        # Conservative: only ~40% of light mechanical turns are safely trivial.
        if (levers.get('LB') and cooldown == 0 and types[i] == 'mech'
                and not heavy[i] and not err[i] and rng.random() < 0.40):
            local_turns += 1
            # no upstream call at all; but the NEXT real turn must re-read the
            # transcript grown by this locally-answered exchange
            transcript += USER_T + OUT_T_MECH
            continue

        route = False
        if levers['R1'] and cooldown == 0 and types[i] == 'mech':
            route = True
        if types[i] == 'hard' or err[i]:
            cooldown = COOLDOWN
        elif cooldown > 0:
            cooldown -= 1

        top_write = T_W1H if use_1h else T_W5

        if route:
            routed_turns += 1
            q = cache_cost(ph, new_in, haiku_warm, H_W5, H_RD, H_IN)
            q += OUT_T_MECH * H_OUT
            haiku_warm = True
            ttl_budget -= 1
            if ttl_budget <= 0:
                top_warm = False
                ttl_budget = TTL_S
        else:
            q = cache_cost(ph, new_in, top_warm, top_write, T_RD, T_IN)
            q += (OUT_T_HARD if types[i] == 'hard' else OUT_T_MECH) * T_OUT
            top_warm = True
            haiku_warm = False

        quota += q

        transcript += (HEAVY_TOOL_T * DIGEST_KEEP) if heavy[i] else LIGHT_TOOL_T
        transcript += USER_T + (OUT_T_MECH if route or types[i] == 'mech' else OUT_T_HARD)

    return quota, routed_turns, local_turns, n


def run(levers, tier, sessions=4000, seed=0):
    rng = random.Random(seed)
    levers = dict(levers); levers['top'] = tier
    qs, routed, local, turns = [], [], [], []
    for _ in range(sessions):
        q, r, lo, n = simulate_session(rng, levers)
        qs.append(q); routed.append(r); local.append(lo); turns.append(n)
    return st.mean(qs), st.mean(routed), st.mean(local), st.mean(turns)


if __name__ == "__main__":
    TIERS = {
        'Sonnet-5': (S_IN, S_W5, S_W1H, S_RD, S_OUT),
        'Opus-4.8': (O_IN, O_W5, O_W5, O_RD, O_OUT),
        'Fable-5':  (F_IN, F_W5, F_W1H, F_RD, F_OUT),
    }
    for tname, tier in TIERS.items():
        base_q, _, _, tn = run(dict(R1=False, R2=False, R3=False), tier)
        print(f"\n=== top tier: {tname}  (mean {tn:.0f} turns/session) ===")
        print(f"  baseline (CVM v1 shipped)          : {base_q:12.1f} quota units")
        for label, lv in [
            ("R1+R2+R3 (orig PSS)",      dict(R1=True, R2=True, R3=True)),
            ("+ L-A tool elision",       dict(R1=True, R2=True, R3=True, LA=True)),
            ("+ L-B local short-circuit",dict(R1=True, R2=True, R3=True, LB=True)),
            ("+ L-A + L-B (PSS v2)",     dict(R1=True, R2=True, R3=True, LA=True, LB=True)),
            ("PSS v2 no R1 (LA+LB+R2+R3)",dict(R1=False,R2=True, R3=True, LA=True, LB=True)),
        ]:
            q, r, lo, _ = run(lv, tier)
            red = 100 * (base_q - q) / base_q
            extra = f"  (routed {r:.0f} local {lo:.0f} /{tn:.0f})"
            print(f"  {label:30s} : {q:12.1f}  {red:+6.1f}%{extra}")
