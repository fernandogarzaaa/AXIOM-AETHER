#!/usr/bin/env python3
"""score.py — pure-local BLEU-4 and Rouge-L scoring for the Axiom TTT benchmark.

Reads a JSON array of {"hypothesis": str, "reference": str} pairs on stdin (or
from a file passed as argv[1]) and prints a JSON object:

    {"bleu4": <float>, "rougeL": <float>, "n": <int>}

BLEU-4 is corpus-level BLEU with the standard 4-gram precision, computed with
sacrebleu and exponential smoothing so a corpus with sparse 4-gram matches still
yields a non-zero, monotonic signal (rather than collapsing to 0.0 the moment a
single n-gram order misses). Rouge-L is the mean sentence-level longest-common-
subsequence F-measure from rouge-score. No network or external API is used.
"""

import json
import sys

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer


def score_pairs(pairs):
    """Compute corpus BLEU-4 and mean Rouge-L F over (hypothesis, reference) pairs.

    Returns {"bleu4": float, "rougeL": float, "n": int}. Both scores are on a
    0–100 scale. BLEU uses exponential smoothing so sparse 4-gram matches still
    produce a continuous, monotonic signal instead of collapsing to zero.
    """
    hyps = [str(p.get("hypothesis", "")) for p in pairs]
    refs = [str(p.get("reference", "")) for p in pairs]

    bleu = BLEU(smooth_method="exp", effective_order=True)
    bleu_score = bleu.corpus_score(hyps, [refs]).score if hyps else 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_vals = [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
    rouge_l = 100.0 * (sum(rouge_vals) / len(rouge_vals)) if rouge_vals else 0.0

    return {
        "bleu4": round(bleu_score, 4),
        "rougeL": round(rouge_l, 4),
        "n": len(pairs),
    }


def load_pairs():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.load(sys.stdin)


def main():
    print(json.dumps(score_pairs(load_pairs())))


if __name__ == "__main__":
    main()
