//! Agentic self-evaluation harness — capability as a *measured number*.
//!
//! The point of this whole pillar is that AXIOM's coding ability should be
//! reported, not asserted. [`run_eval`] materializes a set of seeded broken-repo
//! fixtures, runs the real autonomous [`crate::solve`] loop on each (localize →
//! deterministic mutation repair → verify, all reversible), and reports the
//! fraction it drives green.
//!
//! The built-in fixtures ([`builtin_cases`]) are deliberately **deterministic
//! and offline**: each failure is one the mutation layer can repair without any
//! model or network, so the score is reproducible in CI and means exactly what
//! it says — "the autonomous loop fixed N of M broken repos end-to-end."

use std::path::{Path, PathBuf};

use crate::agentic::{
    agentic_loop, agentic_loop_with_holdout, AttemptMemory, EditSet, ProposeContext, Proposer,
};
use crate::inference::InferencePipeline;
use crate::solve::{posix_shell, run_verify, run_verify_capture, solve, SolveOptions};

/// A seeded broken project: files to materialize and the verify command that
/// must be driven to green. The fix is reachable by the deterministic repair
/// layer, so no LLM is required.
#[derive(Debug, Clone)]
pub struct EvalCase {
    pub name: String,
    /// (relative path, contents) written under a fresh temp project root.
    pub files: Vec<(String, String)>,
    /// Verify program + args, run with the project root as working dir.
    pub command: String,
    pub args: Vec<String>,
    /// When non-empty, this case exercises the **agentic multi-file** path: a
    /// deterministic scripted proposer applies these (relative path, contents)
    /// as one atomic [`EditSet`] through [`agentic_loop`]. Used for fixtures
    /// where no single-file repair passes the verifier, so the single-file
    /// `solve` path cannot solve them — only a coordinated multi-file edit can.
    pub agentic_fix: Vec<(String, String)>,
    /// Optional held-out verifier (program, args). When set on an agentic
    /// case, a candidate edit-set is committed only if it passes BOTH the
    /// train `command` and this held-out command
    /// ([`crate::agentic::agentic_loop_with_holdout`]) — the anti-overfitting
    /// split: a fix shaped against the train verifier that fails held-out
    /// checks is rolled back instead of trusted.
    pub holdout: Option<(String, Vec<String>)>,
    /// Programs that must be on `PATH` for this case to run (`python3`, `go`,
    /// …). A case whose requirement is missing is reported as **skipped** and
    /// excluded from the score rather than counted as a failure, so the suite
    /// stays honest on a machine without that toolchain; `--require-all`
    /// turns a skip into a failure for CI.
    pub requires: Vec<String>,
    /// Files that must be byte-identical after the solve for the case to count
    /// as solved. Used on every module+test fixture to pin the test file: a
    /// "repair" that edits the oracle instead of the code under test is an
    /// overfit and is scored as a failure, however green the verifier is.
    pub expect_unchanged: Vec<String>,
}

impl EvalCase {
    fn new(name: &str, files: &[(&str, &str)], command: &str, args: &[&str]) -> Self {
        EvalCase {
            name: name.to_string(),
            files: files
                .iter()
                .map(|(p, c)| (p.to_string(), c.to_string()))
                .collect(),
            command: command.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            agentic_fix: Vec::new(),
            holdout: None,
            requires: Vec::new(),
            expect_unchanged: Vec::new(),
        }
    }

    /// Require these programs on `PATH` (skip the case otherwise).
    fn requires(mut self, programs: &[&str]) -> Self {
        self.requires = programs.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Pin these files: they must be byte-identical after the solve.
    fn unchanged(mut self, files: &[&str]) -> Self {
        self.expect_unchanged = files.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Attach a held-out verifier: commits require passing both the train
    /// command and this one. Meaningful only on agentic cases.
    fn with_holdout(mut self, command: &str, args: &[&str]) -> Self {
        self.holdout = Some((
            command.to_string(),
            args.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    /// A case solved via the agentic multi-file transaction path: `fix` is the
    /// coordinated edit-set a deterministic proposer applies atomically.
    fn new_agentic(
        name: &str,
        files: &[(&str, &str)],
        command: &str,
        args: &[&str],
        fix: &[(&str, &str)],
    ) -> Self {
        let mut c = Self::new(name, files, command, args);
        c.agentic_fix = fix
            .iter()
            .map(|(p, c)| (p.to_string(), c.to_string()))
            .collect();
        c
    }
}

/// A deterministic [`Proposer`] (no LLM) that yields one predefined edit-set on
/// its first call, then gives up. Lets the eval certify the agentic loop's
/// atomic multi-file apply/verify/commit machinery offline and reproducibly.
struct ScriptedProposer {
    once: Option<EditSet>,
}

impl Proposer for ScriptedProposer {
    fn propose(&mut self, _ctx: &ProposeContext) -> Option<EditSet> {
        self.once.take()
    }
}

/// Result of one case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseResult {
    pub name: String,
    pub solved: bool,
    /// `Some(reason)` when the case did not run (a required program is
    /// missing). Skipped cases are excluded from `total()` and the score.
    pub skipped: Option<String>,
}

/// Aggregate evaluation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalReport {
    pub results: Vec<CaseResult>,
}

impl EvalReport {
    /// Cases that ran (skipped cases are not counted).
    pub fn total(&self) -> usize {
        self.results.iter().filter(|r| r.skipped.is_none()).count()
    }
    pub fn solved(&self) -> usize {
        self.results.iter().filter(|r| r.solved).count()
    }
    /// Cases that did not run because a required program was missing.
    pub fn skipped(&self) -> usize {
        self.results.iter().filter(|r| r.skipped.is_some()).count()
    }
    /// Success rate in [0, 1]; an empty suite scores 0.
    pub fn score(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            self.solved() as f64 / self.total() as f64
        }
    }
    /// Human-readable one-liner per case plus the headline score.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        for r in &self.results {
            match &r.skipped {
                Some(reason) => s.push_str(&format!("  [SKIP] {} ({reason})\n", r.name)),
                None => s.push_str(&format!(
                    "  [{}] {}\n",
                    if r.solved { "PASS" } else { "FAIL" },
                    r.name
                )),
            }
        }
        s.push_str(&format!(
            "score: {}/{} = {:.0}%",
            self.solved(),
            self.total(),
            self.score() * 100.0
        ));
        if self.skipped() > 0 {
            s.push_str(&format!(" ({} skipped)", self.skipped()));
        }
        s
    }
}

/// The built-in deterministic fixture suite — offline, reproducible, no LLM.
///
/// Cases 1–12 are **localization fixtures**: a shell program with a real
/// one-token bug, checked by its behaviour, where the verify command reports
/// the failure in one trace dialect each (rustc, Python, JS, Go, C, TypeScript,
/// javac, Ruby, a nested path). They need nothing but a POSIX shell, so they
/// run everywhere. The check lives in the verify command itself, which cannot
/// be edited, and the fix comes from the mutation layer
/// (`crate::mutation_repair`). Cases 13–14 are scripted multi-file
/// transactions. Cases 15+ are **real-interpreter fixtures**: a genuinely
/// buggy module, a genuine test, and the actual toolchain as the verifier; the
/// test file is pinned so an overfit "fix" scores as a failure. Cases that
/// need a toolchain not on `PATH` are reported as skipped, never as failed.
pub fn builtin_cases() -> Vec<EvalCase> {
    vec![
        // 1. A program blamed by a frame that names it (`file:line:col`): the
        //    loop stops one short, so 1..5 sums to 10. The check is in the
        //    verify command, so there is no test file to edit.
        EvalCase::new(
            "sh-self-naming-frame",
            &[(
                "run.sh",
                "#!/bin/sh\ntotal=0\ni=1\nwhile [ \"$i\" -lt 5 ]; do\n  total=$((total + i))\n  i=$((i + 1))\ndone\necho \"$total\"\n",
            )],
            &posix_shell(),
            &[
                "-c",
                "out=$(sh run.sh); [ \"$out\" = 15 ] || \
                 { echo \"run.sh:4:1: expected 15 got $out\" >&2; exit 1; }",
            ],
        ),
        // 2. A wrong arithmetic operator: an area computed with `+`.
        EvalCase::new(
            "sh-wrong-arith",
            &[("area.sh", "#!/bin/sh\necho $(( $1 + $2 ))\n")],
            &posix_shell(),
            &[
                "-c",
                "a=$(sh area.sh 3 4); b=$(sh area.sh 5 1); \
                 { [ \"$a\" = 12 ] && [ \"$b\" = 5 ]; } || \
                 { echo \"area.sh:2:1: expected 12 and 5 got $a and $b\" >&2; exit 1; }",
            ],
        ),
        // 3. Multi-file: a correct helper is cited first, the failing file
        //    second; the loop must leave the helper alone and repair the
        //    right one (`- 2` where `+ 2` was meant).
        EvalCase::new(
            "multi-file-pick-failing",
            &[
                ("helper.sh", "double() {\n  echo $(( $1 * 2 ))\n}\n"),
                (
                    "main.sh",
                    "#!/bin/sh\n. ./helper.sh\necho $(( $(double 3) - 2 ))\n",
                ),
            ],
            &posix_shell(),
            &[
                "-c",
                "echo 'helper.sh:1:1: note: sourced by main.sh'; out=$(sh main.sh); \
                 [ \"$out\" = 8 ] || { echo \"main.sh:3:1: expected 8 got $out\" >&2; exit 1; }",
            ],
        )
        .unchanged(&["helper.sh"]),
        // 4. rustc-shaped diagnostic (`error[..]` then `  --> file:line:col`):
        //    the lower clamp tests `-gt 0` where `-lt 0` was meant.
        EvalCase::new(
            "rust-arrow-frame",
            &[(
                "clamp.sh",
                "clamp() {\n  if [ \"$1\" -gt 10 ]; then\n    echo 10\n  elif [ \"$1\" -gt 0 ]; then\n    echo 0\n  else\n    echo \"$1\"\n  fi\n}\n",
            )],
            &posix_shell(),
            &[
                "-c",
                ". ./clamp.sh; \
                 { [ \"$(clamp 5)\" = 5 ] && [ \"$(clamp 12)\" = 10 ] && [ \"$(clamp -3)\" = 0 ]; } || \
                 { echo 'error[E0308]: clamp returned the wrong bound'; echo '  --> clamp.sh:4:8'; exit 1; }",
            ],
        ),
        // 5. Python-traceback-shaped frame (`File \"x\", line N, in fn`):
        //    `-gt` where `-ge` was meant.
        EvalCase::new(
            "python-frame-localize",
            &[("limits.sh", "is_adult() {\n  [ \"$1\" -gt 18 ]\n}\n")],
            &posix_shell(),
            &[
                "-c",
                ". ./limits.sh; \
                 { is_adult 18 && is_adult 40 && ! is_adult 17; } || \
                 { echo 'Traceback (most recent call last):'; \
                 echo '  File \"limits.sh\", line 2, in is_adult'; exit 1; }",
            ],
        ),
        // 6. JS-stack-shaped frame (`at fn (file:line:col)`): the last index
        //    of n items computed as `n + 1`.
        EvalCase::new(
            "js-stack-localize",
            &[("index.sh", "last_index() {\n  echo $(( $1 + 1 ))\n}\n")],
            &posix_shell(),
            &[
                "-c",
                ". ./index.sh; \
                 { [ \"$(last_index 5)\" = 4 ] && [ \"$(last_index 1)\" = 0 ]; } || \
                 { echo 'AssertionError: wrong last index'; \
                 echo '    at last_index (index.sh:2:9)'; exit 1; }",
            ],
        ),
        // 7. Go-shaped diagnostic (`./file:line:col:`): a range check joined
        //    with `||` where `&&` was meant.
        EvalCase::new(
            "go-frame-localize",
            &[(
                "server.sh",
                "in_range() {\n  [ \"$1\" -ge 1 ] || [ \"$1\" -le 9 ]\n}\n",
            )],
            &posix_shell(),
            &[
                "-c",
                ". ./server.sh; \
                 { in_range 5 && ! in_range 0 && ! in_range 12; } || \
                 { echo './server.sh:2:18: in_range accepts a value outside 1..9'; exit 1; }",
            ],
        ),
        // 8. C-compiler-shaped diagnostic (`file:line:col: error:`): `-ne`
        //    where `-eq` was meant.
        EvalCase::new(
            "c-compiler-localize",
            &[("calc.sh", "is_zero() {\n  [ \"$1\" -ne 0 ]\n}\n")],
            &posix_shell(),
            &[
                "-c",
                ". ./calc.sh; \
                 { is_zero 0 && ! is_zero 3; } || \
                 { echo 'calc.sh:2:10: error: is_zero has the wrong sense'; exit 1; }",
            ],
        ),
        // 9. TypeScript-shaped diagnostic in the paren form
        //    (`file(line,col): error TSxxxx`): `-z` where `-n` was meant.
        EvalCase::new(
            "ts-frame-localize",
            &[("api.sh", "has_token() {\n  [ -z \"$1\" ]\n}\n")],
            &posix_shell(),
            &[
                "-c",
                ". ./api.sh; \
                 { has_token abc && ! has_token ''; } || \
                 { echo 'api.sh(2,5): error TS2322: has_token has the wrong sense'; exit 1; }",
            ],
        ),
        // 10. javac-shaped diagnostic (`File:line: error:`, no column): a
        //     string compared with `!=` where `=` was meant.
        EvalCase::new(
            "java-frame-localize",
            &[("Gate.sh", "is_admin() {\n  [ \"$1\" != admin ]\n}\n")],
            &posix_shell(),
            &[
                "-c",
                ". ./Gate.sh; \
                 { is_admin admin && ! is_admin guest; } || \
                 { echo 'Gate.sh:2: error: is_admin has the wrong sense'; exit 1; }",
            ],
        ),
        // 11. Ruby-shaped error (`file:line:in ... (RuntimeError)`): a half
        //     computed with `*`.
        EvalCase::new(
            "ruby-frame-localize",
            &[("task.sh", "half() {\n  echo $(( $1 * 2 ))\n}\n")],
            &posix_shell(),
            &[
                "-c",
                ". ./task.sh; \
                 { [ \"$(half 10)\" = 5 ] && [ \"$(half 4)\" = 2 ]; } || \
                 { echo \"task.sh:2:in 'half': wrong result (RuntimeError)\"; exit 1; }",
            ],
        ),
        // 12. Nested-path resolution (`src/util/file:line:col`): the fault is
        //     in a subdirectory, so the localizer must resolve a relative
        //     multi-segment path under the sandbox root. The next port is
        //     computed as `+ 2`.
        EvalCase::new(
            "nested-path-localize",
            &[(
                "src/util/ports.sh",
                "next_port() {\n  echo $(( $1 + 2 ))\n}\n",
            )],
            &posix_shell(),
            &[
                "-c",
                ". ./src/util/ports.sh; \
                 { [ \"$(next_port 8080)\" = 8081 ] && [ \"$(next_port 1)\" = 2 ]; } || \
                 { echo 'src/util/ports.sh:2:8: next_port skips a port'; exit 1; }",
            ],
        ),
        // 13. Coordinated multi-file: the verifier requires BOTH scripts to exit
        //    0, but each is independently broken. No single-file repair passes,
        //    so the single-file solve path cannot solve it — only the agentic
        //    loop's atomic multi-file transaction (fix both, verify, commit) can.
        EvalCase::new_agentic(
            "agentic-multi-file-both-broken",
            &[
                ("a.sh", "#!/bin/sh\nexit 1\n"),
                ("b.sh", "#!/bin/sh\nexit 1\n"),
            ],
            &posix_shell(),
            &["-c", "sh a.sh && sh b.sh"],
            // Coordinated fix: both files must go green together.
            &[
                ("a.sh", "#!/bin/sh\nexit 0\n"),
                ("b.sh", "#!/bin/sh\nexit 0\n"),
            ],
        ),
        // 14. Held-out verification split: the fix must pass the train verifier
        //    (a.sh) AND the held-out verifier (b.sh) before it is committed.
        //    A train-only fix (patching a.sh alone) would be the classic
        //    test-overfit patch — the holdout gate rejects it; the scripted
        //    fix repairs both, so the case solves only through the
        //    agentic_loop_with_holdout path.
        EvalCase::new_agentic(
            "agentic-holdout-split",
            &[
                ("a.sh", "#!/bin/sh\nexit 1\n"),
                ("b.sh", "#!/bin/sh\nexit 1\n"),
            ],
            &posix_shell(),
            &["-c", "sh a.sh"],
            &[
                ("a.sh", "#!/bin/sh\nexit 0\n"),
                ("b.sh", "#!/bin/sh\nexit 0\n"),
            ],
        )
        .with_holdout(&posix_shell(), &["-c", "sh b.sh"]),
        // ------------------------------------------------------------------
        // Real-interpreter fixtures. From here on the verifier is the actual
        // language toolchain (python3, node, sh, go, rustc, gcc, ruby) running
        // a genuine test against a genuinely buggy module, and the repair comes
        // from the deterministic mutation layer (`crate::mutation_repair`):
        // localize the module through the test's imports, mutate the blamed
        // code, keep only what the real test accepts. Each case pins its test
        // file: editing the oracle is scored as a failure.
        // ------------------------------------------------------------------
        // 15. Python: `range(len(xs) - 1)` drops the last element.
        EvalCase::new(
            "py-off-by-one-range",
            &[
                (
                    "total.py",
                    "def total(xs):\n    s = 0\n    for i in range(len(xs) - 1):\n        s += xs[i]\n    return s\n",
                ),
                (
                    "test_total.py",
                    "from total import total\n\nassert total([1, 2, 3]) == 6, \"expected 6 got %d\" % total([1, 2, 3])\nassert total([]) == 0\n",
                ),
            ],
            "python3",
            &["test_total.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_total.py"]),
        // 16. Python: `>` where `>=` was meant.
        EvalCase::new(
            "py-wrong-comparison",
            &[
                ("age.py", "def is_adult(age):\n    return age > 18\n"),
                (
                    "test_age.py",
                    "from age import is_adult\n\nassert is_adult(18) is True\nassert is_adult(17) is False\nassert is_adult(40) is True\n",
                ),
            ],
            "python3",
            &["test_age.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_age.py"]),
        // 17. Python: the computed value is never returned.
        EvalCase::new(
            "py-missing-return",
            &[
                ("dbl.py", "def double(x):\n    x * 2\n"),
                (
                    "test_dbl.py",
                    "from dbl import double\n\nassert double(3) == 6, \"expected 6 got %r\" % double(3)\nassert double(0) == 0\n",
                ),
            ],
            "python3",
            &["test_dbl.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_dbl.py"]),
        // 18. Python: arguments passed in the wrong order.
        EvalCase::new(
            "py-swapped-args",
            &[
                (
                    "names.py",
                    "def fullname(first, last):\n    return first + \" \" + last\n\n\ndef greet(first, last):\n    return \"Hello, \" + fullname(last, first)\n",
                ),
                (
                    "test_names.py",
                    "from names import greet\n\nassert greet(\"Ada\", \"Lovelace\") == \"Hello, Ada Lovelace\", greet(\"Ada\", \"Lovelace\")\n",
                ),
            ],
            "python3",
            &["test_names.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_names.py"]),
        // 19. Python: an inverted boolean.
        EvalCase::new(
            "py-boolean-inversion",
            &[
                ("empty.py", "def is_empty(xs):\n    return not len(xs) == 0\n"),
                (
                    "test_empty.py",
                    "from empty import is_empty\n\nassert is_empty([]) is True\nassert is_empty([1]) is False\nassert is_empty([1, 2]) is False\n",
                ),
            ],
            "python3",
            &["test_empty.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_empty.py"]),
        // 20. Python: `or` where `and` was meant.
        EvalCase::new(
            "py-logical-op",
            &[
                ("rng.py", "def in_range(x):\n    return x >= 0 or x <= 10\n"),
                (
                    "test_rng.py",
                    "from rng import in_range\n\nassert in_range(5) is True\nassert in_range(0) is True\nassert in_range(10) is True\nassert in_range(20) is False\nassert in_range(-1) is False\n",
                ),
            ],
            "python3",
            &["test_rng.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_rng.py"]),
        // 21. Python: an IndexError raised *inside* the module, so the traceback
        //     blames the module directly (no import hop needed) and the
        //     `index out of range` text steers the class order.
        EvalCase::new(
            "py-index-exception",
            &[
                ("last.py", "def last(xs):\n    return xs[len(xs)]\n"),
                (
                    "test_last.py",
                    "from last import last\n\nassert last([1, 2, 3]) == 3\nassert last([7]) == 7\n",
                ),
            ],
            "python3",
            &["test_last.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_last.py"]),
        // 22. JavaScript: `<=` runs one past the end (NaN).
        EvalCase::new(
            "js-off-by-one-loop",
            &[
                (
                    "sum.js",
                    "function sum(xs) {\n  let total = 0;\n  for (let i = 0; i <= xs.length; i++) {\n    total += xs[i];\n  }\n  return total;\n}\nmodule.exports = sum;\n",
                ),
                (
                    "sum.test.js",
                    "const assert = require('assert');\nconst sum = require('./sum');\nassert.strictEqual(sum([1, 2, 3]), 6);\nassert.strictEqual(sum([]), 0);\n",
                ),
            ],
            "node",
            &["sum.test.js"],
        )
        .requires(&["node"])
        .unchanged(&["sum.test.js"]),
        // 23. JavaScript: missing `return` (the test sees `undefined`).
        EvalCase::new(
            "js-missing-return",
            &[
                (
                    "dbl.js",
                    "function double(x) {\n  x * 2;\n}\nmodule.exports = double;\n",
                ),
                (
                    "dbl.test.js",
                    "const assert = require('assert');\nconst double = require('./dbl');\nassert.strictEqual(double(3), 6);\nassert.strictEqual(double(0), 0);\n",
                ),
            ],
            "node",
            &["dbl.test.js"],
        )
        .requires(&["node"])
        .unchanged(&["dbl.test.js"]),
        // 24. JavaScript: wrong arithmetic operator.
        EvalCase::new(
            "js-wrong-arith",
            &[
                (
                    "area.js",
                    "function area(w, h) {\n  return w + h;\n}\nmodule.exports = area;\n",
                ),
                (
                    "area.test.js",
                    "const assert = require('assert');\nconst area = require('./area');\nassert.strictEqual(area(3, 4), 12);\nassert.strictEqual(area(2, 5), 10);\n",
                ),
            ],
            "node",
            &["area.test.js"],
        )
        .requires(&["node"])
        .unchanged(&["area.test.js"]),
        // 25. POSIX shell: a `test` operator with the wrong sense, reached
        //     through the test script's `. ./lib.sh`.
        EvalCase::new(
            "sh-wrong-test-op",
            &[
                ("lib.sh", "is_positive() {\n  [ \"$1\" -lt 0 ]\n}\n"),
                (
                    "test.sh",
                    "#!/bin/sh\n. ./lib.sh\nis_positive 5 || { echo \"test.sh:3: is_positive 5 should succeed\"; exit 1; }\nif is_positive -2; then echo \"test.sh:4: is_positive -2 should fail\"; exit 1; fi\nif is_positive 0; then echo \"test.sh:5: is_positive 0 should fail\"; exit 1; fi\n",
                ),
            ],
            &posix_shell(),
            &["test.sh"],
        )
        .unchanged(&["test.sh"]),
        // 26. Go: `go test` blames `calc_test.go`; the package sibling is the
        //     repair target. Toolchain-heavy, so it is skipped without `go`.
        EvalCase::new(
            "go-off-by-one",
            &[
                ("go.mod", "module calc\n\ngo 1.18\n"),
                (
                    "calc.go",
                    "package calc\n\nfunc Sum(xs []int) int {\n\ts := 0\n\tfor i := 0; i < len(xs)-1; i++ {\n\t\ts += xs[i]\n\t}\n\treturn s\n}\n",
                ),
                (
                    "calc_test.go",
                    "package calc\n\nimport \"testing\"\n\nfunc TestSum(t *testing.T) {\n\tif got := Sum([]int{1, 2, 3}); got != 6 {\n\t\tt.Fatalf(\"expected 6 got %d\", got)\n\t}\n\tif got := Sum(nil); got != 0 {\n\t\tt.Fatalf(\"expected 0 got %d\", got)\n\t}\n}\n",
                ),
            ],
            "go",
            &["test", "./..."],
        )
        .requires(&["go"])
        .unchanged(&["calc_test.go", "go.mod"]),
        // 27. Rust: a trailing `;` turns the tail expression into `()`; rustc's
        //     own diagnostic ("remove this semicolon") steers the repair.
        EvalCase::new(
            "rust-missing-return-semicolon",
            &[("lib.rs", "pub fn inc(x: i32) -> i32 {\n    x + 1;\n}\n")],
            &posix_shell(),
            &[
                "-c",
                "rustc --edition 2021 --crate-type lib -o ./libinc.rlib lib.rs",
            ],
        )
        .requires(&["rustc"]),
        // 28. Rust: an off-by-one range in the same file as its `#[test]`; the
        //     panic blames the assertion line, which is an oracle line and is
        //     never mutated, so the search falls through to the loop bound.
        EvalCase::new(
            "rust-off-by-one-range",
            &[(
                "calc.rs",
                "pub fn sum(xs: &[i32]) -> i32 {\n    let mut s = 0;\n    for i in 0..xs.len() - 1 {\n        s += xs[i];\n    }\n    s\n}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn sums_everything() {\n        assert_eq!(super::sum(&[1, 2, 3]), 6);\n        assert_eq!(super::sum(&[]), 0);\n    }\n}\n",
            )],
            &posix_shell(),
            &[
                "-c",
                "rustc --edition 2021 --test -o ./calc_test calc.rs && ./calc_test",
            ],
        )
        .requires(&["rustc"]),
        // 29. C: the test blames itself; `#include \"calc.h\"` leads to calc.c.
        EvalCase::new(
            "c-wrong-comparison",
            &[
                ("calc.h", "int is_adult(int age);\n"),
                (
                    "calc.c",
                    "#include \"calc.h\"\n\nint is_adult(int age) {\n    return age > 18;\n}\n",
                ),
                (
                    "test.c",
                    "#include <stdio.h>\n#include \"calc.h\"\n\nint main(void) {\n    if (!is_adult(18)) {\n        fprintf(stderr, \"test.c:%d: is_adult(18) should be true\\n\", __LINE__);\n        return 1;\n    }\n    if (is_adult(17)) {\n        fprintf(stderr, \"test.c:%d: is_adult(17) should be false\\n\", __LINE__);\n        return 1;\n    }\n    return 0;\n}\n",
                ),
            ],
            &posix_shell(),
            &["-c", "gcc -o ./t test.c calc.c && ./t"],
        )
        .requires(&["gcc"])
        .unchanged(&["test.c"]),
        // 30. Ruby: `xs[xs.length]` is nil; `require_relative` leads to calc.rb.
        EvalCase::new(
            "rb-off-by-one",
            &[
                ("calc.rb", "def last(xs)\n  xs[xs.length]\nend\n"),
                (
                    "test.rb",
                    "require_relative 'calc'\nraise \"test.rb:2: expected 3 got #{last([1, 2, 3]).inspect}\" unless last([1, 2, 3]) == 3\nraise \"test.rb:3: expected 7\" unless last([7]) == 7\n",
                ),
            ],
            "ruby",
            &["test.rb"],
        )
        .requires(&["ruby"])
        .unchanged(&["test.rb"]),
        // 31. Python: a misspelled import. The traceback names the missing
        //     module; the repair rewrites the import to the closest sibling file.
        EvalCase::new(
            "py-wrong-import",
            &[
                ("calculator.py", "def add(a, b):\n    return a + b\n"),
                (
                    "app.py",
                    "from calcualtor import add\n\n\ndef total(a, b):\n    return add(a, b)\n",
                ),
                (
                    "test_app.py",
                    "from app import total\n\nassert total(2, 3) == 5\nassert total(0, 0) == 0\n",
                ),
            ],
            "python3",
            &["test_app.py"],
        )
        .requires(&["python3"])
        .unchanged(&["test_app.py", "calculator.py"]),
        // 32. JavaScript: a misspelled `require` path.
        EvalCase::new(
            "js-wrong-import",
            &[
                ("calculator.js", "module.exports = (a, b) => a + b;\n"),
                (
                    "app.js",
                    "const add = require('./calcualtor');\nmodule.exports = (a, b) => add(a, b);\n",
                ),
                (
                    "app.test.js",
                    "const assert = require('assert');\nconst total = require('./app');\nassert.strictEqual(total(2, 3), 5);\n",
                ),
            ],
            "node",
            &["app.test.js"],
        )
        .requires(&["node"])
        .unchanged(&["app.test.js", "calculator.js"]),
    ]
}

/// Materialize `cases` into fresh temp roots, run the autonomous loop on each,
/// and collect pass/fail. Each case gets its own throwaway directory which is
/// removed afterward. `solve` is run with no caller-named source, so the loop
/// must localize the fault itself — exactly the production path.
pub fn run_eval(pipeline: &InferencePipeline, cases: &[EvalCase]) -> EvalReport {
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let missing: Vec<&str> = case
            .requires
            .iter()
            .map(String::as_str)
            .filter(|p| !program_available(p))
            .collect();
        if !missing.is_empty() {
            results.push(CaseResult {
                name: case.name.clone(),
                solved: false,
                skipped: Some(format!("missing: {}", missing.join(", "))),
            });
            continue;
        }
        let solved = run_one(pipeline, case).unwrap_or(false);
        results.push(CaseResult {
            name: case.name.clone(),
            solved,
            skipped: None,
        });
    }
    EvalReport { results }
}

/// Is `program` runnable from `PATH`? (A `which` without spawning anything.)
pub fn program_available(program: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        let direct = dir.join(program);
        if direct.is_file() {
            return true;
        }
        cfg!(windows)
            && ["exe", "cmd", "bat"]
                .iter()
                .any(|ext| dir.join(format!("{program}.{ext}")).is_file())
    })
}

fn run_one(pipeline: &InferencePipeline, case: &EvalCase) -> std::io::Result<bool> {
    let root = unique_root(&case.name);
    std::fs::create_dir_all(&root)?;
    for (rel, content) in &case.files {
        // EvalCase is public; never let a case write outside its sandbox root via
        // an absolute path or a `..` escape.
        if !is_safe_rel(rel) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("eval case path escapes the sandbox root: {rel}"),
            ));
        }
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, content)?;
    }
    let pinned: Vec<(String, Vec<u8>)> = case
        .expect_unchanged
        .iter()
        .filter(|rel| is_safe_rel(rel))
        .filter_map(|rel| std::fs::read(root.join(rel)).ok().map(|b| (rel.clone(), b)))
        .collect();
    let mut solved = if case.agentic_fix.is_empty() {
        // Single-file repair path: the autonomous solve loop must localize and
        // fix the fault itself (no caller-named source).
        let opts = SolveOptions {
            max_rounds: 2,
            max_restarts: 1,
            anchor: Some(root.clone()),
            source_path: None,       // force autonomous localization
            patch_memory_path: None, // isolate the eval from real fleet memory
            ..SolveOptions::default()
        };
        solve(pipeline, &case.command, &case.args, &opts)
            .map(|r| r.solved)
            .unwrap_or(false)
    } else {
        solve_agentic_case(case, &root)
    };
    // A green verifier is not enough if the fix landed in a pinned file (the
    // test): that is an overfit, scored as a failure.
    if solved {
        for (rel, before) in &pinned {
            if std::fs::read(root.join(rel)).ok().as_deref() != Some(before.as_slice()) {
                println!("[axiom-eval-agentic] {}: pinned file {rel} was modified — overfit, scored FAIL", case.name);
                solved = false;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
    Ok(solved)
}

/// Run a coordinated multi-file case through the agentic loop with a scripted
/// (deterministic, no-LLM) proposer that applies the case's `agentic_fix` as one
/// atomic edit-set. Certifies the transaction apply/verify/commit machinery on a
/// fixture the single-file path cannot solve.
fn solve_agentic_case(case: &EvalCase, root: &Path) -> bool {
    let mut edits = EditSet::new();
    for (rel, content) in &case.agentic_fix {
        // `agentic_fix` is public too: a committed transaction writes these
        // paths, so an absolute/`..` escape must be refused (treat as unsolved)
        // rather than allowed to overwrite files outside the sandbox root.
        if !is_safe_rel(rel) {
            return false;
        }
        edits = edits.with(root.join(rel), content.clone());
    }
    let mut proposer = ScriptedProposer { once: Some(edits) };
    let mut memory = AttemptMemory::new();
    let working_dir = Some(root);
    let outcome = match &case.holdout {
        // Held-out split: commit only when the candidate passes both the
        // train verifier and the held-out verifier (anti-overfitting gate).
        Some((holdout_cmd, holdout_args)) => agentic_loop_with_holdout(
            &format!("eval:{}", case.name),
            3,
            &mut memory,
            &mut proposer,
            || run_verify(&case.command, &case.args, working_dir),
            || run_verify(holdout_cmd, holdout_args, working_dir),
            || run_verify_capture(&case.command, &case.args, working_dir).1,
        ),
        None => agentic_loop(
            &format!("eval:{}", case.name),
            3,
            &mut memory,
            &mut proposer,
            || run_verify(&case.command, &case.args, working_dir),
            || run_verify_capture(&case.command, &case.args, working_dir).1,
        ),
    };
    outcome.solved
}

/// A case-supplied relative path is safe to materialize/commit under the sandbox
/// root only if it is relative and contains no `..` traversal. Applied to both
/// `EvalCase::files` (seed) and `EvalCase::agentic_fix` (committed edits), since
/// both are public and a committed transaction writes the latter to disk.
fn is_safe_rel(rel: &str) -> bool {
    let p = Path::new(rel);
    // `has_root()` catches rooted-but-not-absolute forms Windows treats as
    // relative (`/etc/passwd`, `\x`); prefix components catch drive/UNC paths
    // (`C:x` is drive-relative yet neither absolute nor rooted on Windows).
    !p.is_absolute()
        && !p.has_root()
        && !p.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::Prefix(_)
            )
        })
}

fn unique_root(name: &str) -> PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let safe: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    std::env::temp_dir().join(format!("axiom_eval_{safe}_{n}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AxiomConfig;
    use candle_core::Device;

    fn tiny_pipeline() -> InferencePipeline {
        let cfg = AxiomConfig {
            d_model: 16,
            n_layers: 2,
            vocab_size: 64,
            lr_inner: 1e-3,
            norm_eps: 1e-6,
        };
        InferencePipeline::new(cfg, Device::Cpu).expect("tiny pipeline")
    }

    #[test]
    fn builtin_suite_is_fully_solved_by_the_deterministic_loop() {
        // The headline guarantee: the autonomous loop repairs every built-in
        // fixture end-to-end with no LLM — a real, reproducible capability score.
        let report = run_eval(&tiny_pipeline(), &builtin_cases());
        assert_eq!(
            report.solved(),
            report.total(),
            "every deterministic fixture should be auto-repaired:\n{}",
            report.summary()
        );
        assert!((report.score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unsafe_case_paths_are_rejected() {
        // Both seed files and committed agentic-fix paths must reject escapes.
        assert!(is_safe_rel("src/a.rs"));
        assert!(is_safe_rel("a.sh"));
        assert!(!is_safe_rel("/etc/passwd"));
        assert!(!is_safe_rel("../escape.rs"));
        assert!(!is_safe_rel("src/../../escape.rs"));

        // An agentic case whose fix escapes the sandbox is refused (unsolved),
        // never committed outside the root.
        let case = EvalCase::new_agentic(
            "escape-attempt",
            &[("a.sh", "#!/bin/sh\nexit 1\n")],
            &posix_shell(),
            &["-c", "exit 1"],
            &[("../../evil.sh", "pwned")],
        );
        let root = unique_root(&case.name);
        std::fs::create_dir_all(&root).unwrap();
        assert!(
            !solve_agentic_case(&case, &root),
            "escaping fix must be refused"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn report_score_math() {
        let report = EvalReport {
            results: vec![
                CaseResult {
                    name: "a".into(),
                    solved: true,
                    skipped: None,
                },
                CaseResult {
                    name: "b".into(),
                    solved: false,
                    skipped: None,
                },
                CaseResult {
                    name: "c".into(),
                    solved: false,
                    skipped: Some("missing: unobtainium".into()),
                },
            ],
        };
        assert_eq!(
            report.total(),
            2,
            "skipped cases are not in the denominator"
        );
        assert_eq!(report.solved(), 1);
        assert_eq!(report.skipped(), 1);
        assert!((report.score() - 0.5).abs() < f64::EPSILON);
        assert!(report.summary().contains("[SKIP] c (missing: unobtainium)"));
        assert!(report.summary().ends_with("score: 1/2 = 50% (1 skipped)"));
        assert!(EvalReport { results: vec![] }.score() == 0.0);
    }

    #[test]
    fn missing_requirement_skips_instead_of_failing() {
        let case = EvalCase::new(
            "needs-unobtainium",
            &[("a.sh", "#!/bin/sh\nexit 1\n")],
            &posix_shell(),
            &["a.sh"],
        )
        .requires(&["axiom-definitely-not-installed-binary"]);
        let report = run_eval(&tiny_pipeline(), &[case]);
        assert_eq!(report.total(), 0);
        assert_eq!(report.skipped(), 1);
        assert!(report.results[0]
            .skipped
            .as_deref()
            .unwrap()
            .contains("missing:"));
        assert!(program_available(&posix_shell()) || cfg!(windows));
        assert!(!program_available("axiom-definitely-not-installed-binary"));
    }

    #[test]
    fn a_fix_that_edits_a_pinned_file_is_scored_as_overfit() {
        // A repair the verifier accepts is still a failure when it lands in a
        // pinned file. The same case solves when nothing is pinned, so it is
        // the pin, not the repair, that scores it FAIL.
        let case = || {
            EvalCase::new(
                "pinned-file",
                &[("limits.sh", "is_adult() {\n  [ \"$1\" -gt 18 ]\n}\n")],
                &posix_shell(),
                &[
                    "-c",
                    ". ./limits.sh; { is_adult 18 && ! is_adult 17; } || \
                     { echo 'limits.sh:2:3: is_adult is wrong'; exit 1; }",
                ],
            )
        };
        let free = run_eval(&tiny_pipeline(), &[case()]);
        assert_eq!(free.solved(), 1, "{}", free.summary());

        let pinned = run_eval(&tiny_pipeline(), &[case().unchanged(&["limits.sh"])]);
        assert_eq!(pinned.total(), 1);
        assert_eq!(pinned.solved(), 0, "{}", pinned.summary());
    }
}
