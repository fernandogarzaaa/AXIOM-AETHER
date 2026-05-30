#!/usr/bin/env node
// axiom-structural.js — deterministic AST/structural pre-filter for axiom-guard.
//
// Extracts structural features from Rust source via comment/string-stripped
// scanning and decides STRICT block/pass. Used by scripts/axiom-guard.sh.
//
// Usage:
//   node axiom-structural.js --report FILE...           human table (calibration)
//   node axiom-structural.js --manifest TSV --json      JSON verdicts for the guard
//        (TSV lines: "<blobPath>\t<displayLabel>")
//
// Env:
//   AXIOM_GUARD_MAX_NESTING        max brace-nesting depth (default 8)
//   AXIOM_GUARD_UNSAFE_WHITELIST   regex; paths matching may use `unsafe`
//                                  (default: quantization|kernel|chunk_kernel|memory_pool)
'use strict';
const fs = require('fs');
const path = require('path');

// Manifest lines are "<index>\t<label>"; the staged blob is "<dir>/blob_<index>".
function manifestEntries(tsv) {
  const dir = path.dirname(tsv);
  return fs.readFileSync(tsv, 'utf8').trim().split('\n').filter(Boolean).map((l) => {
    const i = l.indexOf('\t');
    const idx = l.slice(0, i);
    return { blob: path.join(dir, 'blob_' + idx), label: l.slice(i + 1) };
  });
}

const MAX_NESTING = parseInt(process.env.AXIOM_GUARD_MAX_NESTING || '8', 10);
const UNSAFE_WHITELIST = new RegExp(
  process.env.AXIOM_GUARD_UNSAFE_WHITELIST ||
    '(quantization|kernel|chunk_kernel|memory_pool)'
);

// Strip block comments, line comments, raw/normal strings, and char literals so
// braces/keywords inside them never affect structural counts. Lifetimes ('a)
// and loop labels ('name:) have no closing quote, so the char-literal rule
// leaves them intact — exactly what the goto detector needs.
function strip(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/r#"[\s\S]*?"#/g, '""')
    .replace(/r"[^"]*"/g, '""')
    .replace(/"(\\.|[^"\\])*"/g, '""')
    .replace(/'(\\.|[^'\\])'/g, "' '");
}

function maxBraceDepth(s) {
  let depth = 0, max = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '{') { depth++; if (depth > max) max = depth; }
    else if (c === '}') { if (depth > 0) depth--; }
  }
  return max;
}

function analyze(blobPath, label) {
  const raw = fs.readFileSync(blobPath, 'utf8');
  const s = strip(raw);
  const maxDepth = maxBraceDepth(s);
  const unsafeCount = (s.match(/\bunsafe\b/g) || []).length;
  const staticMut = (s.match(/\bstatic\s+mut\b/g) || []).length;
  const labeledLoop = /'[A-Za-z_]\w*:\s*loop\b/.test(s);
  const numericArms = (s.match(/(^|\n)\s*[0-9]+\s*=>/g) || []).length;
  const goto = labeledLoop && numericArms >= 2;

  const violations = [];
  // Structural rules are Rust-specific; only enforce them on .rs files.
  const isRust = /\.rs$/.test(label);
  if (isRust && unsafeCount > 0 && !UNSAFE_WHITELIST.test(label)) {
    violations.push(
      `Unsafe density: ${unsafeCount} \`unsafe\` block(s) outside whitelisted low-level layers`
    );
  }
  if (isRust && maxDepth > MAX_NESTING) {
    violations.push(
      `Nesting anomaly: max brace-nesting depth ${maxDepth} exceeds limit ${MAX_NESTING} (cyclomatic/nesting complexity)`
    );
  }
  if (isRust && staticMut > 0) {
    violations.push(
      `Foreign paradigm: ${staticMut} \`static mut\` global mutable state declaration(s)`
    );
  }
  if (isRust && goto) {
    violations.push(
      `Foreign paradigm: goto-style state machine (labeled loop + ${numericArms} numeric match arms)`
    );
  }
  return { label, maxDepth, unsafeCount, staticMut, goto, numericArms, violations };
}

const args = process.argv.slice(2);
if (args[0] === '--report') {
  const rows = args.slice(1).map((f) => analyze(f, f)).sort((a, b) => a.maxDepth - b.maxDepth);
  console.log('depth unsafe statMut goto  file');
  for (const r of rows) {
    console.log(
      String(r.maxDepth).padStart(5) + ' ' + String(r.unsafeCount).padStart(6) + ' ' +
      String(r.staticMut).padStart(7) + ' ' + String(r.goto ? 1 : 0).padStart(4) + '  ' + r.label +
      (r.violations.length ? '   <<< ' + r.violations.length + ' violation(s)' : '')
    );
  }
} else if (args[0] === '--manifest') {
  const out = manifestEntries(args[1]).map((e) => analyze(e.blob, e.label));
  process.stdout.write(JSON.stringify(out));
} else if (args[0] === '--gate') {
  // Strict deterministic gate. Block (exit 1) on any structural violation,
  // printing a diagnostic trace to stderr; otherwise exit 0 (silent pass).
  const results = manifestEntries(args[1]).map((e) => analyze(e.blob, e.label));
  const blocked = results.filter((r) => r.violations.length > 0);
  const E = String.fromCharCode(27);
  const R = E + '[31m', Y = E + '[33m', X = E + '[0m', B = E + '[1m';
  if (blocked.length) {
    const w = (s) => process.stderr.write(s + '\n');
    w('');
    w(R + B + '✖ Axiom Structural Gate: COMMIT BLOCKED' + X +
      '  (' + blocked.length + ' file(s) failed deterministic checks)');
    w('');
    for (const r of blocked) {
      w('  ' + R + '✖' + X + ' ' + r.label);
      for (const v of r.violations) w('      - ' + v);
      w('      [features] nesting_depth=' + r.maxDepth + '  unsafe=' + r.unsafeCount +
        '  static_mut=' + r.staticMut + '  goto=' + (r.goto ? 'yes' : 'no'));
    }
    w('');
    w(Y + '  Structural anti-patterns are a hard gate. Options:' + X);
    w('    1. Refactor to remove the flagged construct(s), then re-stage.');
    w('    2. For legitimate low-level code, whitelist the path via');
    w('       AXIOM_GUARD_UNSAFE_WHITELIST (regex) or raise AXIOM_GUARD_MAX_NESTING.');
    w('    3. Override for this one commit:  git commit --no-verify');
    w('');
    process.exit(1);
  }
  process.exit(0);
} else {
  console.error('usage: axiom-structural.js --report FILE... | --manifest TSV --json');
  process.exit(2);
}
