//! ChimeraLang — a Rust port of the core of the ChimeraLang AI-cognition
//! language (https://github.com/fernandogarzaaa/ChimeraLang), integrated into
//! the AXIOM-AETHER engine.
//!
//! ## Why a port (not a subprocess)
//!
//! ChimeraLang and AXIOM already share DNA: `crate::belief::BetaBelief` was
//! itself "ported and adapted from ChimeraLang's `cir/nodes.py::BetaDist`", and
//! `crate::provenance` mirrors ChimeraLang's integrity certificates. Re-porting
//! the *language* in Rust lets the two belief/provenance systems become **one**:
//! ChimeraLang's `belief/inquire/resolve/guard/evolve` runs directly on
//! `BetaBelief`, its certificates are AXIOM `SignedExport`s, and an `inquire`
//! can be answered by AXIOM's own model via the [`InquiryAdapter`] seam — no
//! Python runtime, no IPC.
//!
//! ## Scope (honest)
//!
//! This is a faithful **core subset**, not the entire language. Covered:
//!   * VM path: `val`, `emit`, `for … in … end`, arithmetic / string concat /
//!     comparison expressions, list literals, member access (`.confidence`,
//!     `.raw`), confidence propagation.
//!   * CIR / belief path: `belief NAME := inquire { prompt, agents, ttl }`,
//!     `resolve NAME with consensus { threshold }`,
//!     `guard NAME against hallucination { max_risk }`,
//!     `evolve NAME until stable { max_iter }`, `emit NAME`.
//!   * Certificates over a run (SHA-256 + optional HMAC via `crate::provenance`).
//!
//! Deferred (tracked for follow-up PRs): `gate` quantum-consensus branches,
//! `fn`/`match`/`goal`/`reason`, the type checker's capability enforcement, the
//! PyTorch/LLVM compiler backends, RAG, and symbol emergence. The module is
//! structured (lexer → parser → {vm, cir}) so these slot in incrementally.

use std::collections::BTreeMap;

use crate::belief::BetaBelief;
use crate::provenance::{sign_export, verify_export, ProvenanceError, SignedExport};

// ===========================================================================
// Lexer
// ===========================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    // punctuation / operators
    Assign,      // =
    ColonAssign, // :=
    Colon,       // :
    Comma,       // ,
    Dot,         // .
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Plus,
    Minus,
    Star,
    Slash,
    Lt,
    Gt,
    Le,
    Ge,
    EqEq,
    Ne,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LexError(pub String);

/// Tokenize ChimeraLang source. Whitespace (incl. newlines) is insignificant;
/// `#` starts a line comment.
pub fn lex(src: &str) -> Result<Vec<Tok>, LexError> {
    let b = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();
    let is_id_start = |c: u8| c.is_ascii_alphabetic() || c == b'_';
    let is_id_part = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // identifiers / keywords
        if is_id_start(c) {
            let start = i;
            i += 1;
            while i < b.len() && is_id_part(b[i]) {
                i += 1;
            }
            out.push(Tok::Ident(src[start..i].to_string()));
            continue;
        }
        // numbers
        if c.is_ascii_digit() || (c == b'-' && i + 1 < b.len() && b[i + 1].is_ascii_digit()
            && matches!(out.last(), None | Some(Tok::LParen | Tok::LBracket | Tok::Comma | Tok::Colon | Tok::ColonAssign | Tok::Assign)))
        {
            let start = i;
            if c == b'-' {
                i += 1;
            }
            let mut is_float = false;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                if b[i] == b'.' {
                    // lookahead: a '.' followed by a digit is a decimal point;
                    // otherwise it's member access — stop the number.
                    if i + 1 < b.len() && b[i + 1].is_ascii_digit() {
                        is_float = true;
                        i += 1;
                    } else {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            let text = &src[start..i];
            if is_float {
                out.push(Tok::Float(text.parse().map_err(|_| LexError(format!("bad float {text}")))?));
            } else {
                out.push(Tok::Int(text.parse().map_err(|_| LexError(format!("bad int {text}")))?));
            }
            continue;
        }
        // strings
        if c == b'"' {
            i += 1;
            let mut s = String::new();
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < b.len() {
                    i += 1;
                    s.push(match b[i] {
                        b'n' => '\n',
                        b't' => '\t',
                        other => other as char,
                    });
                } else {
                    s.push(b[i] as char);
                }
                i += 1;
            }
            if i >= b.len() {
                return Err(LexError("unterminated string".into()));
            }
            i += 1; // closing quote
            out.push(Tok::Str(s));
            continue;
        }
        // multi-char operators
        let two = if i + 1 < b.len() { &src[i..i + 2] } else { "" };
        match two {
            ":=" => { out.push(Tok::ColonAssign); i += 2; continue; }
            "<=" => { out.push(Tok::Le); i += 2; continue; }
            ">=" => { out.push(Tok::Ge); i += 2; continue; }
            "==" => { out.push(Tok::EqEq); i += 2; continue; }
            "!=" => { out.push(Tok::Ne); i += 2; continue; }
            _ => {}
        }
        let single = match c {
            b'=' => Tok::Assign,
            b':' => Tok::Colon,
            b',' => Tok::Comma,
            b'.' => Tok::Dot,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b'+' => Tok::Plus,
            b'-' => Tok::Minus,
            b'*' => Tok::Star,
            b'/' => Tok::Slash,
            b'<' => Tok::Lt,
            b'>' => Tok::Gt,
            other => return Err(LexError(format!("unexpected char {:?}", other as char))),
        };
        out.push(single);
        i += 1;
    }
    Ok(out)
}

// ===========================================================================
// AST
// ===========================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Ident(String),
    List(Vec<Expr>),
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Member { base: Box<Expr>, field: String },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Val { name: String, value: Expr },
    Emit(Expr),
    For { var: String, iter: Expr, body: Vec<Stmt> },
    // CIR / belief path
    Belief { name: String, prompt: String, agents: Vec<String>, ttl: u64 },
    Resolve { name: String, threshold: f32 },
    Guard { name: String, max_risk: f32 },
    Evolve { name: String, max_iter: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub stmts: Vec<Stmt>,
    /// True if the program uses any belief construct ⇒ routes to the CIR path.
    pub uses_belief: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError(pub String);

// ===========================================================================
// Parser (recursive descent)
// ===========================================================================

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, want: &Tok) -> Result<(), ParseError> {
        match self.peek() {
            Some(t) if t == want => {
                self.pos += 1;
                Ok(())
            }
            other => Err(ParseError(format!("expected {want:?}, found {other:?}"))),
        }
    }
    fn ident(&mut self) -> Result<String, ParseError> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(ParseError(format!("expected identifier, found {other:?}"))),
        }
    }
    /// A keyword is just an identifier with a specific spelling.
    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == kw)
    }
    fn eat_kw(&mut self, kw: &str) -> Result<(), ParseError> {
        if self.is_kw(kw) {
            self.pos += 1;
            Ok(())
        } else {
            Err(ParseError(format!("expected keyword `{kw}`, found {:?}", self.peek())))
        }
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut stmts = Vec::new();
        let mut uses_belief = false;
        while self.peek().is_some() {
            let s = self.parse_stmt()?;
            if matches!(
                s,
                Stmt::Belief { .. }
                    | Stmt::Resolve { .. }
                    | Stmt::Guard { .. }
                    | Stmt::Evolve { .. }
            ) {
                uses_belief = true;
            }
            stmts.push(s);
        }
        Ok(Program { stmts, uses_belief })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        // Keyword-led statements.
        if self.is_kw("val") {
            self.pos += 1;
            let name = self.ident()?;
            self.eat(&Tok::Assign)?;
            let value = self.parse_expr()?;
            return Ok(Stmt::Val { name, value });
        }
        if self.is_kw("for") {
            self.pos += 1;
            let var = self.ident()?;
            self.eat_kw("in")?;
            let iter = self.parse_expr()?;
            let mut body = Vec::new();
            while !self.is_kw("end") {
                if self.peek().is_none() {
                    return Err(ParseError("unterminated `for` (missing `end`)".into()));
                }
                body.push(self.parse_stmt()?);
            }
            self.eat_kw("end")?;
            return Ok(Stmt::For { var, iter, body });
        }
        if self.is_kw("belief") {
            self.pos += 1;
            let name = self.ident()?;
            self.eat(&Tok::ColonAssign)?;
            self.eat_kw("inquire")?;
            return self.parse_inquire(name);
        }
        if self.is_kw("resolve") {
            self.pos += 1;
            let name = self.ident()?;
            self.eat_kw("with")?;
            self.eat_kw("consensus")?;
            let kv = self.parse_brace_kvs()?;
            let threshold = kv_f32(&kv, "threshold").unwrap_or(0.8);
            return Ok(Stmt::Resolve { name, threshold });
        }
        if self.is_kw("guard") {
            self.pos += 1;
            let name = self.ident()?;
            self.eat_kw("against")?;
            self.eat_kw("hallucination")?;
            let kv = self.parse_brace_kvs()?;
            let max_risk = kv_f32(&kv, "max_risk").unwrap_or(0.2);
            return Ok(Stmt::Guard { name, max_risk });
        }
        if self.is_kw("evolve") {
            self.pos += 1;
            let name = self.ident()?;
            self.eat_kw("until")?;
            self.eat_kw("stable")?;
            let kv = self.parse_brace_kvs()?;
            let max_iter = kv_f32(&kv, "max_iter").unwrap_or(3.0) as u32;
            return Ok(Stmt::Evolve { name, max_iter });
        }
        if self.is_kw("emit") {
            self.pos += 1;
            // `emit EXPR` always. A bare `emit NAME` where NAME is a known belief
            // is disambiguated at eval time on the CIR path (belief emit vs.
            // plain variable emit) — no parse-time guessing.
            let e = self.parse_expr()?;
            return Ok(Stmt::Emit(e));
        }
        Err(ParseError(format!("unexpected token at statement start: {:?}", self.peek())))
    }

    fn parse_inquire(&mut self, name: String) -> Result<Stmt, ParseError> {
        // inquire { prompt: "...", agents: [a, b], ttl: N }
        self.eat(&Tok::LBrace)?;
        let mut prompt = String::new();
        let mut agents = Vec::new();
        let mut ttl = 0u64;
        loop {
            if matches!(self.peek(), Some(Tok::RBrace)) {
                break;
            }
            let key = self.ident()?;
            self.eat(&Tok::Colon)?;
            match key.as_str() {
                "prompt" => match self.next() {
                    Some(Tok::Str(s)) => prompt = s,
                    other => return Err(ParseError(format!("prompt must be a string, got {other:?}"))),
                },
                "agents" => {
                    self.eat(&Tok::LBracket)?;
                    while !matches!(self.peek(), Some(Tok::RBracket)) {
                        agents.push(self.ident()?);
                        if matches!(self.peek(), Some(Tok::Comma)) {
                            self.pos += 1;
                        }
                    }
                    self.eat(&Tok::RBracket)?;
                }
                "ttl" => match self.next() {
                    Some(Tok::Int(n)) => ttl = n.max(0) as u64,
                    other => return Err(ParseError(format!("ttl must be an int, got {other:?}"))),
                },
                other => return Err(ParseError(format!("unknown inquire key `{other}`"))),
            }
            if matches!(self.peek(), Some(Tok::Comma)) {
                self.pos += 1;
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(Stmt::Belief { name, prompt, agents, ttl })
    }

    /// Parse `{ key: value, ... }` where values are numeric or identifiers.
    fn parse_brace_kvs(&mut self) -> Result<BTreeMap<String, f32>, ParseError> {
        self.eat(&Tok::LBrace)?;
        let mut map = BTreeMap::new();
        loop {
            if matches!(self.peek(), Some(Tok::RBrace)) {
                break;
            }
            let key = self.ident()?;
            self.eat(&Tok::Colon)?;
            match self.next() {
                Some(Tok::Float(f)) => { map.insert(key, f as f32); }
                Some(Tok::Int(n)) => { map.insert(key, n as f32); }
                // identifier-valued options (e.g. strategy: dempster_shafer) are
                // accepted and ignored in this core subset.
                Some(Tok::Ident(_)) => {}
                other => return Err(ParseError(format!("bad option value for `{key}`: {other:?}"))),
            }
            if matches!(self.peek(), Some(Tok::Comma)) {
                self.pos += 1;
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(map)
    }

    // expression: comparison over additive over multiplicative over primary
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_cmp()
    }
    fn parse_cmp(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Lt) => BinOp::Lt,
                Some(Tok::Gt) => BinOp::Gt,
                Some(Tok::Le) => BinOp::Le,
                Some(Tok::Ge) => BinOp::Ge,
                Some(Tok::EqEq) => BinOp::Eq,
                Some(Tok::Ne) => BinOp::Ne,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_add()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }
    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => BinOp::Add,
                Some(Tok::Minus) => BinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_mul()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }
    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_postfix()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => BinOp::Mul,
                Some(Tok::Slash) => BinOp::Div,
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_postfix()?;
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }
    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut e = self.parse_primary()?;
        while matches!(self.peek(), Some(Tok::Dot)) {
            self.pos += 1;
            let field = self.ident()?;
            e = Expr::Member { base: Box::new(e), field };
        }
        Ok(e)
    }
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.next() {
            Some(Tok::Int(n)) => Ok(Expr::Int(n)),
            Some(Tok::Float(f)) => Ok(Expr::Float(f)),
            Some(Tok::Str(s)) => Ok(Expr::Str(s)),
            Some(Tok::Ident(s)) => match s.as_str() {
                "true" => Ok(Expr::Bool(true)),
                "false" => Ok(Expr::Bool(false)),
                _ => Ok(Expr::Ident(s)),
            },
            Some(Tok::LParen) => {
                let e = self.parse_expr()?;
                self.eat(&Tok::RParen)?;
                Ok(e)
            }
            Some(Tok::LBracket) => {
                let mut items = Vec::new();
                while !matches!(self.peek(), Some(Tok::RBracket)) {
                    items.push(self.parse_expr()?);
                    if matches!(self.peek(), Some(Tok::Comma)) {
                        self.pos += 1;
                    }
                }
                self.eat(&Tok::RBracket)?;
                Ok(Expr::List(items))
            }
            other => Err(ParseError(format!("unexpected token in expression: {other:?}"))),
        }
    }
}

fn kv_f32(m: &BTreeMap<String, f32>, k: &str) -> Option<f32> {
    m.get(k).copied()
}

/// Lex + parse a ChimeraLang source string into a [`Program`].
pub fn parse(src: &str) -> Result<Program, ParseError> {
    let toks = lex(src).map_err(|e| ParseError(e.0))?;
    let mut p = Parser { toks, pos: 0 };
    p.parse_program()
}

/// Static check (lex + parse) of a source string — the verify gate the repair
/// loop uses for `.chimera` files. Returns a human-readable error on failure.
pub fn check(src: &str) -> Result<(), String> {
    parse(src).map(|_| ()).map_err(|e| e.0)
}

/// Parse and run a source string with the given adapter (mock if `None`).
pub fn run_source(src: &str, adapter: Option<&dyn InquiryAdapter>) -> Result<RunResult, String> {
    let prog = parse(src).map_err(|e| e.0)?;
    let mock = MockAdapter;
    let adapter = adapter.unwrap_or(&mock);
    run(&prog, adapter).map_err(|e| e.0)
}

// ===========================================================================
// Values + VM path
// ===========================================================================

/// A runtime value carrying a confidence in [0,1] (ChimeraLang propagates
/// confidence through every computation).
#[derive(Debug, Clone, PartialEq)]
pub struct CValue {
    pub value: Value,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    List(Vec<CValue>),
}

impl CValue {
    fn certain(v: Value) -> Self {
        CValue { value: v, confidence: 1.0 }
    }
    fn display(&self) -> String {
        match &self.value {
            Value::Int(n) => n.to_string(),
            Value::Float(f) => format!("{f}"),
            Value::Str(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::List(xs) => {
                let inner: Vec<String> = xs.iter().map(|x| x.display()).collect();
                format!("[{}]", inner.join(", "))
            }
        }
    }
}

/// Result of running a program.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunResult {
    /// Values produced by `emit`, rendered to strings.
    pub emitted: Vec<String>,
    /// Execution trace lines (human-readable).
    pub trace: Vec<String>,
    /// Guard violations recorded by the CIR path (belief failed its guard).
    pub guard_violations: Vec<String>,
    /// Final belief means by name (CIR path).
    pub beliefs: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunError(pub String);

/// Run a program, auto-routing to the VM or CIR path. `adapter` answers belief
/// inquiries (only consulted on the CIR path).
pub fn run(program: &Program, adapter: &dyn InquiryAdapter) -> Result<RunResult, RunError> {
    if program.uses_belief {
        run_cir(program, adapter)
    } else {
        run_vm(program)
    }
}

fn run_vm(program: &Program) -> Result<RunResult, RunError> {
    let mut env: BTreeMap<String, CValue> = BTreeMap::new();
    let mut res = RunResult::default();
    for s in &program.stmts {
        exec_vm_stmt(s, &mut env, &mut res)?;
    }
    Ok(res)
}

fn exec_vm_stmt(
    s: &Stmt,
    env: &mut BTreeMap<String, CValue>,
    res: &mut RunResult,
) -> Result<(), RunError> {
    match s {
        Stmt::Val { name, value } => {
            let v = eval(value, env)?;
            res.trace.push(format!("val {name} = {} (conf {:.2})", v.display(), v.confidence));
            env.insert(name.clone(), v);
            Ok(())
        }
        Stmt::Emit(e) => {
            let v = eval(e, env)?;
            res.emitted.push(v.display());
            Ok(())
        }
        Stmt::For { var, iter, body } => {
            let it = eval(iter, env)?;
            let items = match it.value {
                Value::List(xs) => xs,
                Value::Str(s) => s.chars().map(|c| CValue::certain(Value::Str(c.to_string()))).collect(),
                other => return Err(RunError(format!("cannot iterate over {other:?}"))),
            };
            for item in items {
                env.insert(var.clone(), item);
                for b in body {
                    exec_vm_stmt(b, env, res)?;
                }
            }
            Ok(())
        }
        // belief statements never reach the VM path
        other => Err(RunError(format!("belief construct on the VM path: {other:?}"))),
    }
}

fn eval(e: &Expr, env: &BTreeMap<String, CValue>) -> Result<CValue, RunError> {
    match e {
        Expr::Int(n) => Ok(CValue::certain(Value::Int(*n))),
        Expr::Float(f) => Ok(CValue::certain(Value::Float(*f))),
        Expr::Str(s) => Ok(CValue::certain(Value::Str(s.clone()))),
        Expr::Bool(b) => Ok(CValue::certain(Value::Bool(*b))),
        Expr::Ident(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| RunError(format!("undefined variable `{name}`"))),
        Expr::List(items) => {
            let mut xs = Vec::with_capacity(items.len());
            let mut conf = 1.0f32;
            for it in items {
                let v = eval(it, env)?;
                conf = conf.min(v.confidence);
                xs.push(v);
            }
            Ok(CValue { value: Value::List(xs), confidence: conf })
        }
        Expr::Member { base, field } => {
            let b = eval(base, env)?;
            match field.as_str() {
                // confidence is itself a certain float
                "confidence" => Ok(CValue::certain(Value::Float(b.confidence as f64))),
                "raw" => Ok(CValue { confidence: 1.0, ..b }),
                other => Err(RunError(format!("unknown member `.{other}`"))),
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let a = eval(lhs, env)?;
            let c = eval(rhs, env)?;
            // confidence propagates as the minimum of operands (a chain is only
            // as trustworthy as its least-trusted input).
            let conf = a.confidence.min(c.confidence);
            let v = eval_binop(*op, &a.value, &c.value)?;
            Ok(CValue { value: v, confidence: conf })
        }
    }
}

fn eval_binop(op: BinOp, a: &Value, b: &Value) -> Result<Value, RunError> {
    use Value::*;
    // numeric coercion helper
    let as_f = |v: &Value| -> Option<f64> {
        match v {
            Int(n) => Some(*n as f64),
            Float(f) => Some(*f),
            _ => None,
        }
    };
    match op {
        BinOp::Add => match (a, b) {
            (Str(x), y) => Ok(Str(format!("{x}{}", display_value(y)))),
            (x, Str(y)) => Ok(Str(format!("{}{y}", display_value(x)))),
            (Int(x), Int(y)) => Ok(Int(x + y)),
            _ => num_op(as_f(a), as_f(b), |x, y| x + y),
        },
        BinOp::Sub => match (a, b) {
            (Int(x), Int(y)) => Ok(Int(x - y)),
            _ => num_op(as_f(a), as_f(b), |x, y| x - y),
        },
        BinOp::Mul => match (a, b) {
            (Int(x), Int(y)) => Ok(Int(x * y)),
            _ => num_op(as_f(a), as_f(b), |x, y| x * y),
        },
        BinOp::Div => num_op(as_f(a), as_f(b), |x, y| x / y),
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            let (x, y) = (as_f(a), as_f(b));
            match (x, y) {
                (Some(x), Some(y)) => Ok(Bool(match op {
                    BinOp::Lt => x < y,
                    BinOp::Gt => x > y,
                    BinOp::Le => x <= y,
                    BinOp::Ge => x >= y,
                    _ => unreachable!(),
                })),
                _ => Err(RunError("comparison requires numbers".into())),
            }
        }
        BinOp::Eq => Ok(Bool(values_eq(a, b))),
        BinOp::Ne => Ok(Bool(!values_eq(a, b))),
    }
}

fn num_op(a: Option<f64>, b: Option<f64>, f: impl Fn(f64, f64) -> f64) -> Result<Value, RunError> {
    match (a, b) {
        (Some(x), Some(y)) => Ok(Value::Float(f(x, y))),
        _ => Err(RunError("arithmetic requires numbers".into())),
    }
}

fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < f64::EPSILON,
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => {
            (*x as f64 - *y).abs() < f64::EPSILON
        }
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        _ => false,
    }
}

fn display_value(v: &Value) -> String {
    CValue { value: v.clone(), confidence: 1.0 }.display()
}

// ===========================================================================
// CIR / belief path
// ===========================================================================

/// Answer to a belief `inquire`. Mirrors ChimeraLang's `InquiryResponse`.
#[derive(Debug, Clone, PartialEq)]
pub struct InquiryResponse {
    pub confidence: f32,
    pub answer: Option<String>,
}

/// The seam that lets ChimeraLang beliefs be grounded in an external reasoner.
/// AXIOM injects an adapter backed by its own model/backend so `inquire`s are
/// answered by the engine itself (Phase 2 of the integration).
pub trait InquiryAdapter {
    fn inquire(&self, prompt: &str, agents: &[String]) -> InquiryResponse;
}

/// Default offline adapter: a fixed, honest-but-uncertain response. Mirrors
/// ChimeraLang's mock adapter (confidence 0.75) so programs run without a model.
pub struct MockAdapter;

impl InquiryAdapter for MockAdapter {
    fn inquire(&self, prompt: &str, _agents: &[String]) -> InquiryResponse {
        InquiryResponse {
            confidence: 0.75,
            answer: Some(format!("<mock answer for {prompt:?}>")),
        }
    }
}

/// Pseudocount strength used to convert an inquiry confidence into a Beta belief
/// (higher ⇒ more evidence ⇒ lower variance).
const INQUIRY_STRENGTH: f32 = 4.0;
/// Variance ceiling a belief must meet to pass a hallucination guard (matches
/// `crate::belief::ESTABLISHED_VARIANCE`).
const GUARD_VARIANCE: f32 = 0.05;

struct BeliefState {
    belief: BetaBelief,
    answer: Option<String>,
    agents: Vec<String>,
    prompt: String,
}

fn run_cir(program: &Program, adapter: &dyn InquiryAdapter) -> Result<RunResult, RunError> {
    let mut states: BTreeMap<String, BeliefState> = BTreeMap::new();
    let mut env: BTreeMap<String, CValue> = BTreeMap::new();
    let mut res = RunResult::default();

    for s in &program.stmts {
        match s {
            Stmt::Belief { name, prompt, agents, ttl } => {
                // Inquire each agent and fuse with Dempster-Shafer (the same
                // combine_ds AXIOM uses for swarm immunity). Single agent ⇒ just
                // that belief.
                let agent_list = if agents.is_empty() {
                    vec!["default".to_string()]
                } else {
                    agents.clone()
                };
                let mut fused: Option<BetaBelief> = None;
                let mut last_answer = None;
                for ag in &agent_list {
                    let r = adapter.inquire(prompt, std::slice::from_ref(ag));
                    last_answer = r.answer.clone();
                    let b = BetaBelief::from_confidence(r.confidence, INQUIRY_STRENGTH);
                    fused = Some(match fused {
                        None => b,
                        Some(prev) => prev.combine_ds(&b).unwrap_or(prev),
                    });
                }
                let mut belief = fused.unwrap_or_else(BetaBelief::uniform);
                // Temporal decay: a ttl of 0 means "no decay"; otherwise age the
                // belief slightly toward the uniform prior (staleness=uncertainty).
                if *ttl == 0 {
                    belief = belief.decayed(0.98);
                }
                res.trace.push(format!(
                    "belief {name}: mean {:.3} var {:.4}",
                    belief.mean(),
                    belief.variance()
                ));
                states.insert(
                    name.clone(),
                    BeliefState { belief, answer: last_answer, agents: agent_list, prompt: prompt.clone() },
                );
            }
            Stmt::Resolve { name, threshold } => {
                let st = states
                    .get(name)
                    .ok_or_else(|| RunError(format!("resolve: unknown belief `{name}`")))?;
                let mean = st.belief.mean();
                if mean < *threshold {
                    res.guard_violations.push(format!(
                        "resolve {name}: mean {mean:.3} below consensus threshold {threshold:.3}"
                    ));
                }
                res.trace.push(format!("resolve {name}: mean {mean:.3} (threshold {threshold:.3})"));
            }
            Stmt::Guard { name, max_risk } => {
                let st = states
                    .get(name)
                    .ok_or_else(|| RunError(format!("guard: unknown belief `{name}`")))?;
                let mean = st.belief.mean();
                let var = st.belief.variance();
                // Variance-aware check: a high mean with high uncertainty still
                // fails (ChimeraLang's guard semantics).
                let ok = mean >= (1.0 - *max_risk) && var <= GUARD_VARIANCE;
                if !ok {
                    res.guard_violations.push(format!(
                        "guard {name}: mean {mean:.3} (need ≥{:.3}), var {var:.4} (need ≤{GUARD_VARIANCE})",
                        1.0 - *max_risk
                    ));
                }
                res.trace.push(format!("guard {name}: {}", if ok { "pass" } else { "FAIL" }));
            }
            Stmt::Evolve { name, max_iter } => {
                let st = states
                    .get_mut(name)
                    .ok_or_else(|| RunError(format!("evolve: unknown belief `{name}`")))?;
                // Re-inquire and reinforce until the variance stops shrinking
                // meaningfully or we hit max_iter (convergence by uncertainty).
                let prompt = st.prompt.clone();
                let agents = st.agents.clone();
                let mut iters = 0u32;
                for _ in 0..*max_iter {
                    let before = st.belief.variance();
                    let r = adapter.inquire(&prompt, &agents);
                    let b = BetaBelief::from_confidence(r.confidence, INQUIRY_STRENGTH);
                    st.belief = st.belief.combine_ds(&b).unwrap_or(st.belief);
                    iters += 1;
                    if (before - st.belief.variance()).abs() < 1e-4 {
                        break;
                    }
                }
                res.trace.push(format!(
                    "evolve {name}: {iters} iter(s) → mean {:.3} var {:.4}",
                    st.belief.mean(),
                    st.belief.variance()
                ));
            }
            // `emit NAME` where NAME is a known belief → belief emit; otherwise
            // fall through to the plain VM emit (a normal variable / expression).
            Stmt::Emit(Expr::Ident(name)) if states.contains_key(name) => {
                let st = &states[name];
                let mean = st.belief.mean();
                res.beliefs.insert(name.clone(), mean);
                let ans = st.answer.clone().unwrap_or_default();
                res.emitted.push(format!("{name}={ans} (confidence {mean:.3})"));
            }
            // VM statements (val / emit / for) are allowed inside a belief
            // program too, sharing one environment.
            other => {
                exec_vm_stmt(other, &mut env, &mut res)?;
            }
        }
    }
    Ok(res)
}

// ===========================================================================
// Integrity certificate (reuses crate::provenance)
// ===========================================================================

/// Produce a tamper-evident certificate over a run result, reusing AXIOM's
/// provenance layer (SHA-256 + optional HMAC). This unifies ChimeraLang
/// certificates with AXIOM `SignedExport`s — one offline-verifiable format.
pub fn certify(result: &RunResult, fleet_key: Option<&[u8]>) -> SignedExport {
    let payload = serde_json::json!({
        "format": "chimeralang-cert/v2-rust",
        "emitted": result.emitted,
        "beliefs": result.beliefs,
        "guard_violations": result.guard_violations,
    })
    .to_string();
    sign_export(&payload, fleet_key)
}

/// Verify a certificate produced by [`certify`].
pub fn verify_certificate<'a>(
    cert: &'a SignedExport,
    fleet_key: Option<&[u8]>,
) -> Result<&'a str, ProvenanceError> {
    verify_export(cert, fleet_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_basic_tokens() {
        let t = lex("val x = 1 + 2.5").unwrap();
        assert_eq!(
            t,
            vec![
                Tok::Ident("val".into()),
                Tok::Ident("x".into()),
                Tok::Assign,
                Tok::Int(1),
                Tok::Plus,
                Tok::Float(2.5),
            ]
        );
    }

    #[test]
    fn vm_runs_val_emit_for() {
        let src = r#"
            val scores = [1, 2, 3]
            for s in scores
              emit s
            end
        "#;
        let prog = parse(src).unwrap();
        assert!(!prog.uses_belief);
        let res = run(&prog, &MockAdapter).unwrap();
        assert_eq!(res.emitted, vec!["1", "2", "3"]);
    }

    #[test]
    fn vm_string_concat_and_arithmetic() {
        let prog = parse(r#"emit "ans: " + (2 + 3)"#).unwrap();
        let res = run(&prog, &MockAdapter).unwrap();
        assert_eq!(res.emitted, vec!["ans: 5"]);
    }

    #[test]
    fn vm_confidence_member_access() {
        // A list literal's confidence is the min of its elements (all certain),
        // so .confidence is 1.0.
        let prog = parse("val xs = [1, 2]\nemit xs.confidence").unwrap();
        let res = run(&prog, &MockAdapter).unwrap();
        assert_eq!(res.emitted, vec!["1"]);
    }

    #[test]
    fn cir_belief_pipeline_runs_and_routes() {
        let src = r#"
            belief cause := inquire {
              prompt: "why do black holes form?",
              agents: [claude],
              ttl: 3600
            }
            resolve cause with consensus { threshold: 0.8 }
            guard cause against hallucination { max_risk: 0.2 }
            evolve cause until stable { max_iter: 3 }
            emit cause
        "#;
        let prog = parse(src).unwrap();
        assert!(prog.uses_belief, "belief construct must route to the CIR path");
        let res = run(&prog, &MockAdapter).unwrap();
        assert!(res.beliefs.contains_key("cause"));
        // The mock returns 0.75; after evolve reinforcement the mean should be
        // a sensible probability.
        let mean = res.beliefs["cause"];
        assert!((0.0..=1.0).contains(&mean));
        assert_eq!(res.emitted.len(), 1);
    }

    #[test]
    fn cir_guard_flags_low_confidence_belief() {
        // A low-confidence adapter must trip the hallucination guard.
        struct LowAdapter;
        impl InquiryAdapter for LowAdapter {
            fn inquire(&self, _p: &str, _a: &[String]) -> InquiryResponse {
                InquiryResponse { confidence: 0.30, answer: Some("unsure".into()) }
            }
        }
        let src = r#"
            belief weak := inquire { prompt: "?", agents: [x], ttl: 0 }
            guard weak against hallucination { max_risk: 0.1 }
            emit weak
        "#;
        let prog = parse(src).unwrap();
        let res = run(&prog, &LowAdapter).unwrap();
        assert!(
            !res.guard_violations.is_empty(),
            "a low-confidence belief must violate a strict guard"
        );
    }

    #[test]
    fn certificate_roundtrips_and_detects_tampering() {
        let prog = parse("emit \"hello\"").unwrap();
        let res = run(&prog, &MockAdapter).unwrap();
        let key: &[u8] = b"fleet";
        let cert = certify(&res, Some(key));
        assert!(verify_certificate(&cert, Some(key)).is_ok());

        let mut tampered = cert.clone();
        tampered.payload.push_str("{\"injected\":true}");
        assert!(verify_certificate(&tampered, Some(key)).is_err());
    }

    #[test]
    fn parse_error_is_reported() {
        assert!(parse("val = 3").is_err());
    }
}
