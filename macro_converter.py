"""
macro_converter.py
──────────────────
Hybrid SAS Macro → R Function Converter  (v9 — robust rewrite)

Architecture:
    SAS Macro
        ↓
    MacroIR (Intermediate Representation)
        ↓
    ComplexityScorer
        ↓
    HIGH confidence  → RuleBasedConverter  (FREE, deterministic)
    LOW confidence   → LLMConverter        (COST, fallback only)
        ↓
    ConversionCache  (reuse identical macros)
        ↓
    Reusable R Functions

Design principles:
    - Deterministic first, LLM last
    - Cache everything
    - Confidence scoring at every step
    - Modular — each component replaceable
    - AST-ready IR for future compiler evolution

Changes v9 (robustness fixes):
    - All regexes: possessive quantifiers replaced with atomic groups / explicit
      non-greedy to avoid catastrophic backtracking on large inputs
    - MacroParser: added %LET, CALL SYMPUT, single-line %IF, PROC SORT options
      (NODUPKEY/NODUPRECS), DATA step WHERE/KEEP/DROP/RENAME/MERGE,
      %DO %WHILE / %DO %UNTIL stubs
    - MacroParser: statement deduplication by raw-text span to prevent double-
      parsing when multiple regexes match the same block
    - RuleBasedConverter._proc_means: fixed !!sym() quoting — column names are
      already strings so use sym("col"), not !!sym({v})
    - RuleBasedConverter._proc_freq: fixed character-class regex [\\s*] → \\s+
    - RuleBasedConverter._proc_sql: WHERE = → == translation now preserves
      <=, >=, != by only replacing bare = (negative look-around)
    - RuleBasedConverter._data_step: handles WHERE clause; compound conditions;
      KEEP/DROP lists; RENAME map
    - RuleBasedConverter._if_else / _do_loop: recursively converts body lines
      instead of emitting a comment stub
    - ComplexityScorer: moved %if/%then from SIMPLE to COMPLEX (it was already
      positive-weight there, the label was wrong)
    - ConversionCache._make_key: stable key via json.dumps(sorted params)
    - HybridMacroConverter.convert_all: variable name collision fixed
      (outer `macro_calls` list vs inner `macro_calls_in_body`)
    - SAS operator translation: centralised _sas_cond_to_r() helper used by
      all converters for consistent AND/OR/NOT/IN/EQ/NE/GT/LT/GE/LE/^= maps
"""

import re
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _sas_cond_to_r(cond: str, params: Optional[list] = None) -> str:
    """
    Translate a SAS condition string to R.
    Handles: NE GT LT GE LE EQ AND OR NOT IN ^= ~= operators.
    Replaces &param macro references with bare param names.
    """
    r = cond.strip()

    # macro variable references  &var  →  var
    r = re.sub(r'&(\w+)', lambda m: m.group(1).lower(), r)

    # SAS word operators (order matters: longer first)
    replacements = [
        (r'\bNOT\b',              '!',     re.IGNORECASE),
        (r'\bAND\b',              '&&',    re.IGNORECASE),
        (r'\bOR\b',               '||',    re.IGNORECASE),
        (r'\bEQ\b',               '==',    re.IGNORECASE),
        (r'\bNE\b',               '!=',    re.IGNORECASE),
        (r'\bGT\b',               '>',     re.IGNORECASE),
        (r'\bLT\b',               '<',     re.IGNORECASE),
        (r'\bGE\b',               '>=',    re.IGNORECASE),
        (r'\bLE\b',               '<=',    re.IGNORECASE),
        (r'\^=',                  '!=',    0),
        (r'~=',                   '!=',    0),
        # bare = that is NOT already part of <=, >=, !=, ==
        (r'(?<![<>!=])=(?!=)',    '==',    0),
    ]
    for pattern, repl, flags in replacements:
        r = re.sub(pattern, repl, r, flags=flags) if flags else re.sub(pattern, repl, r)

    # SAS IN operator:  var in (a, b, c)  →  var %in% c(a, b, c)
    r = re.sub(
        r'(\w+)\s+in\s*\((.*?)\)',
        lambda m: f'{m.group(1)} %in% c({m.group(2)})',
        r, flags=re.IGNORECASE
    )

    return r


def _strip_macro_ref(name: str) -> str:
    """Remove leading & or % from a SAS name reference."""
    return name.lstrip('&%').lower()


# ─────────────────────────────────────────────────────────────────
# INTERMEDIATE REPRESENTATION (IR)
# ─────────────────────────────────────────────────────────────────

@dataclass
class MacroStatement:
    """Single statement inside a macro body."""
    kind: str          # 'proc_sort' | 'proc_means' | 'proc_freq' |
                       # 'proc_sql'  | 'data_step'  | 'if_else'   |
                       # 'do_loop'   | 'let'        | 'call_symput'|
                       # 'proc_transpose' | 'unknown'
    raw:  str          # original SAS text
    attrs: dict = field(default_factory=dict)  # parsed attributes
    span: tuple = field(default_factory=lambda: (0, 0))  # (start, end) in body


@dataclass
class MacroIR:
    """Intermediate Representation of one SAS macro."""
    name:       str
    params:     list
    body_raw:   str
    statements: list = field(default_factory=list)
    complexity: int   = 0       # computed score
    confidence: float = 0.0     # rule-based confidence 0.0-1.0


# ─────────────────────────────────────────────────────────────────
# CONVERSION CACHE
# ─────────────────────────────────────────────────────────────────

class ConversionCache:
    """
    In-memory + optional JSON-file cache.
    Key = SHA256(macro_name + sorted_params + body + dialect)
    """

    def __init__(self, cache_file: Optional[str] = None):
        self._mem: dict = {}
        self._file = cache_file
        if cache_file:
            self._load()

    def _make_key(self, ir: MacroIR, dialect: str) -> str:
        # Use json.dumps with sorted params for a stable key regardless of
        # Python version or list identity
        raw = json.dumps(
            {"name": ir.name, "params": sorted(ir.params),
             "body": ir.body_raw, "dialect": dialect},
            sort_keys=True
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, ir: MacroIR, dialect: str) -> Optional[dict]:
        key = self._make_key(ir, dialect)
        return self._mem.get(key)

    def put(self, ir: MacroIR, dialect: str, result: dict):
        key = self._make_key(ir, dialect)
        self._mem[key] = result
        if self._file:
            self._save()

    def _load(self):
        try:
            with open(self._file, 'r') as f:
                self._mem = json.load(f)
        except Exception:
            self._mem = {}

    def _save(self):
        try:
            with open(self._file, 'w') as f:
                json.dump(self._mem, f, indent=2)
        except Exception:
            pass

    @property
    def size(self) -> int:
        return len(self._mem)


# ─────────────────────────────────────────────────────────────────
# COMPLEXITY SCORER
# ─────────────────────────────────────────────────────────────────

class ComplexityScorer:
    """
    Scores macro complexity to decide rule-based vs LLM.
    Score 0-10:  rule-based handles it
    Score 11+:   LLM needed
    """

    # Patterns that INCREASE complexity (harder to parse)
    COMPLEX = [
        (r'call\s+symput',          8,  "CALL SYMPUT — dynamic macro var creation"),
        (r'%sysfunc\s*\(',          7,  "SYSFUNC — system function call"),
        (r'%scan\s*\(',             5,  "SCAN — string parsing"),
        (r'%substr\s*\(',           4,  "SUBSTR — substring"),
        (r'proc\s+sql',             3,  "PROC SQL — handled but complex"),
        (r'proc\s+report',          7,  "PROC REPORT"),
        (r'proc\s+tabulate',        7,  "PROC TABULATE"),
        (r'%do\s+%while',           6,  "%DO %WHILE loop"),
        (r'%do\s+%until',           6,  "%DO %UNTIL loop"),
        (r'%syscall',               6,  "SYSCALL"),
        (r'proc\s+iml',             9,  "PROC IML — matrix language"),
        (r'\barray\s+\w+',          5,  "ARRAY statement"),
        (r'\bretain\s+',            4,  "RETAIN statement"),
        (r'\blag\s*\(',             4,  "LAG function"),
        (r'\binfile\s+',            6,  "INFILE — external data"),
        (r'%if\s+.*?%then',         3,  "%IF/%THEN — conditional"),
        (r'%do\s+\w+\s*=\s*',       3,  "%DO numeric loop"),
    ]

    # Patterns that DECREASE complexity (easy to rule-convert)
    SIMPLE = [
        (r'proc\s+sort',            -3, "PROC SORT — simple"),
        (r'proc\s+means',           -2, "PROC MEANS — simple"),
        (r'proc\s+freq',            -2, "PROC FREQ — simple"),
        (r'data\s+\w+;\s*set\s+',   -2, "DATA step SET — simple"),
        (r'proc\s+transpose',       -1, "PROC TRANSPOSE — handled"),
    ]

    THRESHOLD = 10  # score above this → LLM

    def score(self, ir: MacroIR) -> tuple:
        """
        Returns (score, confidence, reasons).
        confidence = 1.0 means rule-based is fully reliable.
        """
        score = 0
        reasons = []
        body = ir.body_raw.lower()

        for pattern, weight, reason in self.COMPLEX:
            if re.search(pattern, body, re.IGNORECASE):
                score += weight
                reasons.append(f"+{weight} {reason}")

        for pattern, weight, reason in self.SIMPLE:
            if re.search(pattern, body, re.IGNORECASE):
                score += weight
                reasons.append(f"{weight:+d} {reason}")

        score = max(0, score)
        # confidence inversely proportional to score
        confidence = max(0.0, 1.0 - (score / (self.THRESHOLD * 2)))

        ir.complexity = score
        ir.confidence = confidence
        return score, confidence, reasons


# ─────────────────────────────────────────────────────────────────
# MACRO PARSER → IR
# ─────────────────────────────────────────────────────────────────

class MacroParser:
    """Parses SAS macro text into MacroIR."""

    # Builtins that are not user-defined macro calls
    _MACRO_BUILTINS = frozenset({
        'IF', 'THEN', 'ELSE', 'DO', 'END', 'LET', 'PUT', 'MEND', 'MACRO',
        'GLOBAL', 'LOCAL', 'SYSFUNC', 'SCAN', 'SUBSTR', 'UPCASE', 'LOWCASE',
        'TRIM', 'LEFT', 'RIGHT', 'LENGTH', 'INDEX', 'QUOTE', 'NRQUOTE',
        'STR', 'NRSTR', 'BQUOTE', 'NRBQUOTE', 'SUPERQ', 'EVAL', 'SYSEVAL',
        'QSCAN', 'QSUBSTR', 'QLEFT', 'QTRIM', 'RETURN', 'GOTO', 'ABORT',
    })

    def parse(self, name: str, params: list, body: str) -> MacroIR:
        ir = MacroIR(name=name, params=params, body_raw=body)
        ir.statements = self._parse_statements(body)
        return ir

    @staticmethod
    def _clean(v):
        """Strip & prefix and lowercase."""
        if isinstance(v, str):
            return v.lstrip('&%').lower().strip()
        if isinstance(v, list):
            return [i.lstrip('&%').lower().strip() for i in v]
        return v

    def _parse_statements(self, body: str) -> list:
        stmts = []
        used_spans: set = set()   # (start, end) — prevent double-parsing

        def _register(m, kind, attrs):
            """Add statement only if its span hasn't already been claimed."""
            start, end = m.span()
            # Check for overlap with any already-claimed span
            for us, ue in used_spans:
                if start < ue and end > us:  # overlap
                    return
            used_spans.add((start, end))
            stmts.append(MacroStatement(
                kind=kind, raw=m.group(0), attrs=attrs, span=(start, end)
            ))

        clean = self._clean

        # ── %LET ────────────────────────────────────────────────
        for m in re.finditer(
            r'%let\s+(\w+)\s*=\s*([^;]*?)\s*;',
            body, re.IGNORECASE
        ):
            _register(m, 'let', {
                'var':   m.group(1).lower(),
                'value': m.group(2).strip(),
            })

        # ── CALL SYMPUT ─────────────────────────────────────────
        for m in re.finditer(
            r'call\s+symput\s*\(\s*(["\']?)(\w+)\1\s*,\s*([^)]+)\)',
            body, re.IGNORECASE
        ):
            _register(m, 'call_symput', {
                'var':   m.group(2).lower(),
                'value': m.group(3).strip(),
            })

        # ── PROC SORT ───────────────────────────────────────────
        for m in re.finditer(
            r'proc\s+sort(?:\s+data\s*=\s*&?(\w+))?((?:[^;])*?)\s*;'
            r'(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            inner = m.group(3) or ''
            by_m = re.search(r'\bby\s+(.*?);', inner, re.IGNORECASE)
            opts_raw = (m.group(2) or '').lower()
            out_m = re.search(r'\bout\s*=\s*&?(\w+)', opts_raw, re.IGNORECASE)
            _register(m, 'proc_sort', {
                'input':      clean(m.group(1) or ''),
                'output':     clean(out_m.group(1) if out_m else (m.group(1) or '')),
                'by_vars':    clean(by_m.group(1).split()) if by_m else [],
                'nodupkey':   bool(re.search(r'\bnodupkey\b', opts_raw)),
                'noduprecs':  bool(re.search(r'\bnoduprecs\b', opts_raw)),
            })

        # ── PROC MEANS ──────────────────────────────────────────
        for m in re.finditer(
            r'proc\s+means\s+data\s*=\s*&?(\w+)([^;]*?);(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            opts  = m.group(2)
            inner = m.group(3)
            class_m = re.search(r'\bclass\s+(.*?);', inner, re.IGNORECASE)
            var_m   = re.search(r'\bvar\s+(.*?);',   inner, re.IGNORECASE)
            out_m   = re.search(r'\boutput\s+out\s*=\s*&?(\w+)([^;]*?);',
                                 inner, re.IGNORECASE)
            _register(m, 'proc_means', {
                'input':     clean(m.group(1)),
                'class_var': clean(class_m.group(1).split()) if class_m else [],
                'var':       clean(var_m.group(1).split()) if var_m else [],
                'output':    clean(out_m.group(1)) if out_m else None,
                'stats':     self._parse_means_stats(opts),
            })

        # ── PROC FREQ ───────────────────────────────────────────
        for m in re.finditer(
            r'proc\s+freq\s+data\s*=\s*&?(\w+)\s*;(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            tables_m = re.search(r'\btables\s+(.*?);', m.group(2), re.IGNORECASE)
            _register(m, 'proc_freq', {
                'input':  clean(m.group(1)),
                'tables': clean(tables_m.group(1).strip()) if tables_m else '',
            })

        # ── DATA STEP (simple set / merge) ──────────────────────
        for m in re.finditer(
            r'data\s+&?(\w+)\s*;(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            ds_body = m.group(2)
            out_ds  = clean(m.group(1))

            # SET with optional WHERE
            set_m = re.search(
                r'\bset\s+(&?\w+(?:\s+&?\w+)*)\s*(?:;|\(where\s*=\s*\((.*?)\)\);?)',
                ds_body,
                re.IGNORECASE
            )
            # MERGE
            merge_m = re.search(r'\bmerge\s+(.*?);', ds_body, re.IGNORECASE)
            by_m    = re.search(r'\bby\s+(.*?);',    ds_body, re.IGNORECASE)
            where_m = re.search(r'\bwhere\s+(.*?);', ds_body, re.IGNORECASE)
            keep_m  = re.search(r'\bkeep\s+(.*?);',  ds_body, re.IGNORECASE)
            drop_m  = re.search(r'\bdrop\s+(.*?);',  ds_body, re.IGNORECASE)
            rename_m= re.search(r'\brename\s+(.*?);',ds_body, re.IGNORECASE)

            # Simple assignments
            assigns = re.findall(r'(\w+)\s*=\s*([^;]+);', ds_body)
            assigns = [
                (v.lstrip('&'), e.strip().replace('&', ''))
                for v, e in assigns
                if v.lower() not in ('data', 'set', 'merge', 'by', 'where',
                                     'keep', 'drop', 'rename', 'run')
            ]

            # IF (data step, not macro) filter
            if_filters = re.findall(
                r'\bif\s+(.*?)\s*(?:then\s+(?:output|delete)\s*;|;)',
                ds_body, re.IGNORECASE
            )

            if set_m or merge_m:
                _register(m, 'data_step', {
                    'output':  out_ds,
                    'input':   clean(set_m.group(1).split()[0]) if set_m else '',
                    'inputs':  clean(set_m.group(1).split()) if set_m else
                               (clean(merge_m.group(1).split()) if merge_m else []),
                    'is_merge': bool(merge_m),
                    'by_vars': clean(by_m.group(1).split()) if by_m else [],
                    'where':   (where_m.group(1) or
                                (set_m.group(2) if set_m and set_m.group(2) else '')
                               ).strip(),
                    'keep':    clean(keep_m.group(1).split()) if keep_m else [],
                    'drop':    clean(drop_m.group(1).split()) if drop_m else [],
                    'rename':  self._parse_rename(rename_m.group(1)) if rename_m else {},
                    'assigns': assigns,
                    'if_filters': if_filters,
                    'body':    ds_body.strip(),
                })

        # ── PROC TRANSPOSE ──────────────────────────────────────
        for m in re.finditer(
            r'proc\s+transpose\s+data\s*=\s*&?(\w+)(?:\s+out\s*=\s*&?(\w+))?'
            r'[^;]*;(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            inner = m.group(3)
            var_m = re.search(r'\bvar\s+(.*?);', inner, re.IGNORECASE)
            by_m  = re.search(r'\bby\s+(.*?);',  inner, re.IGNORECASE)
            id_m  = re.search(r'\bid\s+(.*?);',   inner, re.IGNORECASE)
            _register(m, 'proc_transpose', {
                'input':  clean(m.group(1)),
                'output': clean(m.group(2) or (m.group(1) + '_t')),
                'var':    clean(var_m.group(1).split()) if var_m else [],
                'by':     clean(by_m.group(1).split()) if by_m else [],
                'id':     clean(id_m.group(1).strip()) if id_m else '',
            })

        # ── PROC SQL ────────────────────────────────────────────
        for m in re.finditer(
            r'proc\s+sql\s*;(.*?)quit\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            sql_body = m.group(1)
            create_m = re.search(
                r'create\s+table\s+&?(\w+)\s+as\s+select\s+(.*?)\s+from\s+&?(\w+)'
                r'(?:\s+where\s+(.*?))?(?:\s+group\s+by\s+(.*?))?'
                r'(?:\s+order\s+by\s+(.*?))?\s*;',
                sql_body, re.IGNORECASE | re.DOTALL
            )
            if create_m:
                _register(m, 'proc_sql', {
                    'output':   clean(create_m.group(1)),
                    'select':   create_m.group(2).strip(),
                    'input':    clean(create_m.group(3)),
                    'where':    (create_m.group(4) or '').strip(),
                    'group_by': (create_m.group(5) or '').strip(),
                    'order_by': (create_m.group(6) or '').strip(),
                })
            else:
                # Fallback: capture as unknown SQL block
                _register(m, 'proc_sql', {
                    'output': '', 'select': '', 'input': '',
                    'where': '', 'group_by': '', 'order_by': '',
                    'raw_sql': sql_body.strip(),
                })

        # ── %IF / %THEN (block form) ─────────────────────────────
        for m in re.finditer(
            r'%if\s+(.*?)\s*%then\s*%do\s*;(.*?)%end\s*;'
            r'(?:\s*%else\s*%do\s*;(.*?)%end\s*;)?',
            body, re.IGNORECASE | re.DOTALL
        ):
            _register(m, 'if_else', {
                'condition':  m.group(1).strip(),
                'then_block': m.group(2).strip(),
                'else_block': (m.group(3) or '').strip(),
                'inline':     False,
            })

        # ── %IF / %THEN (single-line form, no %do) ──────────────
        for m in re.finditer(
            r'%if\s+(.*?)\s*%then\s+(?!%do)(.*?);'
            r'(?:\s*%else\s+(?!%do)(.*?);)?',
            body, re.IGNORECASE
        ):
            _register(m, 'if_else', {
                'condition':  m.group(1).strip(),
                'then_block': m.group(2).strip(),
                'else_block': (m.group(3) or '').strip(),
                'inline':     True,
            })

        # ── %DO numeric loop (literal bounds) ───────────────────
        for m in re.finditer(
            r'%do\s+(\w+)\s*=\s*(&?\w+)\s*%to\s*(&?\w+)'
            r'(?:\s*%by\s*(-?\w+))?\s*;(.*?)%end\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            start_raw = m.group(2).lstrip('&')
            end_raw   = m.group(3).lstrip('&')
            step_raw  = (m.group(4) or '1').lstrip('&')
            _register(m, 'do_loop', {
                'var':        m.group(1).lower(),
                'start':      start_raw,   # may be a string (macro var name)
                'end':        end_raw,
                'step':       step_raw,
                'body':       m.group(5).strip(),
                'is_literal': start_raw.isdigit() and end_raw.isdigit(),
            })

        # ── %DO %WHILE / %DO %UNTIL (stub) ──────────────────────
        for m in re.finditer(
            r'%do\s+%(?:while|until)\s*\((.*?)\)\s*;(.*?)%end\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            _register(m, 'do_while', {
                'condition': m.group(1).strip(),
                'body':      m.group(2).strip(),
                'kind':      'while' if 'while' in m.group(0).lower() else 'until',
            })

        # ── Fallback: unknown ────────────────────────────────────
        if not stmts:
            stmts.append(MacroStatement(kind='unknown', raw=body, span=(0, len(body))))

        # Sort by position in source
        stmts.sort(key=lambda s: s.span[0])
        return stmts

    @staticmethod
    def _parse_means_stats(opts: str) -> list:
        known = ['mean', 'std', 'min', 'max', 'median', 'n', 'sum', 'var']
        found = [s for s in known if re.search(rf'\b{s}\b', opts, re.IGNORECASE)]
        return found or ['mean', 'std']

    @staticmethod
    def _parse_rename(rename_str: str) -> dict:
        """Parse  old1=new1 old2=new2  into {old: new}."""
        pairs = re.findall(r'(\w+)\s*=\s*(\w+)', rename_str)
        return {old.lower(): new.lower() for old, new in pairs}


# ─────────────────────────────────────────────────────────────────
# RULE-BASED R FUNCTION GENERATOR
# ─────────────────────────────────────────────────────────────────

class RuleBasedConverter:
    """
    Converts MacroIR → R function using deterministic rules.
    Returns (r_code, confidence).
    """

    def convert(self, ir: MacroIR, dialect: str = "Modern R (dplyr)") -> tuple:
        func_name    = ir.name.lower()
        params_lower = [p.lower() for p in ir.params]
        params_r     = ", ".join(params_lower)
        body_lines   = []
        total_conf   = 1.0

        for stmt in ir.statements:
            r_lines, conf = self._convert_statement(stmt, params_lower, dialect)
            body_lines.extend(r_lines)
            total_conf = min(total_conf, conf)

        # Add return statement for last assigned result variable
        last_result = None
        for ln in reversed(body_lines):
            m = re.match(r'\s*(\w+)\s*<-', ln)
            if m:
                last_result = m.group(1)
                break
        if last_result:
            body_lines.append(f"return({last_result})")

        body = "\n".join(f"  {ln}" for ln in body_lines if ln.strip())

        r_func = (
            f"# SAS macro %{ir.name} converted to R function\n"
            f"{func_name} <- function({params_r}) {{\n"
            f"{body}\n"
            f"}}\n"
        )

        call_args = ", ".join(f'{p} = <value>' for p in params_lower)
        r_func += f"\n# Example call:\n# {func_name}({call_args})\n"

        return r_func, total_conf

    def _convert_statement(self, stmt: MacroStatement, params: list, dialect: str) -> tuple:
        dispatch = {
            'proc_sort':      self._proc_sort,
            'proc_means':     self._proc_means,
            'proc_freq':      self._proc_freq,
            'data_step':      self._data_step,
            'proc_sql':       self._proc_sql,
            'proc_transpose': self._proc_transpose,
            'if_else':        self._if_else,
            'do_loop':        self._do_loop,
            'do_while':       self._do_while,
            'let':            self._let_stmt,
            'call_symput':    self._call_symput,
        }
        handler = dispatch.get(stmt.kind)
        if handler:
            if stmt.kind in ('if_else', 'do_loop', 'do_while', 'let', 'call_symput'):
                return handler(stmt, params, dialect)
            return handler(stmt, dialect)

        # unknown — check for nested macro calls
        macro_calls = re.findall(r'%(\w+)\s*\(([^)]*)\)', stmt.raw, re.IGNORECASE)
        macro_calls = [
            (n, a) for n, a in macro_calls
            if n.upper() not in MacroParser._MACRO_BUILTINS
        ]
        if macro_calls:
            lines = []
            for call_name, call_args in macro_calls:
                r_args = []
                for arg in call_args.split(','):
                    arg = arg.strip()
                    if '=' in arg:
                        k, v = arg.split('=', 1)
                        r_args.append(
                            f"{k.strip().lstrip('&').lower()} = "
                            f"{v.strip().lstrip('&').lower()}"
                        )
                    elif arg:
                        r_args.append(arg.lstrip('&').lower())
                lines.append(f"{call_name.lower()}({', '.join(r_args)})")
            return lines, 0.75

        snippet = stmt.raw[:100].replace('\n', ' ')
        return [f"# TODO: Convert manually:\n  # {snippet}"], 0.2

    # ── %LET ────────────────────────────────────────────────────
    def _let_stmt(self, stmt: MacroStatement, params: list, dialect: str) -> tuple:
        var   = stmt.attrs['var']
        value = stmt.attrs['value'].lstrip('&')
        # Try to detect if numeric
        try:
            float(value)
            return [f"{var} <- {value}"], 0.90
        except ValueError:
            # String
            return [f"{var} <- \"{value}\""], 0.85

    # ── CALL SYMPUT ─────────────────────────────────────────────
    def _call_symput(self, stmt: MacroStatement, params: list, dialect: str) -> tuple:
        var   = stmt.attrs['var']
        value = stmt.attrs['value'].strip()
        # CALL SYMPUT creates a macro var from a data step value
        # Best we can do: assign to an R variable
        lines = [
            f"# CALL SYMPUT: macro var '{var}' set from data step value",
            f"{var} <- {_sas_cond_to_r(value, params)}",
        ]
        return lines, 0.60

    # ── PROC SORT ───────────────────────────────────────────────
    def _proc_sort(self, stmt: MacroStatement, dialect: str) -> tuple:
        inp       = stmt.attrs['input']
        out       = stmt.attrs['output']
        by_vars   = stmt.attrs['by_vars']
        nodupkey  = stmt.attrs.get('nodupkey', False)
        noduprecs = stmt.attrs.get('noduprecs', False)

        # Detect descending prefix
        desc_vars = []
        clean_by  = []
        i = 0
        while i < len(by_vars):
            if by_vars[i].lower() == 'descending' and i + 1 < len(by_vars):
                desc_vars.append(by_vars[i + 1])
                clean_by.append(by_vars[i + 1])
                i += 2
            else:
                clean_by.append(by_vars[i])
                i += 1

        if dialect == "Modern R (dplyr)":
            arrange_args = [
                f'desc(.data[["{v}"]])' if v in desc_vars else f'.data[["{v}"]]'
                for v in clean_by
            ]
            lines = [
                f"{out} <- {inp} %>%",
                f"  arrange({', '.join(arrange_args)})",
            ]
            if nodupkey or noduprecs:
                dup_cols = ', '.join(f'"{v}"' for v in clean_by) if nodupkey else 'everything()'
                lines[-1] += " %>%"
                lines.append(f"  distinct({dup_cols}, .keep_all = TRUE)")
        else:
            order_args = [
                f'-{inp}[["{v}"]]' if v in desc_vars else f'{inp}[["{v}"]]'
                for v in clean_by
            ]
            lines = [
                f"{out} <- {inp}[order({', '.join(order_args)}), ]",
            ]
            if nodupkey or noduprecs:
                dup_cols = ', '.join(f'"{v}"' for v in clean_by) if nodupkey else 'NULL'
                lines.append(f"{out} <- {out}[!duplicated({out}[, c({dup_cols})]), ]")

        return lines, 0.95

    # ── PROC MEANS ──────────────────────────────────────────────
    def _proc_means(self, stmt: MacroStatement, dialect: str) -> tuple:
        inp      = stmt.attrs['input']
        grp_vars = stmt.attrs['class_var']
        num_vars = stmt.attrs['var']
        out      = stmt.attrs['output'] or f"{inp}_means"
        stats    = stmt.attrs['stats']

        # dplyr: use across() for multiple vars to avoid !!sym() quoting issues
        if dialect == "Modern R (dplyr)":
            stat_fns = {
                'mean':   'mean',
                'std':    'sd',
                'min':    'min',
                'max':    'max',
                'median': 'median',
                'n':      'length',
                'sum':    'sum',
                'var':    'var',
            }
            lines = [f"{out} <- {inp} %>%"]
            if grp_vars:
                grp_list = ', '.join(f'"{g}"' for g in grp_vars)
                lines.append(f"  group_by(across(all_of(c({grp_list})))) %>%")
            if num_vars:
                cols_list = ', '.join(f'"{v}"' for v in num_vars)
                fn_list   = ', '.join(
                    f'{s} = ~{stat_fns.get(s, s)}(., na.rm=TRUE)'
                    for s in stats if s != 'n'
                )
                n_part = ", n = ~length(.)" if 'n' in stats else ""
                lines.append(
                    f"  summarise(across(all_of(c({cols_list})), "
                    f"list({fn_list}{n_part})), .groups='drop')"
                )
            else:
                lines.append("  summarise(across(where(is.numeric), list("
                             + ', '.join(f'{s} = ~mean(., na.rm=TRUE)' for s in stats)
                             + ")), .groups='drop')")
        else:
            fun_map = {
                'mean': 'mean', 'std': 'sd', 'min': 'min',
                'max': 'max', 'median': 'median', 'n': 'length', 'sum': 'sum'
            }
            lines = []
            agg_dfs = []
            for v in num_vars:
                for s in stats:
                    agg_name = f"agg_{s}_{v}"
                    fun      = fun_map.get(s, 'mean')
                    if grp_vars:
                        lines.append(
                        f"{agg_name} <- aggregate("
                        f"as.formula(paste('{v}', '~', grp)), "
                        f"data={inp}, FUN={fun}, na.rm=TRUE)"
                        )
                        lines.append(f"names({agg_name})[ncol({agg_name})] <- '{s}_{v}'")
                    else:
                        lines.append(
                            f"{agg_name} <- data.frame(`{s}_{v}` = "
                            f"{fun}({inp}[['{v}']], na.rm=TRUE))"
                        )
                    agg_dfs.append(agg_name)
            if len(agg_dfs) > 1:
                by_cols = "grp" if grp_vars else "NULL"
                lines.append(
                    f"{out} <- Reduce(function(a,b) merge(a, b, by={by_cols}),"
                    f" list({', '.join(agg_dfs)}))"
                )
            elif agg_dfs:
                lines.append(f"{out} <- {agg_dfs[0]}")

        return lines, 0.88

    # ── PROC FREQ ───────────────────────────────────────────────
    def _proc_freq(self, stmt: MacroStatement, dialect: str) -> tuple:
        inp    = stmt.attrs['input']
        tables = stmt.attrs['tables']
        # FIX: was [\\s*] (wrong char class), now split on whitespace/asterisk
        vars_  = [v.strip() for v in re.split(r'[\s*]+', tables) if v.strip()]
        out    = f"{inp}_freq"

        if dialect == "Modern R (dplyr)":
            grp_list = ', '.join(f'"{v}"' for v in vars_)
            lines = [
                f"{out} <- {inp} %>%",
                f"  group_by(across(all_of(c({grp_list})))) %>%",
                f"  summarise(COUNT = n(), .groups='drop')",
            ]
        else:
            tbl_args = ', '.join(f'{inp}[["{v}"]]' for v in vars_)
            lines = [
                f"{out} <- as.data.frame(table({', '.join(f'{inp}[[\"{v}\"]]' for v in vars_)}))",
                f"names({out}) <- c({', '.join(repr(v) for v in vars_)}, 'COUNT')",
                f"{out} <- {out}[{out}$COUNT > 0, ]",
            ]

        return lines, 0.92

    # ── DATA STEP ───────────────────────────────────────────────
    def _data_step(self, stmt: MacroStatement, dialect: str) -> tuple:
        inp      = stmt.attrs['input']
        out      = stmt.attrs['output']
        assigns  = stmt.attrs.get('assigns', [])
        where    = stmt.attrs.get('where', '')
        keep     = stmt.attrs.get('keep', [])
        drop     = stmt.attrs.get('drop', [])
        rename   = stmt.attrs.get('rename', {})
        is_merge = stmt.attrs.get('is_merge', False)
        inputs   = stmt.attrs.get('inputs', [inp] if inp else [])
        by_vars  = stmt.attrs.get('by_vars', [])
        if_filt  = stmt.attrs.get('if_filters', [])

        lines = []
        conf  = 0.85

        if dialect == "Modern R (dplyr)":
            if is_merge and len(inputs) >= 2:
                by_cols = ', '.join(f'"{v}"' for v in by_vars) if by_vars else None
                if by_cols:
                    lines.append(
                        f"{out} <- merge({inputs[0]}, {inputs[1]},"
                        f" by=c({by_cols}), all=TRUE)"
                    )
                else:
                    lines.append(f"{out} <- bind_rows({', '.join(inputs)})")
                conf = 0.80
            elif inp:
                lines.append(f"{out} <- {inp}")
            else:
                lines.append(f"{out} <- data.frame()")
                conf = 0.40

            # WHERE / IF filter
            filter_cond = where or (if_filt[0] if if_filt else '')
            if filter_cond:
                r_cond = _sas_cond_to_r(filter_cond)
                lines[-1] += " %>%"
                lines.append(f"  filter({r_cond})")

            # Assignments / mutate
            if assigns:
                mutate_parts = []
                for v, e in assigns:
                    e_r = _sas_cond_to_r(e)
                    mutate_parts.append(f"{v} = {e_r}")
                lines[-1] += " %>%"
                lines.append(f"  mutate({', '.join(mutate_parts)})")

            # KEEP → select
            if keep:
                cols = ', '.join(f'"{c}"' for c in keep)
                lines[-1] += " %>%"
                lines.append(f"  select(all_of(c({cols})))")

            # DROP → select with minus
            if drop:
                cols = ', '.join(f'"{c}"' for c in drop)
                lines[-1] += " %>%"
                lines.append(f"  select(-all_of(c({cols})))")

            # RENAME → rename()
            if rename:
                ren_parts = ', '.join(f'"{new}" = "{old}"' for old, new in rename.items())
                lines[-1] += " %>%"
                lines.append(f"  rename({ren_parts})")

        else:  # Base R
            if is_merge and len(inputs) >= 2:
                by_cols = ', '.join(f'"{v}"' for v in by_vars) if by_vars else None
                if by_cols:
                    lines.append(
                        f"{out} <- merge({inputs[0]}, {inputs[1]},"
                        f" by=c({by_cols}), all=TRUE)"
                    )
                else:
                    lines.append(f"{out} <- rbind({', '.join(inputs)})")
            elif inp:
                lines.append(f"{out} <- {inp}")
            else:
                lines.append(f"{out} <- data.frame()")

            filter_cond = where or (if_filt[0] if if_filt else '')
            if filter_cond:
                r_cond = _sas_cond_to_r(filter_cond)
                lines.append(f"{out} <- {out}[{r_cond}, ]")

            for v, e in assigns:
                e_r = _sas_cond_to_r(e)
                lines.append(f'{out}[["{v}"]] <- {e_r}')

            if keep:
                cols = ', '.join(f'"{c}"' for c in keep)
                lines.append(f"{out} <- {out}[, c({cols})]")
            if drop:
                cols = ', '.join(f'"{c}"' for c in drop)
                lines.append(f"{out} <- {out}[, !names({out}) %in% c({cols})]")
            if rename:
                for old, new in rename.items():
                    lines.append(
                        f'names({out})[names({out}) == "{old}"] <- "{new}"'
                    )

        return lines, conf

    # ── PROC SQL ────────────────────────────────────────────────
    def _proc_sql(self, stmt: MacroStatement, dialect: str) -> tuple:
        inp      = stmt.attrs['input']
        out      = stmt.attrs['output']
        select   = stmt.attrs['select']
        where    = stmt.attrs['where']
        group_by = stmt.attrs['group_by']
        order_by = stmt.attrs['order_by']

        # Raw SQL only (no CREATE TABLE parsed)
        if stmt.attrs.get('raw_sql'):
            return [
                f"# PROC SQL (complex) — manual conversion needed",
                f"# {stmt.attrs['raw_sql'][:120].replace(chr(10),' ')}",
            ], 0.20

        has_agg = bool(re.search(r'\b(count|sum|mean|avg|min|max)\s*\(', select, re.IGNORECASE))

        # FIX: translate WHERE = to == without breaking <= >= !=
        def where_to_r(w):
            return _sas_cond_to_r(w) if w else ''

        if dialect == "Modern R (dplyr)":
            lines = [f"{out} <- {inp} %>%"]
            if where:
                lines.append(f"  filter({where_to_r(where)}) %>%")
            if group_by:
                grp_cols = [c.strip() for c in group_by.split(',')]
                grp_list = ', '.join(f'"{c}"' for c in grp_cols)
                lines.append(f"  group_by(across(all_of(c({grp_list})))) %>%")
            if has_agg:
                agg_parts = []
                for expr in select.split(','):
                    expr = expr.strip()
                    alias_m = re.search(r'(\w+\s*\([^)]*\))\s+as\s+(\w+)', expr, re.IGNORECASE)
                    if alias_m:
                        agg_parts.append(f"{alias_m.group(2)} = {alias_m.group(1)}")
                    else:
                        agg_parts.append(expr)
                lines.append(f"  summarise({', '.join(agg_parts)}, .groups='drop')")
            else:
                cols = [c.strip() for c in select.split(',')]
                # handle "table.col" → just col
                cols = [c.split('.')[-1] for c in cols]
                col_list = ', '.join(f'"{c}"' for c in cols)
                lines.append(f"  select(all_of(c({col_list})))")
            if order_by:
                ord_cols = [c.strip() for c in order_by.split(',')]
                ord_list = ', '.join(f'"{c}"' for c in ord_cols)
                lines.append(f"  arrange(across(all_of(c({ord_list}))))")
            # Remove trailing %>%
            lines = [ln.rstrip(' %>%') if i == len(lines) - 1 else ln
                     for i, ln in enumerate(lines)]
        else:
            lines = [f"# PROC SQL → base R"]
            if where:
                lines.append(f"{out} <- {inp}[{where_to_r(where)}, ]")
            else:
                lines.append(f"{out} <- {inp}")

        return lines, 0.72

    # ── PROC TRANSPOSE ──────────────────────────────────────────
    def _proc_transpose(self, stmt: MacroStatement, dialect: str) -> tuple:
        inp = stmt.attrs['input']
        out = stmt.attrs['output']
        var = stmt.attrs['var']
        by  = stmt.attrs['by']
        id_ = stmt.attrs['id']

        if dialect == "Modern R (dplyr)":
            if var:
                cols_str = ', '.join(f'"{v}"' for v in var)
                lines = [
                    f"{out} <- {inp} %>%",
                    f"  pivot_longer(cols = all_of(c({cols_str})),",
                    f"               names_to = 'variable',",
                    f"               values_to = 'value')",
                ]
            else:
                lines = [f"{out} <- {inp} %>% pivot_longer(everything())"]
        else:
            if var:
                cols_str = ', '.join(f'"{v}"' for v in var)
                lines = [
                    f"{out} <- reshape({inp}, varying = c({cols_str}),",
                    f"                  v.names = 'value', timevar = 'variable',",
                    f"                  direction = 'long')",
                ]
            else:
                lines = [f"{out} <- reshape({inp}, direction='long')"]

        return lines, 0.80

    # ── %IF / %THEN ─────────────────────────────────────────────
    def _if_else(self, stmt: MacroStatement, params: list, dialect: str) -> tuple:
        cond   = stmt.attrs['condition']
        then_  = stmt.attrs['then_block']
        else_  = stmt.attrs['else_block']
        inline = stmt.attrs.get('inline', False)

        r_cond = _sas_cond_to_r(cond, params)

        if inline:
            # Single-line then/else: emit as R inline
            lines = [f"if ({r_cond}) {then_.lstrip('%')}"]
            if else_:
                lines.append(f"else {else_.lstrip('%')}")
            return lines, 0.65
        else:
            lines = [f"if ({r_cond}) {{"]
            # Recursively convert body lines
            then_stmts = MacroParser().parse('_inner', params, then_).statements
            for s in then_stmts:
                inner_lines, _ = self._convert_statement(s, params, dialect)
                for ln in inner_lines:
                    lines.append(f"  {ln}")
            if else_:
                lines.append("} else {")
                else_stmts = MacroParser().parse('_inner', params, else_).statements
                for s in else_stmts:
                    inner_lines, _ = self._convert_statement(s, params, dialect)
                    for ln in inner_lines:
                        lines.append(f"  {ln}")
            lines.append("}")
            return lines, 0.70

    # ── %DO numeric loop ────────────────────────────────────────
    def _do_loop(self, stmt: MacroStatement, params: list, dialect: str) -> tuple:
        var        = stmt.attrs['var']
        start      = stmt.attrs['start']
        end        = stmt.attrs['end']
        step       = stmt.attrs['step']
        body       = stmt.attrs['body']
        is_literal = stmt.attrs.get('is_literal', False)

        if is_literal:
            step_i = int(step)
            seq = f"seq({start}, {end}, by={step})" if step_i != 1 else f"{start}:{end}"
        else:
            seq = f"seq({start}, {end}, by={step})"

        lines = [f"for ({var} in {seq}) {{"]
        # Recursively convert loop body
        inner_stmts = MacroParser().parse('_loop', params, body).statements
        for s in inner_stmts:
            inner_lines, _ = self._convert_statement(s, params, dialect)
            for ln in inner_lines:
                lines.append(f"  {ln}")
        lines.append("}")
        return lines, 0.75

    # ── %DO %WHILE / %DO %UNTIL ─────────────────────────────────
    def _do_while(self, stmt: MacroStatement, params: list, dialect: str) -> tuple:
        cond  = _sas_cond_to_r(stmt.attrs['condition'], params)
        body  = stmt.attrs['body']
        kind  = stmt.attrs.get('kind', 'while')

        if kind == 'until':
            r_cond = f"!({cond})"
        else:
            r_cond = cond

        lines = [f"while ({r_cond}) {{"]
        inner_stmts = MacroParser().parse('_while', params, body).statements
        for s in inner_stmts:
            inner_lines, _ = self._convert_statement(s, params, dialect)
            for ln in inner_lines:
                lines.append(f"  {ln}")
        lines.append("}")
        return lines, 0.65


# ─────────────────────────────────────────────────────────────────
# LLM CONVERTER (fallback only)
# ─────────────────────────────────────────────────────────────────

class LLMConverter:
    """
    Fallback converter for complex macros.
    Uses Groq first (cheap), Gemini as backup.
    """

    def __init__(self, groq_client, gemini_client):
        self.groq   = groq_client
        self.gemini = gemini_client

    def convert(self, ir: MacroIR, dialect: str) -> tuple:
        dialect_hint = "tidyverse/dplyr" if "dplyr" in dialect else "base R"
        prompt = (
            f"Convert this SAS macro to a reusable R function using {dialect_hint}.\n\n"
            f"MACRO NAME: {ir.name}\n"
            f"PARAMETERS: {', '.join(ir.params)}\n"
            f"BODY:\n{ir.body_raw}\n\n"
            "RULES:\n"
            f"1. Create ONE R function named exactly '{ir.name.lower()}'\n"
            "2. ALL parameter names MUST be lowercase\n"
            "3. Use df[[\"colname\"]] for dynamic column references — never bare names\n"
            "4. Dataset name parameters → dataframe arguments\n"
            "5. PROC SORT → arrange(); PROC MEANS → group_by/summarise();\n"
            "   PROC FREQ → group_by/tally(); PROC TRANSPOSE → pivot_longer()\n"
            "6. %if/%then → if/else in R\n"
            "7. %do loops → for loops in R\n"
            "8. Function should be self-contained and reusable\n"
            "9. Add a comment showing example usage\n"
            "10. Return ONLY the R function code — no explanations\n"
        )

        raw = None
        try:
            res = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = res.choices[0].message.content
        except Exception:
            try:
                raw = self.gemini.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt
                ).text
            except Exception:
                return f"# Could not convert macro %{ir.name} — manual conversion needed\n", 0.0

        # Strip markdown fences
        raw = re.sub(r'```[rR]?\n?', '', raw)
        raw = re.sub(r'```', '', raw)
        return raw.strip() + "\n", 0.75


# ─────────────────────────────────────────────────────────────────
# MAIN CONVERTER ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────

class HybridMacroConverter:
    """
    Main entry point. Orchestrates:
    Parser → Scorer → Cache → RuleBased or LLM → Result
    """

    CONFIDENCE_THRESHOLD = 0.65  # below → use LLM

    def __init__(self, groq_client=None, gemini_client=None, cache_file=None):
        self.parser    = MacroParser()
        self.scorer    = ComplexityScorer()
        self.rules     = RuleBasedConverter()
        self.llm       = LLMConverter(groq_client, gemini_client) if groq_client else None
        self.cache     = ConversionCache(cache_file)

        self.stats = {
            "total":      0,
            "cached":     0,
            "rule_based": 0,
            "llm":        0,
            "failed":     0,
        }

    def convert_all(
        self,
        macro_definitions: dict,
        macro_call_list: list,       # renamed from macro_calls to avoid shadowing
        dialect: str = "Modern R (dplyr)"
    ) -> dict:
        """
        Convert all macros and generate call statements.

        Args:
            macro_definitions: {name: {params, body}} from macro_processor
            macro_call_list:   list of macro call strings found in code
            dialect:           'Modern R (dplyr)' or 'Base R'

        Returns:
            {
                'r_functions': str,     # all R function definitions
                'r_calls':     str,     # converted macro calls
                'stats':       dict,    # conversion statistics
                'warnings':    list,    # issues encountered
            }
        """
        r_functions = []
        r_calls     = []
        warnings    = []

        for name, macro in macro_definitions.items():
            self.stats["total"] += 1

            ir = self.parser.parse(
                name=name,
                params=macro.get("params", []),
                body=macro.get("body", "")
            )

            # Detect macros whose body is primarily calls to other macros
            raw_body = macro.get("body", "")
            body_macro_calls = re.findall(
                r'%(\w+)\s*\(([^)]*)\)', raw_body, re.IGNORECASE
            )
            body_macro_calls = [
                (n, a) for n, a in body_macro_calls
                if n.upper() not in MacroParser._MACRO_BUILTINS
            ]

            # Check cache first
            cached = self.cache.get(ir, dialect)
            if cached:
                self.stats["cached"] += 1
                r_functions.append(cached["r_code"])
                warnings.extend(cached.get("warnings", []))
                continue

            # Score complexity
            score, confidence, reasons = self.scorer.score(ir)

            # Body is mainly inter-macro calls → generate chained function calls
            if body_macro_calls and len(ir.statements) <= 1:
                r_lines = []
                prev_result = None
                for i, (call_name, call_args) in enumerate(body_macro_calls):
                    r_args = []
                    for arg in call_args.split(','):
                        arg = arg.strip()
                        if '=' in arg:
                            k, v = arg.split('=', 1)
                            k_clean = k.strip().lstrip('&').lower()
                            v_clean = v.strip().lstrip('&').lower()
                            if prev_result and k_clean in ('ds', 'data', 'df'):
                                r_args.append(f"{k_clean} = {prev_result}")
                            else:
                                r_args.append(f"{k_clean} = {v_clean}")
                        elif arg:
                            r_args.append(arg.lstrip('&').lower())
                    is_last    = (i == len(body_macro_calls) - 1)
                    result_var = "result" if is_last else f"step{i+1}_result"
                    prev_result = result_var
                    r_lines.append(
                        f"  {result_var} <- {call_name.lower()}("
                        + ", ".join(r_args) + ")"
                    )
                r_lines.append(
                    "  # NOTE: Review chaining — ensure correct dataset "
                    "passed to each function"
                )
                r_lines.append("  return(result)")
                params_r  = ", ".join(p.lower() for p in ir.params)
                func_name = name.lower()
                r_code = (
                    f"# SAS macro %{name} converted to R function\n"
                    f"{func_name} <- function({params_r}) {{\n"
                    + "\n".join(r_lines) + "\n"
                    f"}}\n"
                    f"\n# Example call:\n"
                    f"# {func_name}({', '.join(p.lower()+'=<value>' for p in ir.params)})\n"
                )
                method       = "rule-based (macro calls)"
                actual_conf  = 0.90
                self.stats["rule_based"] += 1
                self.cache.put(ir, dialect, {"r_code": r_code, "warnings": []})
                r_functions.append(
                    f"# {'─'*60}\n"
                    f"# Macro: %{name} | Method: {method} | Confidence: {actual_conf:.0%}\n"
                    f"# {'─'*60}\n"
                    + r_code
                )
                continue

            # Choose converter
            if confidence >= self.CONFIDENCE_THRESHOLD or self.llm is None:
                r_code, actual_conf = self.rules.convert(ir, dialect)
                method = "rule-based"
                self.stats["rule_based"] += 1

                # If rule-based confidence still too low and LLM available → fallback
                if actual_conf < self.CONFIDENCE_THRESHOLD and self.llm is not None:
                    r_code, actual_conf = self.llm.convert(ir, dialect)
                    method = "LLM (rule fallback)"
                    self.stats["llm"] += 1
                    self.stats["rule_based"] -= 1
            else:
                if self.llm:
                    r_code, actual_conf = self.llm.convert(ir, dialect)
                    method = "LLM"
                    self.stats["llm"] += 1
                else:
                    r_code = (
                        f"# Macro %{name} is complex (score={score}) "
                        f"— manual conversion needed\n"
                    )
                    actual_conf = 0.0
                    method      = "skipped"
                    self.stats["failed"] += 1

            # Add metadata header
            r_code = (
                f"# {'─'*60}\n"
                f"# Macro: %{name} | Method: {method} | "
                f"Confidence: {actual_conf:.0%}\n"
                f"# {'─'*60}\n"
                + r_code
            )

            if actual_conf < 0.5:
                warnings.append(
                    f"⚠️  Low confidence ({actual_conf:.0%}) converting %{name} — "
                    f"review generated R function carefully."
                )

            self.cache.put(ir, dialect, {"r_code": r_code, "warnings": warnings[-1:]})
            r_functions.append(r_code)

        # Convert macro call strings → R function calls
        for call in macro_call_list:
            r_call = self._convert_call(call, macro_definitions)
            if r_call:
                r_calls.append(r_call)

        return {
            "r_functions": "\n".join(r_functions),
            "r_calls":     "\n".join(r_calls),
            "stats":       dict(self.stats),
            "warnings":    warnings,
        }

    def _convert_call(self, call: str, macro_defs: dict) -> Optional[str]:
        """Convert a %macro_name(args) call to R function call."""
        m = re.match(r'%(\w+)\s*(?:\(([^)]*)\))?\s*;?', call.strip(), re.IGNORECASE)
        if not m:
            return None

        name     = m.group(1).upper()
        args_raw = m.group(2) or ""

        if name not in macro_defs:
            return f"# {call.strip()}  # macro not found in definitions"

        r_args = []
        for arg in args_raw.split(','):
            arg = arg.strip()
            if '=' in arg:
                k, v = arg.split('=', 1)
                r_args.append(f"{k.strip().lstrip('&').lower()} = {v.strip()}")
            elif arg:
                r_args.append(arg.lstrip('&').lower())

        return f"{name.lower()}({', '.join(r_args)})"


# ─────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION (used by app.py)
# ─────────────────────────────────────────────────────────────────

def convert_macros_to_r(
    macro_definitions: dict,
    macro_calls: list,
    dialect: str,
    groq_client=None,
    gemini_client=None,
    cache_file: str = None
) -> dict:
    """
    Main entry point for app.py.

    Returns dict with r_functions, r_calls, stats, warnings.

    Note: `macro_calls` here is the list of call-strings from the SAS source;
    it is passed as `macro_call_list` internally to avoid variable shadowing.
    """
    converter = HybridMacroConverter(
        groq_client=groq_client,
        gemini_client=gemini_client,
        cache_file=cache_file
    )
    return converter.convert_all(macro_definitions, macro_calls, dialect)
