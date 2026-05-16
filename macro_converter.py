"""
macro_converter.py
──────────────────
Hybrid SAS Macro → R Function Converter

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
"""

import re
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# INTERMEDIATE REPRESENTATION (IR)
# Future-proof: add more fields as we evolve toward full AST
# ─────────────────────────────────────────────────────────────────

@dataclass
class MacroStatement:
    """Single statement inside a macro body."""
    kind: str          # 'proc_sort' | 'proc_means' | 'proc_freq' |
                       # 'proc_sql'  | 'data_step'  | 'if_else'   |
                       # 'do_loop'   | 'let'        | 'call'      | 'unknown'
    raw:  str          # original SAS text
    attrs: dict = field(default_factory=dict)  # parsed attributes


@dataclass
class MacroIR:
    """Intermediate Representation of one SAS macro."""
    name:       str
    params:     list[str]
    body_raw:   str
    statements: list[MacroStatement] = field(default_factory=list)
    complexity: int   = 0       # computed score
    confidence: float = 0.0     # rule-based confidence 0.0-1.0


# ─────────────────────────────────────────────────────────────────
# CONVERSION CACHE
# ─────────────────────────────────────────────────────────────────

class ConversionCache:
    """
    In-memory + optional JSON-file cache.
    Key = SHA256(macro_name + params + body + dialect)
    """

    def __init__(self, cache_file: Optional[str] = None):
        self._mem: dict[str, dict] = {}
        self._file = cache_file
        if cache_file:
            self._load()

    def _make_key(self, ir: MacroIR, dialect: str) -> str:
        raw = f"{ir.name}|{ir.params}|{ir.body_raw}|{dialect}"
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
        (r'proc\s+sql',             6,  "PROC SQL — complex query"),
        (r'proc\s+transpose',       5,  "PROC TRANSPOSE"),
        (r'proc\s+report',          7,  "PROC REPORT"),
        (r'proc\s+tabulate',        7,  "PROC TABULATE"),
        (r'%do\s+%while',           6,  "%DO %WHILE loop"),
        (r'%do\s+%until',           6,  "%DO %UNTIL loop"),
        (r'%syscall',               6,  "SYSCALL"),
        (r'proc\s+iml',             9,  "PROC IML — matrix language"),
        (r'array\s+\w+',            5,  "ARRAY statement"),
        (r'retain\s+',              4,  "RETAIN statement"),
        (r'lag\s*\(',               4,  "LAG function"),
        (r'infile\s+',              6,  "INFILE — external data"),
    ]

    # Patterns that DECREASE complexity (easy to rule-convert)
    SIMPLE = [
        (r'proc\s+sort',            -3, "PROC SORT — simple"),
        (r'proc\s+means',           -2, "PROC MEANS — simple"),
        (r'proc\s+freq',            -2, "PROC FREQ — simple"),
        (r'data\s+\w+;\s*set\s+',   -2, "DATA step SET — simple"),
        (r'%if\s+.*?%then',         +3, "%IF/%THEN — conditional"),
        (r'%do\s+\w+\s*=\s*\d+',   +3, "%DO numeric loop"),
    ]

    THRESHOLD = 10  # score above this → LLM

    def score(self, ir: MacroIR) -> tuple[int, float, list]:
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

    def parse(self, name: str, params: list[str], body: str) -> MacroIR:
        ir = MacroIR(name=name, params=params, body_raw=body)
        ir.statements = self._parse_statements(body)
        return ir

    def _parse_statements(self, body: str) -> list[MacroStatement]:
        stmts = []

        # PROC SORT
        for m in re.finditer(
            r'proc\s+sort\s+data\s*=\s*&?(\w+)\s*(?:out\s*=\s*&?(\w+))?\s*;'
            r'(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            by_vars = re.findall(r'\bby\s+(.*?);', m.group(3), re.IGNORECASE)
            stmts.append(MacroStatement(
                kind='proc_sort',
                raw=m.group(0),
                attrs={
                    'input':   m.group(1),
                    'output':  m.group(2) or m.group(1),
                    'by_vars': by_vars[0].split() if by_vars else [],
                }
            ))

        # PROC MEANS
        for m in re.finditer(
            r'proc\s+means\s+data\s*=\s*&?(\w+)(.*?);(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            opts  = m.group(2)
            inner = m.group(3)
            class_m = re.search(r'\bclass\s+(.*?);', inner, re.IGNORECASE)
            var_m   = re.search(r'\bvar\s+(.*?);',   inner, re.IGNORECASE)
            out_m   = re.search(r'\boutput\s+out\s*=\s*&?(\w+)(.*?);', inner, re.IGNORECASE)
            stmts.append(MacroStatement(
                kind='proc_means',
                raw=m.group(0),
                attrs={
                    'input':     m.group(1),
                    'class_var': class_m.group(1).split() if class_m else [],
                    'var':       var_m.group(1).split() if var_m else [],
                    'output':    out_m.group(1) if out_m else None,
                    'stats':     self._parse_means_stats(opts),
                }
            ))

        # PROC FREQ
        for m in re.finditer(
            r'proc\s+freq\s+data\s*=\s*&?(\w+)\s*;(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            tables_m = re.search(r'\btables\s+(.*?);', m.group(2), re.IGNORECASE)
            stmts.append(MacroStatement(
                kind='proc_freq',
                raw=m.group(0),
                attrs={
                    'input':  m.group(1),
                    'tables': tables_m.group(1).strip() if tables_m else '',
                }
            ))

        # DATA STEP (simple set)
        for m in re.finditer(
            r'data\s+&?(\w+)\s*;\s*set\s+&?(\w+)\s*;(.*?)run\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            stmts.append(MacroStatement(
                kind='data_step',
                raw=m.group(0),
                attrs={
                    'output': m.group(1),
                    'input':  m.group(2),
                    'body':   m.group(3).strip(),
                }
            ))

        # %IF/%THEN (simple)
        for m in re.finditer(
            r'%if\s+(.*?)\s*%then\s*%do\s*;(.*?)%end\s*;'
            r'(?:\s*%else\s*%do\s*;(.*?)%end\s*;)?',
            body, re.IGNORECASE | re.DOTALL
        ):
            stmts.append(MacroStatement(
                kind='if_else',
                raw=m.group(0),
                attrs={
                    'condition':  m.group(1).strip(),
                    'then_block': m.group(2).strip(),
                    'else_block': (m.group(3) or '').strip(),
                }
            ))

        # %DO numeric loop
        for m in re.finditer(
            r'%do\s+(\w+)\s*=\s*(\d+)\s*%to\s*(\d+)(?:\s*%by\s*(\d+))?\s*;(.*?)%end\s*;',
            body, re.IGNORECASE | re.DOTALL
        ):
            stmts.append(MacroStatement(
                kind='do_loop',
                raw=m.group(0),
                attrs={
                    'var':   m.group(1),
                    'start': int(m.group(2)),
                    'end':   int(m.group(3)),
                    'step':  int(m.group(4) or 1),
                    'body':  m.group(5).strip(),
                }
            ))

        # If nothing parsed → unknown
        if not stmts:
            stmts.append(MacroStatement(kind='unknown', raw=body))

        return stmts

    def _parse_means_stats(self, opts: str) -> list[str]:
        known = ['mean', 'std', 'min', 'max', 'median', 'n', 'sum', 'var']
        found = []
        for s in known:
            if re.search(rf'\b{s}\b', opts, re.IGNORECASE):
                found.append(s)
        return found or ['mean', 'std']


# ─────────────────────────────────────────────────────────────────
# RULE-BASED R FUNCTION GENERATOR
# ─────────────────────────────────────────────────────────────────

class RuleBasedConverter:
    """
    Converts MacroIR → R function using deterministic rules.
    Returns (r_code, confidence).
    """

    def convert(self, ir: MacroIR, dialect: str = "Modern R (dplyr)") -> tuple[str, float]:
        params_r = ", ".join(ir.params) if ir.params else ""
        body_lines = []
        total_conf = 1.0

        for stmt in ir.statements:
            r_lines, conf = self._convert_statement(stmt, ir.params, dialect)
            body_lines.extend(r_lines)
            total_conf = min(total_conf, conf)

        body = "\n".join(f"  {ln}" for ln in body_lines if ln.strip())

        r_func = (
            f"# SAS macro %{ir.name} converted to R function\n"
            f"{ir.name} <- function({params_r}) {{\n"
            f"{body}\n"
            f"}}\n"
        )

        # Generate example call
        call_args = ", ".join(f'{p} = <value>' for p in ir.params)
        r_func += f"\n# Example call:\n# {ir.name}({call_args})\n"

        return r_func, total_conf

    def _convert_statement(
        self, stmt: MacroStatement, params: list[str], dialect: str
    ) -> tuple[list[str], float]:

        if stmt.kind == 'proc_sort':
            return self._proc_sort(stmt, dialect)
        elif stmt.kind == 'proc_means':
            return self._proc_means(stmt, dialect)
        elif stmt.kind == 'proc_freq':
            return self._proc_freq(stmt, dialect)
        elif stmt.kind == 'data_step':
            return self._data_step(stmt, dialect)
        elif stmt.kind == 'if_else':
            return self._if_else(stmt, params, dialect)
        elif stmt.kind == 'do_loop':
            return self._do_loop(stmt, params, dialect)
        else:
            return [f"# TODO: Convert manually:\n  # {stmt.raw[:80]}"], 0.2

    def _ref(self, name: str, dialect: str) -> str:
        """Return correct column reference style."""
        if dialect == "Modern R (dplyr)":
            return f'.data[[{name}]]'
        return f'df${name}'

    def _proc_sort(self, stmt: MacroStatement, dialect: str) -> tuple[list[str], float]:
        inp    = stmt.attrs['input']
        out    = stmt.attrs['output']
        by_vars = stmt.attrs['by_vars']

        # Detect descending
        desc_vars = []
        asc_vars  = []
        i = 0
        while i < len(by_vars):
            if by_vars[i].upper() == 'DESCENDING' and i + 1 < len(by_vars):
                desc_vars.append(by_vars[i + 1])
                i += 2
            else:
                asc_vars.append(by_vars[i])
                i += 1

        if dialect == "Modern R (dplyr)":
            arrange_args = (
                [f'desc({v})' if v in desc_vars else v for v in by_vars]
            )
            lines = [
                f"{out} <- {inp} %>%",
                f"  arrange({', '.join(arrange_args)})",
            ]
        else:
            order_args = [f'-{inp}${v}' if v in desc_vars else f'{inp}${v}' for v in by_vars]
            lines = [
                f"{out} <- {inp}[order({', '.join(order_args)}), ]",
            ]

        return lines, 0.95

    def _proc_means(self, stmt: MacroStatement, dialect: str) -> tuple[list[str], float]:
        inp       = stmt.attrs['input']
        grp_vars  = stmt.attrs['class_var']
        num_vars  = stmt.attrs['var']
        out       = stmt.attrs['output'] or f"{inp}_means"
        stats     = stmt.attrs['stats']

        stat_map = {
            'mean':   'mean({v}, na.rm=TRUE)',
            'std':    'sd({v}, na.rm=TRUE)',
            'min':    'min({v}, na.rm=TRUE)',
            'max':    'max({v}, na.rm=TRUE)',
            'median': 'median({v}, na.rm=TRUE)',
            'n':      'n()',
            'sum':    'sum({v}, na.rm=TRUE)',
        }

        if dialect == "Modern R (dplyr)":
            summarise_parts = []
            for v in num_vars:
                for s in stats:
                    if s in stat_map:
                        expr = stat_map[s].replace('{v}', v)
                        summarise_parts.append(f"{s}_{v} = {expr}")

            lines = [f"{out} <- {inp} %>%"]
            if grp_vars:
                lines.append(f"  group_by({', '.join(grp_vars)}) %>%")
            lines.append(f"  summarise({', '.join(summarise_parts)}, .groups='drop')")
        else:
            lines = []
            agg_dfs = []
            for v in num_vars:
                for s in stats:
                    agg_name = f"agg_{s}_{v}"
                    if grp_vars:
                        formula = f"{v} ~ {' + '.join(grp_vars)}"
                        fun_map = {'mean': 'mean', 'std': 'sd', 'min': 'min',
                                   'max': 'max', 'median': 'median', 'n': 'length', 'sum': 'sum'}
                        fun = fun_map.get(s, 'mean')
                        lines.append(f"{agg_name} <- aggregate({formula}, data={inp}, FUN={fun})")
                        lines.append(f"names({agg_name})[ncol({agg_name})] <- '{s}_{v}'")
                    else:
                        lines.append(f"{agg_name} <- data.frame({s}_{v} = {s}({inp}${v}, na.rm=TRUE))")
                    agg_dfs.append(agg_name)

            if len(agg_dfs) > 1:
                merge_by = grp_vars if grp_vars else 'NULL'
                lines.append(f"{out} <- Reduce(function(a,b) merge(a,b,by={merge_by}), list({', '.join(agg_dfs)}))")
            elif agg_dfs:
                lines.append(f"{out} <- {agg_dfs[0]}")

        return lines, 0.88

    def _proc_freq(self, stmt: MacroStatement, dialect: str) -> tuple[list[str], float]:
        inp    = stmt.attrs['input']
        tables = stmt.attrs['tables']
        vars_  = [v.strip() for v in re.split(r'[\s*]', tables) if v.strip()]
        out    = f"{inp}_freq"

        if dialect == "Modern R (dplyr)":
            lines = [
                f"{out} <- {inp} %>%",
                f"  count({', '.join(vars_)}) %>%",
                f"  rename(COUNT = n)",
            ]
        else:
            lines = [
                f"{out} <- as.data.frame(table({', '.join(f'{inp}${v}' for v in vars_)}))",
                f"names({out}) <- c({', '.join(repr(v) for v in vars_)}, 'COUNT')",
                f"{out} <- {out}[{out}$COUNT > 0, ]",
            ]

        return lines, 0.92

    def _data_step(self, stmt: MacroStatement, dialect: str) -> tuple[list[str], float]:
        inp  = stmt.attrs['input']
        out  = stmt.attrs['output']
        body = stmt.attrs['body']

        # Extract simple assignments: var = expression;
        assigns = re.findall(r'(\w+)\s*=\s*([^;]+);', body)

        if dialect == "Modern R (dplyr)":
            if assigns:
                mutate_parts = [f"{v} = {e.strip()}" for v, e in assigns]
                lines = [
                    f"{out} <- {inp} %>%",
                    f"  mutate({', '.join(mutate_parts)})",
                ]
            else:
                lines = [f"{out} <- {inp}  # TODO: Review DATA step body"]
            conf = 0.80 if assigns else 0.40
        else:
            if assigns:
                lines = [f"{out} <- {inp}"]
                for v, e in assigns:
                    lines.append(f"{out}${v} <- {e.strip()}")
            else:
                lines = [f"{out} <- {inp}  # TODO: Review DATA step body"]
            conf = 0.80 if assigns else 0.40

        return lines, conf

    def _if_else(self, stmt: MacroStatement, params: list[str], dialect: str) -> tuple[list[str], float]:
        cond  = stmt.attrs['condition']
        then_ = stmt.attrs['then_block']
        else_ = stmt.attrs['else_block']

        # Convert SAS condition to R
        r_cond = cond
        r_cond = re.sub(r'\bne\b',  '!=', r_cond, flags=re.IGNORECASE)
        r_cond = re.sub(r'\bgt\b',  '>',  r_cond, flags=re.IGNORECASE)
        r_cond = re.sub(r'\blt\b',  '<',  r_cond, flags=re.IGNORECASE)
        r_cond = re.sub(r'\bge\b',  '>=', r_cond, flags=re.IGNORECASE)
        r_cond = re.sub(r'\ble\b',  '<=', r_cond, flags=re.IGNORECASE)
        r_cond = re.sub(r'\^=',     '!=', r_cond)
        # param references
        for p in params:
            r_cond = re.sub(rf'&{p}\b', p, r_cond, flags=re.IGNORECASE)

        lines = [
            f"if ({r_cond}) {{",
            f"  # {then_[:60]}",
            f"}}",
        ]
        if else_:
            lines[-1] = f"}} else {{"
            lines.append(f"  # {else_[:60]}")
            lines.append(f"}}")

        return lines, 0.70

    def _do_loop(self, stmt: MacroStatement, params: list[str], dialect: str) -> tuple[list[str], float]:
        var   = stmt.attrs['var']
        start = stmt.attrs['start']
        end   = stmt.attrs['end']
        step  = stmt.attrs['step']
        body  = stmt.attrs['body']

        seq = f"seq({start}, {end}, by={step})" if step != 1 else f"{start}:{end}"
        lines = [
            f"for ({var} in {seq}) {{",
            f"  # {body[:60]}",
            f"}}",
        ]
        return lines, 0.75


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

    def convert(self, ir: MacroIR, dialect: str) -> tuple[str, float]:
        dialect_hint = "tidyverse/dplyr" if "dplyr" in dialect else "base R"
        prompt = (
            f"Convert this SAS macro to a reusable R function using {dialect_hint}.\n\n"
            f"MACRO NAME: {ir.name}\n"
            f"PARAMETERS: {', '.join(ir.params)}\n"
            f"BODY:\n{ir.body_raw}\n\n"
            f"RULES:\n"
            f"1. Create ONE R function named exactly '{ir.name}'\n"
            f"2. Function parameters match macro parameters exactly\n"
            f"3. &param references → function arguments\n"
            f"4. Dataset name parameters → dataframe arguments\n"
            f"5. PROC steps → equivalent R/dplyr code inside function\n"
            f"6. %if/%then → if/else in R\n"
            f"7. %do loops → for loops in R\n"
            f"8. Function should be self-contained and reusable\n"
            f"9. Add a comment showing example usage\n"
            f"10. Return ONLY the R function code — no explanations\n"
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

        # Clean
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

        # Stats
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
        macro_calls: list[str],
        dialect: str = "Modern R (dplyr)"
    ) -> dict:
        """
        Convert all macros and generate call statements.

        Args:
            macro_definitions: {name: {params, body}} from macro_processor
            macro_calls:       list of macro call strings found in code
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

            # Check cache first
            cached = self.cache.get(ir, dialect)
            if cached:
                self.stats["cached"] += 1
                r_functions.append(cached["r_code"])
                warnings.extend(cached.get("warnings", []))
                continue

            # Score complexity
            score, confidence, reasons = self.scorer.score(ir)

            # Choose converter
            if confidence >= self.CONFIDENCE_THRESHOLD or self.llm is None:
                r_code, actual_conf = self.rules.convert(ir, dialect)
                method = "rule-based"
                self.stats["rule_based"] += 1

                # If rule-based confidence too low and LLM available → fallback
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
                    r_code = f"# Macro %{name} is complex — manual conversion needed\n"
                    actual_conf = 0.0
                    method = "skipped"
                    self.stats["failed"] += 1

            # Add metadata comment
            r_code = (
                f"# {'─'*60}\n"
                f"# Macro: %{name} | Method: {method} | "
                f"Confidence: {actual_conf:.0%}\n"
                f"# {'─'*60}\n"
                + r_code
            )

            if actual_conf < 0.5:
                warnings.append(
                    f"⚠️ Low confidence ({actual_conf:.0%}) converting %{name} — "
                    f"review generated R function carefully."
                )

            # Cache result
            self.cache.put(ir, dialect, {"r_code": r_code, "warnings": warnings[-1:]})
            r_functions.append(r_code)

        # Convert macro calls → R function calls
        for call in macro_calls:
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

        name = m.group(1).upper()
        args_raw = m.group(2) or ""

        if name not in macro_defs:
            return f"# {call.strip()}  # macro not found"

        # Parse named args
        r_args = []
        for arg in args_raw.split(','):
            arg = arg.strip()
            if '=' in arg:
                k, v = arg.split('=', 1)
                r_args.append(f"{k.strip()} = {v.strip()}")
            elif arg:
                r_args.append(arg)

        return f"{name.lower()}({', '.join(r_args)})"


# ─────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION (used by app.py)
# ─────────────────────────────────────────────────────────────────

def convert_macros_to_r(
    macro_definitions: dict,
    macro_calls: list[str],
    dialect: str,
    groq_client=None,
    gemini_client=None,
    cache_file: str = None
) -> dict:
    """
    Main entry point for app.py.

    Returns dict with r_functions, r_calls, stats, warnings.
    """
    converter = HybridMacroConverter(
        groq_client=groq_client,
        gemini_client=gemini_client,
        cache_file=cache_file
    )
    return converter.convert_all(macro_definitions, macro_calls, dialect)
