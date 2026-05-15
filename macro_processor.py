"""
macro_processor.py
──────────────────
SAS Macro Pre-processor for Smart SAS to R Converter.

Handles:
  - %let variable substitution
  - Simple and nested macro definitions/calls
  - %if/%then/%else logic
  - %do/%to/%end numeric loops
  - Dynamic macro name resolution
  - SQL-generated macro variable hints
  - Multi-file macro library merging
"""

import re


# ─────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────

def _strip_comments(code: str) -> str:
    """Remove SAS block comments /* ... */ and * ... ; line comments."""
    # Block comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Line comments starting with * (only at statement start)
    code = re.sub(r'^\s*\*[^;]*;', '', code, flags=re.MULTILINE)
    return code


def _normalize_whitespace(code: str) -> str:
    """Collapse multiple blank lines to one."""
    return re.sub(r'\n{3,}', '\n\n', code).strip()


# ─────────────────────────────────────────────────────────────────
# MAIN PROCESSOR CLASS
# ─────────────────────────────────────────────────────────────────

class SASMacroProcessor:
    """
    Pre-processes SAS code by expanding macros before LLM conversion.
    """

    MAX_DEPTH = 10  # max recursion depth for nested macro calls

    def __init__(self):
        self.macro_library: dict = {}   # {NAME: {params, body}}
        self.let_vars: dict = {}        # {NAME: value}
        self.sql_var_hints: list = []   # hints for LLM
        self.warnings: list = []        # non-fatal issues

    # ── PUBLIC ENTRY POINT ────────────────────────────────────────

    def process(self, sas_code: str, extra_files: list[str] = None) -> tuple[str, list, list]:
        """
        Main entry point.

        Args:
            sas_code:    Main SAS program text
            extra_files: List of additional .sas macro library strings

        Returns:
            (expanded_code, warnings, sql_hints)
        """
        self.macro_library = {}
        self.let_vars = {}
        self.sql_var_hints = []
        self.warnings = []

        # 1. Merge extra macro library files first
        combined = ""
        if extra_files:
            for f in extra_files:
                combined += _strip_comments(f) + "\n"
        combined += _strip_comments(sas_code)

        # 2. Extract %let variables (global)
        self._parse_let_variables(combined)

        # 3. Extract all macro definitions
        self._parse_macro_definitions(combined)

        # 4. Remove macro definitions from code (keep only calls + non-macro code)
        code = self._remove_macro_definitions(combined)

        # 5. Substitute global %let variables
        code = self._substitute_let_vars(code, self.let_vars)

        # 6. Expand numeric %do loops at top level
        code = self._expand_do_loops(code)

        # 7. Detect SQL-generated macro vars (add hints, can't fully resolve)
        self._detect_sql_macro_vars(code)

        # 8. Expand all macro calls (recursive)
        code = self._expand_macro_calls(code, depth=0)

        # 9. Clean up residual macro artifacts
        code = self._cleanup(code)

        return _normalize_whitespace(code), self.warnings, self.sql_var_hints

    # ── STEP 2: PARSE %LET ───────────────────────────────────────

    def _parse_let_variables(self, code: str):
        """Extract all %let name = value; statements."""
        pattern = re.compile(
            r'%let\s+(\w+)\s*=\s*(.*?)\s*;',
            re.IGNORECASE | re.DOTALL
        )
        for m in pattern.finditer(code):
            name  = m.group(1).strip().upper()
            value = m.group(2).strip()
            self.let_vars[name] = value

    # ── STEP 3: PARSE MACRO DEFINITIONS ─────────────────────────

    def _parse_macro_definitions(self, code: str):
        """
        Extract %macro name(params); body %mend; blocks.
        Supports optional macro name after %mend.
        """
        pattern = re.compile(
            r'%macro\s+(\w+)\s*(?:\(([^)]*)\))?\s*;(.*?)%mend(?:\s+\w+)?\s*;',
            re.IGNORECASE | re.DOTALL
        )
        for m in pattern.finditer(code):
            name   = m.group(1).strip().upper()
            params_raw = m.group(2) or ""
            body   = m.group(3).strip()

            # Parse parameters (handle default values: param=default)
            params = []
            for p in params_raw.split(','):
                p = p.strip()
                if not p:
                    continue
                param_name = p.split('=')[0].strip().lstrip('&')
                params.append(param_name.upper())

            self.macro_library[name] = {
                "params": params,
                "body":   body,
            }

    # ── STEP 4: REMOVE MACRO DEFINITIONS ────────────────────────

    def _remove_macro_definitions(self, code: str) -> str:
        """Strip %macro...%mend blocks from code."""
        return re.sub(
            r'%macro\s+\w+\s*(?:\([^)]*\))?\s*;.*?%mend(?:\s+\w+)?\s*;',
            '',
            code,
            flags=re.IGNORECASE | re.DOTALL
        )

    # ── STEP 5: SUBSTITUTE %LET VARS ────────────────────────────

    def _substitute_let_vars(self, code: str, let_dict: dict) -> str:
        """Replace &var and &var. references with their values."""
        for name, value in let_dict.items():
            # &NAME. (with dot separator)
            code = re.sub(rf'&{name}\.', value, code, flags=re.IGNORECASE)
            # &NAME (without dot)
            code = re.sub(rf'&{name}\b', value, code, flags=re.IGNORECASE)
        return code

    # ── STEP 6: EXPAND %DO LOOPS ────────────────────────────────

    def _expand_do_loops(self, code: str, local_vars: dict = None) -> str:
        """
        Expand %do i = start %to end; body %end; loops.
        Handles nested loops recursively.
        """
        pattern = re.compile(
            r'%do\s+(\w+)\s*=\s*(\d+)\s*%to\s*(\d+)(?:\s*%by\s*(\d+))?\s*;(.*?)%end\s*;',
            re.IGNORECASE | re.DOTALL
        )

        def expand(match):
            var   = match.group(1).upper()
            start = int(match.group(2))
            end   = int(match.group(3))
            step  = int(match.group(4) or 1)
            body  = match.group(5)

            expanded = ""
            for i in range(start, end + 1, step):
                iteration = body
                # substitute loop variable
                iteration = re.sub(rf'&{var}\.', str(i), iteration, flags=re.IGNORECASE)
                iteration = re.sub(rf'&{var}\b', str(i), iteration, flags=re.IGNORECASE)
                expanded += iteration + "\n"
            return expanded

        # Up to 5 passes for nested loops
        for _ in range(5):
            new_code = pattern.sub(expand, code)
            if new_code == code:
                break
            code = new_code

        return code

    # ── STEP 7: DETECT SQL MACRO VARS ───────────────────────────

    def _detect_sql_macro_vars(self, code: str):
        """
        Find PROC SQL INTO: patterns — these create macro vars from data.
        We can't resolve them statically, but we generate LLM hints.
        """
        pattern = re.compile(
            r'select\s+(.+?)\s+into\s+:(\w+)\s+from\s+([\w.]+)',
            re.IGNORECASE | re.DOTALL
        )
        for m in pattern.finditer(code):
            expr  = m.group(1).strip()
            var   = m.group(2).strip()
            table = m.group(3).strip()
            hint  = (f"Macro variable &{var} was generated by SQL: "
                     f"SELECT {expr} INTO :{var} FROM {table}. "
                     f"In R, compute this with dplyr before using the value.")
            self.sql_var_hints.append(hint)
            self.warnings.append(
                f"⚠️ SQL-generated macro variable &{var} detected — "
                f"auto-resolved to R comment. Review generated code."
            )

    # ── STEP 8: EXPAND MACRO CALLS ──────────────────────────────

    def _expand_macro_calls(self, code: str, depth: int) -> str:
        """
        Recursively expand macro calls in code.
        Handles: %macro_name; and %macro_name(args);
        """
        if depth > self.MAX_DEPTH:
            self.warnings.append(
                "⚠️ Maximum macro recursion depth reached. "
                "Some nested macros may not be fully expanded."
            )
            return code

        # Pattern: %NAME; or %NAME(args);
        call_pattern = re.compile(
            r'%(\w+)\s*(?:\(([^)]*)\))?\s*;',
            re.IGNORECASE
        )

        changed = True
        passes = 0
        while changed and passes < 20:
            changed = False
            passes += 1

            def replace_call(match):
                nonlocal changed
                name     = match.group(1).upper()
                args_raw = match.group(2) or ""

                # Skip SAS built-in macro statements
                BUILTINS = {
                    'IF', 'THEN', 'ELSE', 'DO', 'END', 'LET', 'PUT',
                    'GLOBAL', 'LOCAL', 'SYMDEL', 'INCLUDE', 'MEND',
                    'MACRO', 'RETURN', 'ABORT', 'GOTO', 'LABEL',
                    'SYSEXEC', 'SYSCALL', 'RUN', 'QUIT'
                }
                if name in BUILTINS:
                    return match.group(0)

                if name not in self.macro_library:
                    # Try dynamic resolution
                    resolved = self._try_resolve_dynamic_name(name)
                    if resolved and resolved in self.macro_library:
                        name = resolved
                    else:
                        self.warnings.append(
                            f"⚠️ Macro %{name} called but not defined — left as-is."
                        )
                        return match.group(0)

                macro    = self.macro_library[name]
                params   = macro["params"]
                body     = macro["body"]
                local_let = {}

                # Parse named args: param=value
                if args_raw.strip():
                    arg_dict = {}
                    for arg in self._split_args(args_raw):
                        if '=' in arg:
                            k, v = arg.split('=', 1)
                            arg_dict[k.strip().lstrip('&').upper()] = v.strip()
                        # positional — match by order
                    # Fill positional args
                    positional_vals = [
                        v for k, v in sorted(
                            ((i, a.strip()) for i, a in enumerate(self._split_args(args_raw))
                             if '=' not in a),
                            key=lambda x: x[0]
                        )
                    ]
                    for i, param in enumerate(params):
                        if param in arg_dict:
                            local_let[param] = arg_dict[param]
                        elif i < len(positional_vals):
                            local_let[param] = positional_vals[i]
                        else:
                            local_let[param] = ""  # default empty

                # Substitute global %let vars first
                expanded = self._substitute_let_vars(body, self.let_vars)
                # Then local parameter substitution
                expanded = self._substitute_let_vars(expanded, local_let)
                # Extract and apply local %let inside macro body
                local_lets = {}
                for lm in re.finditer(r'%let\s+(\w+)\s*=\s*(.*?)\s*;', expanded, re.IGNORECASE):
                    local_lets[lm.group(1).upper()] = lm.group(2).strip()
                if local_lets:
                    expanded = re.sub(r'%let\s+\w+\s*=\s*.*?;', '', expanded, flags=re.IGNORECASE)
                    expanded = self._substitute_let_vars(expanded, local_lets)

                # Expand %do loops in body
                expanded = self._expand_do_loops(expanded, {**self.let_vars, **local_let})

                # Evaluate %if/%then/%else in body
                expanded = self._evaluate_if_else(expanded, {**self.let_vars, **local_let})

                changed = True
                return expanded + "\n"

            code = call_pattern.sub(replace_call, code)

        # Recurse for any newly introduced macro calls
        if changed:
            code = self._expand_macro_calls(code, depth + 1)

        return code

    # ── HELPER: SPLIT MACRO ARGS ─────────────────────────────────

    def _split_args(self, args_raw: str) -> list[str]:
        """Split macro arguments by comma, respecting nested parentheses."""
        args = []
        depth = 0
        current = ""
        for ch in args_raw:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        return args

    # ── HELPER: EVALUATE %IF/%ELSE ───────────────────────────────

    def _evaluate_if_else(self, code: str, local_vars: dict) -> str:
        """
        Evaluate %if condition %then %do; ... %end; %else %do; ... %end;
        Also handles single-statement %if/%then without %do.
        """
        # Full %do block form
        block_pattern = re.compile(
            r'%if\s+(.*?)\s*%then\s*%do\s*;(.*?)%end\s*;'
            r'(?:\s*%else\s*%do\s*;(.*?)%end\s*;)?',
            re.IGNORECASE | re.DOTALL
        )

        def eval_block(match):
            condition  = match.group(1).strip()
            then_block = match.group(2) or ""
            else_block = match.group(3) or ""
            if self._evaluate_condition(condition, local_vars):
                return then_block
            else:
                return else_block

        code = block_pattern.sub(eval_block, code)

        # Single statement form: %if cond %then statement;
        single_pattern = re.compile(
            r'%if\s+(.*?)\s*%then\s+([^;]+;)'
            r'(?:\s*%else\s+([^;]+;))?',
            re.IGNORECASE
        )

        def eval_single(match):
            condition  = match.group(1).strip()
            then_stmt  = match.group(2) or ""
            else_stmt  = match.group(3) or ""
            if self._evaluate_condition(condition, local_vars):
                return then_stmt
            else:
                return else_stmt

        code = single_pattern.sub(eval_single, code)
        return code

    # ── HELPER: EVALUATE CONDITION ───────────────────────────────

    def _evaluate_condition(self, condition: str, local_vars: dict) -> bool:
        """
        Evaluate a simple SAS macro condition.
        Supports: =, ne, ^=, >, <, >=, <=, and, or, not
        Also handles %symexist, blank checks.
        """
        # Substitute variables
        cond = self._substitute_let_vars(condition, {**self.let_vars, **local_vars})

        # Normalize operators
        cond = re.sub(r'\bne\b',  '!=', cond, flags=re.IGNORECASE)
        cond = re.sub(r'\bgt\b',  '>',  cond, flags=re.IGNORECASE)
        cond = re.sub(r'\blt\b',  '<',  cond, flags=re.IGNORECASE)
        cond = re.sub(r'\bge\b',  '>=', cond, flags=re.IGNORECASE)
        cond = re.sub(r'\ble\b',  '<=', cond, flags=re.IGNORECASE)
        cond = re.sub(r'\band\b', 'and',cond, flags=re.IGNORECASE)
        cond = re.sub(r'\bor\b',  'or', cond, flags=re.IGNORECASE)
        cond = re.sub(r'\bnot\b', 'not',cond, flags=re.IGNORECASE)
        cond = re.sub(r'\^=',     '!=', cond)

        # Handle %symexist(var) — check if macro var is defined
        cond = re.sub(
            r'%symexist\s*\(\s*(\w+)\s*\)',
            lambda m: '1' if m.group(1).upper() in {**self.let_vars, **local_vars} else '0',
            cond, flags=re.IGNORECASE
        )

        # Handle blank check: var = (empty string)
        cond = re.sub(r"=\s*''", "== ''", cond)
        cond = re.sub(r'ne\s*""',  "!= ''", cond, flags=re.IGNORECASE)

        # Convert SAS string comparison to Python
        # e.g.  flag = HIGH  →  'HIGH' == 'HIGH'
        def quote_bare_word(m):
            word = m.group(1)
            if word.upper() in ('AND', 'OR', 'NOT', 'TRUE', 'FALSE'):
                return word
            try:
                float(word)
                return word
            except ValueError:
                return f"'{word}'"

        cond = re.sub(r'\b([A-Za-z_]\w*)\b', quote_bare_word, cond)

        # Evaluate safely
        try:
            result = bool(eval(cond, {"__builtins__": {}}))
        except Exception:
            # If we can't evaluate — assume True (include the block)
            result = True

        return result

    # ── HELPER: DYNAMIC MACRO NAME RESOLUTION ───────────────────

    def _try_resolve_dynamic_name(self, name: str) -> str | None:
        """
        Try to resolve dynamic macro names like &prefix._process.
        If the resolved name matches a known macro, return it.
        """
        substituted = self._substitute_let_vars(name, self.let_vars)
        if substituted != name and substituted.upper() in self.macro_library:
            return substituted.upper()
        return None

    # ── STEP 9: CLEANUP ──────────────────────────────────────────

    def _cleanup(self, code: str) -> str:
        """Remove residual macro artifacts."""
        # Remove leftover %let statements
        code = re.sub(r'%let\s+\w+\s*=\s*[^;]*;', '', code, flags=re.IGNORECASE)
        # Remove leftover %put statements
        code = re.sub(r'%put\s+[^;]*;', '', code, flags=re.IGNORECASE)
        # Remove stray % references that couldn't be resolved
        code = re.sub(r'&\w+\.?', '', code)
        return code


# ─────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION (used by app.py)
# ─────────────────────────────────────────────────────────────────

def expand_sas_macros(sas_code: str, extra_files: list[str] = None) -> tuple[str, list, list]:
    """
    Convenience wrapper around SASMacroProcessor.

    Returns:
        (expanded_code, warnings, sql_hints)
    """
    processor = SASMacroProcessor()
    return processor.process(sas_code, extra_files=extra_files)


def has_macros(sas_code: str) -> bool:
    """Quick check — does this SAS code contain any macro definitions or calls?"""
    return bool(re.search(r'%macro\s+\w+|%\w+\s*[\(;]', sas_code, re.IGNORECASE))
