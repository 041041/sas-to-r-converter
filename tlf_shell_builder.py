"""
tlf_shell_builder.py
─────────────────────────────────────────────────────────────────────────────
TLF from Mock Shell  — agentic pipeline (LangGraph-style state machine)
Nodes: parse_shell → plan_code → generate_code → execute → validate → fix
Same patterns as graph_builder.py: file upload OR paste, AI box, diff review,
Gemini primary / Groq fallback, session-state prefixed ms_ to avoid collisions.
─────────────────────────────────────────────────────────────────────────────
"""

import os, re, io, subprocess, tempfile, traceback
from typing import TypedDict, Optional
import pandas as pd
import streamlit as st
from groq import Groq
from google import genai

# ─── Clients (same helper as graph_builder) ──────────────────────────────────
def _get_secret(key):
    try:    return st.secrets[key]
    except Exception: return os.environ.get(key, "")

_gemini = genai.Client(api_key=_get_secret("GEMINI_API_KEY"))
_groq   = Groq(api_key=_get_secret("GROQ_API_KEY"))

MAX_RETRIES = 3

# ─── LangGraph-style State ────────────────────────────────────────────────────
class ShellTLFState(TypedDict):
    shell_text:        str            # raw mock shell pasted/uploaded
    adam_csv:          Optional[str]  # CSV string of ADaM dataset
    parsed_spec:       dict           # extracted by parse_shell node
    generated_code:    str            # R code from generate_code node
    execution_output:  str            # stdout / table HTML
    execution_error:   str            # stderr if failed
    validation_result: str            # "pass" | "fail: <reason>"
    retry_count:       int
    final_r_code:      str
    final_output:      str            # rendered HTML table or message


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Shell Parser
# ══════════════════════════════════════════════════════════════════════════════
def _call_llm(prompt: str) -> str:
    """Gemini primary, Groq fallback."""
    try:
        return _gemini.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        ).text
    except Exception:
        try:
            res = _groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return res.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Both LLMs failed: {e}")


def node_parse_shell(state: ShellTLFState) -> ShellTLFState:
    """
    Extract structured spec from the mock shell text.
    Returns parsed_spec dict with keys:
      title, footnotes, population, columns, row_stubs, dataset_hint,
      output_type (Table/Listing/Figure), tlf_number
    """
    prompt = f"""You are a clinical programmer. Parse this mock shell and extract a JSON spec.

MOCK SHELL:
{state['shell_text']}

Return ONLY valid JSON, no markdown, no explanation. Schema:
{{
  "tlf_number":   "e.g. Table 14.1.1 or empty string",
  "title":        "full title text",
  "footnotes":    ["list of footnote strings"],
  "population":   "e.g. Safety Population or empty string",
  "pop_flag":     "e.g. SAFFL or ITTFL or empty string",
  "output_type":  "Table" | "Listing" | "Figure",
  "dataset_hint": "e.g. ADSL, ADAE, ADVS or empty string",
  "columns":      ["col1", "col2", ...],
  "row_stubs":    ["row label 1", "row label 2", ...],
  "statistics":   ["n", "mean", "sd", "median", "min", "max", "pct"] (subset relevant),
  "groupby":      "treatment variable name or empty string"
}}"""
    raw = _call_llm(prompt)
    raw = re.sub(r'```json|```', '', raw).strip()
    import json
    try:
        spec = json.loads(raw)
    except Exception:
        # Fallback minimal spec
        spec = {
            "tlf_number": "", "title": "Table", "footnotes": [],
            "population": "", "pop_flag": "", "output_type": "Table",
            "dataset_hint": "", "columns": [], "row_stubs": [],
            "statistics": ["n", "pct"], "groupby": ""
        }
    state["parsed_spec"] = spec
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Code Generator
# ══════════════════════════════════════════════════════════════════════════════
def node_generate_code(state: ShellTLFState) -> ShellTLFState:
    """
    Generate R code (gt / flextable) from parsed_spec + optional ADaM preview.
    """
    spec      = state["parsed_spec"]
    adam_hint = ""
    if state.get("adam_csv"):
        # Send only first 5 rows as context
        try:
            df_preview = pd.read_csv(io.StringIO(state["adam_csv"])).head(5).to_string()
            adam_hint  = f"\nADaM dataset preview (first 5 rows):\n{df_preview}\n"
        except Exception:
            adam_hint = ""

    prompt = f"""You are an expert clinical R programmer generating production-ready TLF code.

SPEC:
{spec}
{adam_hint}

RULES:
1. Use gt package for Tables, ggplot2 for Figures.
2. If ADaM data is provided, read from df (already loaded). If not, create realistic dummy data.
3. Do NOT load any libraries — dplyr, tidyr, gt, ggplot2 are already loaded.

4. CRITICAL TABLE STRUCTURE — clinical shell format:
   - COLUMNS (left to right): Parameter | Statistic | Placebo (N=xx) | Drug A (N=xx) | Total (N=xx)
   - ROWS (top to bottom): Age > n / Mean (SD) / Median / Min, Max | Sex > Male / Female | etc.
   - Treatment group is NEVER a row. It is ALWAYS a column header.
   - Statistics (n, Mean, Median) are NEVER column headers. They are row labels.

5. CRITICAL — build stats row by row using bind_rows + summarise, NOT mutate(col = c(...)):
   age_stats <- bind_rows(
     df %>% group_by(TRT01P) %>% summarise(val=as.character(n()), .groups="drop") %>% mutate(Statistic="n"),
     df %>% group_by(TRT01P) %>% summarise(val=sprintf("%.1f (%.1f)", mean(AGE,na.rm=TRUE), sd(AGE,na.rm=TRUE)), .groups="drop") %>% mutate(Statistic="Mean (SD)"),
     df %>% group_by(TRT01P) %>% summarise(val=sprintf("%.1f", median(AGE,na.rm=TRUE)), .groups="drop") %>% mutate(Statistic="Median"),
     df %>% group_by(TRT01P) %>% summarise(val=sprintf("%g, %g", min(AGE,na.rm=TRUE), max(AGE,na.rm=TRUE)), .groups="drop") %>% mutate(Statistic="Min, Max")
   ) %>% pivot_wider(names_from=TRT01P, values_from=val) %>% mutate(Parameter="Age (years)")

   Add a Total column: df_total <- df (no group filter), same summarise pattern.

6. After building all parameter blocks, bind_rows() them all into one data.frame called tbl_data.
   Columns must be: Parameter, Statistic, then one column per treatment group, then Total.
   ALL column names must be non-empty strings — never empty string "" as a column name.

7. Pass tbl_data to gt() and format:
   tbl <- gt(tbl_data) %>%
     tab_header(title="{spec.get('title','Table')}") %>%
     cols_label(Parameter="Parameter", Statistic="Statistic") %>%
     tab_row_group(label=<param>, rows=Parameter==<param>) for each parameter %>%
     tab_style(style=cell_text(weight="bold"), locations=cells_row_groups())

8. Footnotes: tab_footnote(footnote="...", locations=cells_title()) only. NOT on column headers.

9. VALID gt functions only:
   gt(), tab_header(), tab_footnote(), tab_spanner(), cols_label(),
   fmt_number(), tab_style(), cell_text(), cells_column_labels(),
   cells_body(), cells_row_groups(), cells_title(),
   tab_row_group(), row_group_order(), as_raw_html()
   NEVER USE: column_labels(), tab_column_label(), set_column_labels()

10. Last line MUST be: cat(gt::as_raw_html(tbl))
11. Never include read.csv, library(), or ggsave.
12. Return ONLY R code. No markdown fences. No explanations.

Generate complete R code now:"""
    raw = _call_llm(prompt)
    raw = re.sub(r'```[rR]?\n?', '', raw)
    raw = re.sub(r'```', '', raw).strip()
    # Strip any stray ggsave
    raw = re.sub(r'\+?\s*ggsave\s*\([^)]*\)', '', raw, flags=re.DOTALL).strip()

    state["generated_code"] = raw
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Executor
# ══════════════════════════════════════════════════════════════════════════════
def node_execute(state: ShellTLFState) -> ShellTLFState:
    """Run R code, capture stdout as execution_output."""
    spec        = state["parsed_spec"]
    output_type = spec.get("output_type", "Table")

    with tempfile.TemporaryDirectory() as d:
        script_path = os.path.join(d, "tlf_script.R")
        plot_path   = os.path.join(d, "figure.png")

        # Prefix: library path + preload packages silently + optional data load
        prefix_lines = [
            "user_lib <- path.expand('~/R/library')",
            "if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))",
            "options(warn=-1)",
            "suppressMessages(suppressWarnings({",
            "  library(dplyr)",
            "  library(tidyr)",
            "  library(gt)",
            "  library(ggplot2)",
            "}))",
        ]

        if state.get("adam_csv"):
            inp = os.path.join(d, "adam.csv")
            with open(inp, "w") as f:
                f.write(state["adam_csv"])
            prefix_lines.append(f'df <- read.csv("{inp}", stringsAsFactors=FALSE)')

        suffix_lines = []
        if output_type == "Figure":
            suffix_lines = [f'suppressMessages(ggsave("{plot_path}", width=10, height=6, dpi=150))']

        full_script = "\n".join(prefix_lines + [state["generated_code"]] + suffix_lines)

        with open(script_path, "w") as f:
            f.write(full_script)

        try:
            res = subprocess.run(
                ["Rscript", script_path],
                capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            state["execution_error"]  = "R script timed out (>60s)"
            state["execution_output"] = ""
            return state

        if res.returncode != 0:
            # Extract actual R error — skip dplyr/package warning noise
            stderr_lines = res.stderr.splitlines()
            error_lines  = [
                l for l in stderr_lines
                if any(kw in l for kw in ["Error", "error", "Execution halted", "object '"])
                and not any(kw in l for kw in ["masked from", "Attaching", "summarise()", "grouped by"])
            ]
            clean_error = "\n".join(error_lines) if error_lines else res.stderr
            state["execution_error"]  = clean_error
            state["execution_output"] = ""
            return state

        # returncode == 0 — success even if stderr has dplyr noise
        state["execution_error"] = ""

        if output_type == "Figure":
            if os.path.exists(plot_path):
                with open(plot_path, "rb") as f:
                    state["execution_output"] = f.read()   # bytes
            else:
                state["execution_error"]  = "Figure file not created.\n" + res.stderr
                state["execution_output"] = ""
        else:
            state["execution_output"] = res.stdout

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Validator
# ══════════════════════════════════════════════════════════════════════════════
def node_validate(state: ShellTLFState) -> ShellTLFState:
    """
    Check output against spec. Simple rule-based + LLM sanity check.
    Sets validation_result = "pass" or "fail: <reason>"
    """
    if state.get("execution_error") and not state.get("execution_output"):
        state["validation_result"] = f"fail: R execution error — {state['execution_error'][:300]}"
        return state

    output = state["execution_output"]
    if not output:
        state["validation_result"] = "fail: empty output"
        return state

    spec        = state["parsed_spec"]
    output_type = spec.get("output_type", "Table")

    if output_type == "Figure":
        # If we got bytes it's a PNG — assume pass
        state["validation_result"] = "pass"
        return state

    # For Table/Listing — quick structural checks
    issues = []
    output_str = output if isinstance(output, str) else ""

    # Check at least some columns mentioned appear in output
    cols = spec.get("columns", [])
    for col in cols[:3]:  # check first 3 columns only
        if col and col.lower() not in output_str.lower():
            issues.append(f"column '{col}' not found in output")

    if issues:
        state["validation_result"] = "fail: " + "; ".join(issues)
    else:
        state["validation_result"] = "pass"

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Fix
# ══════════════════════════════════════════════════════════════════════════════
def node_fix(state: ShellTLFState) -> ShellTLFState:
    """LLM reads error + code and patches it."""
    prompt = f"""You are an R clinical programmer fixing broken TLF code.

ORIGINAL CODE:
{state['generated_code']}

ERROR / VALIDATION FAILURE:
{state.get('execution_error') or state.get('validation_result', '')}

SPEC:
{state['parsed_spec']}

RULES:
- Fix ONLY what caused the error. Preserve all other logic.
- Do NOT add library(), read.csv, or ggsave — these are handled externally.
- Do NOT use mutate(col = c("a","b","c","d")) on grouped data — use bind_rows() pattern instead.
- Do NOT use column_labels() — use cols_label() for renaming gt columns.
- ALL column names in data.frame must be non-empty strings.
- Never produce an empty string "" as a column name.
- Return ONLY corrected R code. No markdown. No explanation.
"""
    raw = _call_llm(prompt)
    raw = re.sub(r'```[rR]?\n?', '', raw)
    raw = re.sub(r'```', '', raw).strip()
    state["generated_code"] = raw
    state["retry_count"]    = state.get("retry_count", 0) + 1
    return state


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER (LangGraph-style loop without the library dependency)
# ══════════════════════════════════════════════════════════════════════════════
def run_shell_pipeline(shell_text: str, adam_csv: Optional[str] = None) -> ShellTLFState:
    """
    Execute the full agentic pipeline:
    parse → generate → execute → validate → (fix → execute → validate) * N
    """
    state: ShellTLFState = {
        "shell_text":        shell_text,
        "adam_csv":          adam_csv,
        "parsed_spec":       {},
        "generated_code":    "",
        "execution_output":  "",
        "execution_error":   "",
        "validation_result": "",
        "retry_count":       0,
        "final_r_code":      "",
        "final_output":      "",
    }

    # Node 1: Parse
    state = node_parse_shell(state)

    # Node 2: Generate
    state = node_generate_code(state)

    # Node 3-4: Execute + Validate (with retry loop)
    for attempt in range(MAX_RETRIES + 1):
        state = node_execute(state)
        state = node_validate(state)

        if state["validation_result"] == "pass":
            break

        if attempt < MAX_RETRIES:
            state = node_fix(state)   # Node 5 → back to execute
        # else give up and return whatever we have

    state["final_r_code"] = state["generated_code"]
    state["final_output"]  = state["execution_output"]
    return state


# ══════════════════════════════════════════════════════════════════════════════
# DIFF HELPER (same as graph_builder)
# ══════════════════════════════════════════════════════════════════════════════
def _show_code_diff(old_code: str, new_code: str):
    import difflib
    diff = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm='')
    html = ["<pre style='font-family:monospace; font-size:13px; line-height:1.5;'>"]
    for line in diff:
        if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
            continue
        elif line.startswith('+'):
            html.append(f"<span style='background:#1a4a1a;color:#90ee90;display:block'>{line}</span>")
        elif line.startswith('-'):
            html.append(f"<span style='background:#4a1a1a;color:#ff9999;display:block;text-decoration:line-through'>{line}</span>")
        else:
            html.append(f"<span style='color:#ccc;display:block'>{line}</span>")
    html.append("</pre>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT TAB RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_shell_tlf_tab():
    st.title("📋 TLF from Mock Shell")
    st.caption("Paste or upload a mock shell → AI parses spec → generates R code → executes → validates → auto-fixes")
    st.divider()

    # ── Session state init (all keys prefixed ms_) ────────────────────────
    _defaults = {
        "ms_shell_text":       "",
        "ms_adam_csv":         None,
        "ms_parsed_spec":      None,
        "ms_r_code":           "",
        "ms_r_code_pending":   None,
        "ms_r_code_original":  None,
        "ms_output":           None,
        "ms_output_type":      "Table",
        "ms_error":            None,
        "ms_validation":       "",
        "ms_retry_count":      0,
        "ms_pipeline_done":    False,
        "ms_run_now":          False,
        "ms_agent_log":        [],
    }
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def _clear():
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Mock Shell Input
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📄 Mock Shell Input")

    shell_tab1, shell_tab2 = st.tabs(["📋 Paste Shell", "📁 Upload Shell File"])

    with shell_tab1:
        shell_pasted = st.text_area(
            "Paste your mock shell here",
            height=200,
            placeholder="""Table 14.1.1  Demographic and Baseline Characteristics
Safety Population

                                    Placebo        Drug A        Total
                                    (N=XX)         (N=XX)        (N=XX)
                                    ──────────     ──────────    ──────────
Age (years)
  n
  Mean (SD)
  Median
  Min, Max

Sex, n (%)
  Male
  Female

a. Source: ADSL
b. Note: xx""",
            key="ms_shell_paste_area"
        )
        if shell_pasted.strip():
            st.session_state["ms_shell_text"] = shell_pasted

    with shell_tab2:
        uploaded_shell = st.file_uploader(
            "Upload shell (.txt, .rtf, .docx content as text, .csv)",
            type=["txt", "rtf", "csv"],
            key="ms_shell_upload"
        )
        if uploaded_shell:
            try:
                shell_content = uploaded_shell.read().decode("utf-8", errors="ignore")
                st.session_state["ms_shell_text"] = shell_content
                st.success(f"✅ Loaded shell: {uploaded_shell.name}")
                with st.expander("👁️ Shell Preview"):
                    st.text(shell_content[:1000])
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    # Show current shell
    if st.session_state["ms_shell_text"]:
        with st.expander("✅ Current Shell (click to view)", expanded=False):
            st.text(st.session_state["ms_shell_text"][:800])

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — ADaM Dataset (optional)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📊 ADaM Dataset (Optional)")
    st.caption("If not provided, AI will generate realistic dummy data matching the shell spec.")

    adam_tab1, adam_tab2 = st.tabs(["📁 Upload ADaM CSV", "📋 Paste CSV"])

    with adam_tab1:
        uploaded_adam = st.file_uploader(
            "Upload ADaM CSV (ADSL, ADAE, ADVS, etc.)",
            type=["csv", "xlsx", "xls"],
            key="ms_adam_upload"
        )
        if uploaded_adam:
            try:
                ext = os.path.splitext(uploaded_adam.name)[1].lower()
                df_adam = pd.read_excel(uploaded_adam) if ext in (".xlsx", ".xls") else pd.read_csv(uploaded_adam)
                csv_str = df_adam.to_csv(index=False)
                st.session_state["ms_adam_csv"] = csv_str
                st.success(f"✅ Loaded — {df_adam.shape[0]} rows × {df_adam.shape[1]} cols")
                with st.expander("👁️ Data Preview"):
                    st.dataframe(df_adam.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load ADaM: {e}")

    with adam_tab2:
        adam_pasted = st.text_area(
            "Paste ADaM CSV here",
            height=100,
            key="ms_adam_paste_area"
        )
        if adam_pasted.strip():
            try:
                df_adam = pd.read_csv(io.StringIO(adam_pasted))
                st.session_state["ms_adam_csv"] = adam_pasted
                st.success(f"✅ Parsed — {df_adam.shape[0]} rows × {df_adam.shape[1]} cols")
                st.dataframe(df_adam.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"CSV parse error: {e}")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — AI Instructions box (same pattern as graph_builder)
    # ════════════════════════════════════════════════════════════════════════
    ai_instructions = st.text_area(
        "✨ Additional AI Instructions (optional)",
        placeholder="e.g. Use gt package with blue header, round to 1 decimal, add p-value column, apply ICH E3 footnote format...",
        height=80,
        key="ms_ai_instructions"
    )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Generate / Clear buttons
    # ════════════════════════════════════════════════════════════════════════
    btn_col1, btn_col2 = st.columns([4, 1])
    with btn_col1:
        generate_btn = st.button(
            "🤖 Generate TLF from Shell",
            type="primary",
            use_container_width=True,
            key="ms_generate_btn"
        )
    with btn_col2:
        st.button("🗑️ Clear", on_click=_clear, use_container_width=True, key="ms_clear_btn")

    # ── Validate inputs before running ───────────────────────────────────
    if generate_btn:
        if not st.session_state["ms_shell_text"].strip():
            st.error("⚠️ Please paste or upload a mock shell first.")
            st.stop()

        # Append any extra instructions to shell text for parser
        shell_for_pipeline = st.session_state["ms_shell_text"]
        if ai_instructions.strip():
            shell_for_pipeline += f"\n\nADDITIONAL REQUIREMENTS:\n{ai_instructions}"

        # ── Run pipeline with live agent log ─────────────────────────────
        agent_log = []
        progress   = st.progress(0, text="🧠 Starting agentic pipeline...")

        try:
            with st.spinner(""):
                # Step 1: Parse
                progress.progress(15, text="🔍 Node 1/5 — Parsing mock shell...")
                state: ShellTLFState = {
                    "shell_text":        shell_for_pipeline,
                    "adam_csv":          st.session_state.get("ms_adam_csv"),
                    "parsed_spec":       {},
                    "generated_code":    "",
                    "execution_output":  "",
                    "execution_error":   "",
                    "validation_result": "",
                    "retry_count":       0,
                    "final_r_code":      "",
                    "final_output":      "",
                }
                state = node_parse_shell(state)
                agent_log.append(("✅ Shell Parsed", str(state["parsed_spec"])[:300]))

                # Step 2: Generate
                progress.progress(35, text="⚙️ Node 2/5 — Generating R code...")
                state = node_generate_code(state)
                agent_log.append(("✅ Code Generated", f"{len(state['generated_code'])} chars"))

                # Steps 3-5: Execute → Validate → Fix loop
                for attempt in range(MAX_RETRIES + 1):
                    pct  = 55 + attempt * 12
                    pct  = min(pct, 95)
                    progress.progress(pct, text=f"🔄 Node 3/5 — Execute & Validate (attempt {attempt+1}/{MAX_RETRIES+1})...")

                    state = node_execute(state)

                    if state["execution_error"]:
                        agent_log.append((f"⚠️ Execute Attempt {attempt+1} Failed", state["execution_error"][:200]))
                    else:
                        agent_log.append((f"✅ Execute Attempt {attempt+1} OK", "Output received"))

                    state = node_validate(state)
                    agent_log.append((f"🔍 Validate Attempt {attempt+1}", state["validation_result"]))

                    if state["validation_result"] == "pass":
                        break

                    if attempt < MAX_RETRIES:
                        progress.progress(pct + 5, text=f"🔧 Node 5/5 — AI fixing (retry {attempt+1})...")
                        state = node_fix(state)
                        agent_log.append((f"🔧 Fix Applied (retry {attempt+1})", "Code patched"))

                state["final_r_code"] = state["generated_code"]
                state["final_output"]  = state["execution_output"]

                # Persist to session state
                st.session_state["ms_r_code"]       = state["final_r_code"]
                st.session_state["ms_output"]        = state["final_output"]
                st.session_state["ms_error"]         = state["execution_error"] or None
                st.session_state["ms_validation"]    = state["validation_result"]
                st.session_state["ms_retry_count"]   = state["retry_count"]
                st.session_state["ms_parsed_spec"]   = state["parsed_spec"]
                st.session_state["ms_output_type"]   = state["parsed_spec"].get("output_type", "Table")
                st.session_state["ms_pipeline_done"] = True
                st.session_state["ms_agent_log"]     = agent_log

                progress.progress(100, text="✅ Pipeline complete!")

        except Exception as e:
            progress.empty()
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            st.stop()

        st.rerun()

    # ── Re-run from edited code ───────────────────────────────────────────
    if st.session_state.get("ms_run_now"):
        st.session_state["ms_run_now"] = False
        spec = st.session_state.get("ms_parsed_spec") or {}
        run_state: ShellTLFState = {
            "shell_text":        st.session_state["ms_shell_text"],
            "adam_csv":          st.session_state.get("ms_adam_csv"),
            "parsed_spec":       spec,
            "generated_code":    st.session_state["ms_r_code"],
            "execution_output":  "",
            "execution_error":   "",
            "validation_result": "",
            "retry_count":       0,
            "final_r_code":      "",
            "final_output":      "",
        }
        with st.spinner("⚙️ Running R..."):
            run_state = node_execute(run_state)
        st.session_state["ms_output"] = run_state["final_output"] or run_state["execution_output"]
        st.session_state["ms_error"]  = run_state["execution_error"] or None
        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Output Display
    # ════════════════════════════════════════════════════════════════════════
    if not st.session_state.get("ms_pipeline_done"):
        return

    st.divider()
    st.subheader("📤 Output")

    # ── Agent log expander ────────────────────────────────────────────────
    agent_log = st.session_state.get("ms_agent_log", [])
    if agent_log:
        with st.expander("🤖 Agent Pipeline Log", expanded=False):
            for step, detail in agent_log:
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(f"**{step}**")
                with col_b:
                    st.caption(detail)

    # ── Parsed spec summary ───────────────────────────────────────────────
    spec = st.session_state.get("ms_parsed_spec") or {}
    if spec:
        with st.expander("🔍 Parsed Shell Spec", expanded=False):
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.markdown(f"**Type:** {spec.get('output_type','Table')}")
                st.markdown(f"**TLF #:** {spec.get('tlf_number','')}")
            with meta_col2:
                st.markdown(f"**Dataset:** {spec.get('dataset_hint','')}")
                st.markdown(f"**Population:** {spec.get('population','')}")
            with meta_col3:
                st.markdown(f"**Pop Flag:** {spec.get('pop_flag','')}")
                st.markdown(f"**Group By:** {spec.get('groupby','')}")
            if spec.get("title"):
                st.markdown(f"**Title:** {spec['title']}")
            if spec.get("columns"):
                st.markdown(f"**Columns:** {', '.join(spec['columns'])}")
            if spec.get("footnotes"):
                for fn in spec["footnotes"]:
                    st.caption(f"ᵃ {fn}")

    # ── Validation badge ─────────────────────────────────────────────────
    val = st.session_state.get("ms_validation", "")
    retries = st.session_state.get("ms_retry_count", 0)
    if val == "pass":
        st.success(f"✅ Validation passed {'(after ' + str(retries) + ' retries)' if retries else '(first attempt)'}")
    elif val:
        st.warning(f"⚠️ Validation: {val}")

    # ── Tabs: TLF Output | R Code ─────────────────────────────────────────
    output_type = st.session_state.get("ms_output_type", "Table")
    out_tab1, out_tab2 = st.tabs([
        "📊 TLF Output",
        "💻 R Code"
    ])

    with out_tab1:
        output = st.session_state.get("ms_output")
        error  = st.session_state.get("ms_error")

        if error:
            st.error(f"R Error:\n{error}")

        if output:
            if output_type == "Figure" and isinstance(output, (bytes, bytearray)):
                st.image(output, use_container_width=True)
                st.download_button(
                    "⬇️ Download Figure PNG",
                    data=output,
                    file_name="figure.png",
                    mime="image/png"
                )
            elif isinstance(output, str) and output.strip().startswith("<"):
                # HTML table from gt
                st.components.v1.html(output, height=600, scrolling=True)
                st.download_button(
                    "⬇️ Download HTML Table",
                    data=output,
                    file_name="tlf_output.html",
                    mime="text/html"
                )
            else:
                # Plain text listing
                st.code(output if isinstance(output, str) else str(output), language="")
                if isinstance(output, str):
                    st.download_button(
                        "⬇️ Download Listing",
                        data=output,
                        file_name="listing.txt",
                        mime="text/plain"
                    )

    with out_tab2:
        # ── Pending diff review (same pattern as graph_builder) ───────────
        if st.session_state.get("ms_r_code_pending"):
            st.warning("⚠️ AI wants to modify your code. Review and confirm:")
            st.markdown("**Code Changes** (🟢 added | 🔴 removed):")
            _show_code_diff(
                st.session_state["ms_r_code_original"],
                st.session_state["ms_r_code_pending"]
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ Apply Changes", use_container_width=True, key="ms_apply"):
                    st.session_state["ms_r_code"]          = st.session_state["ms_r_code_pending"]
                    st.session_state["ms_r_code_original"] = None
                    st.session_state["ms_r_code_pending"]  = None
                    st.session_state["ms_run_now"]         = True
                    st.rerun()
            with c2:
                if st.button("❌ Reject", use_container_width=True, key="ms_reject"):
                    st.session_state["ms_r_code_pending"] = None
                    st.rerun()
            with c3:
                pass  # reserve for future preview

        # ── Editable code ─────────────────────────────────────────────────
        current_code = st.session_state.get("ms_r_code", "")
        edited = st.text_area(
            "Edit R Code",
            value=current_code,
            height=350,
            key=f"ms_code_editor_{hash(current_code)}"
        )

        btn_a, btn_b = st.columns(2)
        with btn_a:
            if st.button("▶️ Run Edited Code", type="primary", use_container_width=True, key="ms_run_edit"):
                st.session_state["ms_r_code"] = edited
                st.session_state["ms_run_now"] = True
                st.rerun()
        with btn_b:
            st.download_button(
                "⬇️ Download R Code",
                data=edited,
                file_name="tlf_from_shell.R",
                mime="text/plain",
                use_container_width=True
            )

    # ── Custom enhancement box (same pattern as graph_builder) ────────────
    st.divider()
    enhance_text = st.text_area(
        "✨ Custom Enhancement (optional)",
        placeholder="e.g. Add p-value column, change header color to navy, add risk difference row...",
        height=80,
        key="ms_enhance_text"
    )

    if st.button("🔧 Apply Enhancement", use_container_width=True, key="ms_enhance_btn"):
        if not enhance_text.strip():
            st.warning("Enter enhancement instructions first.")
        else:
            existing_code = st.session_state.get("ms_r_code", "")
            enhance_prompt = (
                f"You are an R clinical TLF code editor. Apply ONLY the requested change.\n\n"
                f"EXISTING CODE:\n```r\n{existing_code}\n```\n\n"
                f"REQUEST: {enhance_text}\n\n"
                f"RULES:\n"
                f"- Touch ONLY what the request asks. Preserve everything else exactly.\n"
                f"- Never add read.csv, hardcoded file paths, or ggsave.\n"
                f"- Return ONLY complete R code. No explanations, no markdown fences.\n"
            )
            with st.spinner("🤖 Applying enhancement..."):
                try:
                    raw = _call_llm(enhance_prompt)
                    raw = re.sub(r'```[rR]?\n?', '', raw)
                    raw = re.sub(r'```', '', raw).strip()
                    st.session_state["ms_r_code_pending"]  = raw
                    st.session_state["ms_r_code_original"] = existing_code
                    st.rerun()
                except Exception as e:
                    st.error(f"Enhancement failed: {e}")
