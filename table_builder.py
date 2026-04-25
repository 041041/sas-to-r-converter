import os, re, subprocess, tempfile
import pandas as pd
import streamlit as st
from groq import Groq
from google import genai 

# ─────────────────────────────────────────────
# CONSTANTS  — easy to extend
# ─────────────────────────────────────────────
TABLE_TYPES = [
    "Table 1 — Demographics & Baseline",
    "Adverse Events Summary",
]

STAT_OPTIONS = ["Mean (SD)", "Median (IQR)", "Mean (SD) + Median (IQR)"]

OUTPUT_FORMATS = ["HTML (.html)"]

# ─────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────
def _get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

def _make_clients():
    gemini = genai.Client(api_key=_get_secret("GEMINI_API_KEY"))
    groq   = Groq(api_key=_get_secret("GROQ_API_KEY"))
    return gemini, groq

# ─────────────────────────────────────────────
# R PACKAGE AUTO-INSTALLER
# ─────────────────────────────────────────────
REQUIRED_R_PACKAGES = ["gtsummary", "flextable", "officer", "dplyr", "gt", "broom", "htmltools"]

def ensure_r_packages():
    """Install missing R packages silently on first run."""
    install_script = """
pkgs <- c({pkgs})
user_lib <- path.expand("~/R/library")
if (!dir.exists(user_lib)) dir.create(user_lib, recursive=TRUE)
.libPaths(c(user_lib, .libPaths()))
missing <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(missing) > 0) {{
  message("Installing: ", paste(missing, collapse=", "))
  install.packages(missing, repos="https://cloud.r-project.org", lib=user_lib, quiet=FALSE)
}}
# Force upgrade broom if version is too old
if (packageVersion("broom") < "1.0.8") {{
  install.packages("broom", repos="https://cloud.r-project.org", lib=user_lib, quiet=FALSE)
}}
message("All packages ready")
""".format(pkgs=", ".join(f'"{p}"' for p in REQUIRED_R_PACKAGES))

    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(install_script)
        script_path = f.name

    result = subprocess.run(
        ["Rscript", script_path],
        capture_output=True, text=True, timeout=120
    )
    os.unlink(script_path)
    return result.returncode == 0, result.stderr

# ─────────────────────────────────────────────
# CODE GENERATORS  — pure Python, no LLM
# ─────────────────────────────────────────────
def generate_table1_code(selections):
    """Generate gtsummary Table 1 R code from selections."""
    vars_list    = selections["variables"]
    group_col    = selections.get("group_col")
    title        = selections.get("title", "Table 1 — Baseline Characteristics")
    stat_option  = selections.get("stat_option", "Mean (SD)")
    output_fmt   = selections.get("output_format", "Word (.docx)")

    vars_r = "c(" + ", ".join(f'"{v}"' for v in vars_list) + ")"

    if stat_option == "Mean (SD)":
        stat_str = '"{mean} ({sd})"'
    elif stat_option == "Median (IQR)":
        stat_str = '"{median} ({p25}, {p75})"'
    else:
        stat_str = '"{mean} ({sd}); {median} ({p25}, {p75})"'

    # Auto-exclude subject ID from variables
    subj_col = selections.get("subj_col")
    clean_vars = [v for v in vars_list if v != subj_col]
    vars_r = "c(" + ", ".join(f'"{v}"' for v in clean_vars) + ")"

    # Build factor conversion for character columns
    factor_hint = f"""
# Convert character columns to factors for proper categorical display
df <- df %>% mutate(across(where(is.character), as.factor))
"""

if group_col:
        tbl_code = f"""
{factor_hint}
tbl <- df %>%
  select(all_of(c({vars_r}, "{group_col}"))) %>%
  tbl_summary(
    by = {group_col},
    statistic = list(all_continuous() ~ {stat_str},
                     all_categorical() ~ "{{n}} ({{p}}%)"),
    missing = "no"
  ) %>%
  add_overall(last = TRUE) %>%
  add_p() %>%
  bold_labels() %>%
  modify_caption("**{title}**")
"""
    else:
        tbl_code = f"""
{factor_hint}
tbl <- df %>%
  select(all_of({vars_r})) %>%
  tbl_summary(
    statistic = list(all_continuous() ~ {stat_str},
                     all_categorical() ~ "{{n}} ({{p}}%)"),
    missing = "no"
  ) %>%
  bold_labels() %>%
  modify_caption("**{title}**")
"""

    export_code = f"""
gt_tbl <- as_gt(tbl)
html_content <- as_raw_html(gt_tbl)
writeLines(html_content, html_path)
writeLines(html_content, output_path)
"""

    code = f"""library(dplyr)
library(gtsummary)
library(flextable)
library(officer)
library(gt)

# df is already loaded
{tbl_code}
{export_code}
cat("TABLE_DONE")
"""
    return code
    
def merge_footnotes(old_code, new_code):
    """Always preserve all footnotes from old code in new code."""
    old_match = re.search(r'modify_footnote\s*\(\s*everything\(\)\s*~\s*[\'"]([^\'"]+)[\'"]', old_code)
    new_match = re.search(r'modify_footnote\s*\(\s*everything\(\)\s*~\s*[\'"]([^\'"]+)[\'"]', new_code)

    if old_match:
        old_text = old_match.group(1)
        if new_match:
            new_text = new_match.group(1)
            if old_text not in new_text:
                combined = f"{old_text}; {new_text}"
                new_code = re.sub(
                    r'modify_footnote\s*\(\s*everything\(\)\s*~\s*[\'"][^\'"]+[\'"]\s*\)',
                    f'modify_footnote(everything() ~ "{combined}")',
                    new_code
                )
        else:
            new_code = new_code.replace(
                'modify_caption(',
                f'modify_footnote(everything() ~ "{old_text}") %>%\n  modify_caption('
            )
    return new_code
    
def generate_ae_code(selections):
    """Generate Adverse Events summary R code from selections."""
    soc_col     = selections.get("soc_col", "SOC")
    pt_col      = selections.get("pt_col", "PT")
    group_col   = selections.get("group_col")
    subj_col    = selections.get("subj_col", "USUBJID")
    title       = selections.get("title", "Adverse Events Summary")
    output_fmt  = selections.get("output_format", "Word (.docx)")

    if group_col:
        count_code = f"""
ae_summary <- df %>%
  group_by({soc_col}, {pt_col}, {group_col}) %>%
  summarise(n = n_distinct({subj_col}), .groups = "drop") %>%
  group_by({group_col}) %>%
  mutate(total = n_distinct(df${subj_col}),
         pct = round(n / total * 100, 1),
         n_pct = paste0(n, " (", pct, "%)")) %>%
  select({soc_col}, {pt_col}, {group_col}, n_pct) %>%
  tidyr::pivot_wider(names_from = {group_col}, values_from = n_pct, values_fill = "0 (0.0%)")
"""
    else:
        count_code = f"""
total_subj <- n_distinct(df${subj_col})
ae_summary <- df %>%
  group_by({soc_col}, {pt_col}) %>%
  summarise(n = n_distinct({subj_col}), .groups = "drop") %>%
  mutate(pct = round(n / total_subj * 100, 1),
         `n (%)` = paste0(n, " (", pct, "%)")) %>%
  select({soc_col}, {pt_col}, `n (%)`)
"""

    export_code = f"""
gt_tbl <- gt(ae_summary) %>%
  tab_header(title = md("**{title}**")) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  opt_stylize(style = 1)
html_content <- as_raw_html(gt_tbl)
writeLines(html_content, html_path)
writeLines(html_content, output_path)
"""

    code = f"""library(dplyr)
library(tidyr)
library(flextable)
library(officer)
library(gt)

# df is already loaded
{count_code}
{export_code}
cat("TABLE_DONE")
"""
    return code


# ─────────────────────────────────────────────
# LLM ENHANCEMENT  — same pattern as graph_builder
# ─────────────────────────────────────────────
def build_enhance_prompt(current_code, custom_request):
    existing_footnote = extract_existing_footnotes(current_code)
    footnote_instruction = (
        f"8. Code already has this footnote: '{existing_footnote}'. "
        f"You MUST preserve it and append new footnote text separated by '; '.\n"
        if existing_footnote else
        f"8. No existing footnote — add new one if requested.\n"
    )
    return (
        f"You are a clinical R table code editor.\n\n"
        f"EXISTING CODE:\n```r\n{current_code}\n```\n\n"
        f"REQUEST: {custom_request}\n\n"
        f"RULES:\n"
        f"1. Touch ONLY what the request asks. Preserve everything else exactly.\n"
        f"2. Never add data loading code (read.csv, read.xlsx, hardcoded data).\n"
        f"3. Never remove output_path, html_path, writeLines, or cat('TABLE_DONE').\n"
        f"4. Keep all existing library() calls.\n\n"
        f"GTSUMMARY RULES:\n"
        f"5. Only use REAL functions: modify_caption, modify_header, modify_footnote, add_overall, add_p, bold_labels, bold_levels, italicize_labels.\n"
        f"6. modify_footnote syntax: modify_footnote(everything() ~ 'text') — NEVER plain string.\n"
        f"7. Each function appears AT MOST once — if already exists, REPLACE it not add another.\n"
        f"{footnote_instruction}"
        f"GT RULES:\n"
        f"9. gt functions (tab_style, tab_options, cols_move) ONLY after as_gt(tbl).\n"
        f"10. NEVER apply gt functions on tbl_summary objects.\n"
        f"11. NEVER invent function names.\n\n"
        f"Return ONLY the complete modified R code. No explanations."
    )

def call_llm(prompt, groq_client, gemini_client):
    """Try Groq first, fall back to Gemini."""
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content
    except Exception:
        pass
    try:
        return gemini_client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        ).text
    except Exception:
        return None


def clean_llm_output(raw):
    """Strip markdown fences and dangerous lines from LLM output."""
    raw = re.sub(r'```[rR]?\n?', '', raw)
    raw = re.sub(r'```', '', raw)
    raw = re.sub(r'read\.csv\s*\(.*?\)', '', raw)
    raw = re.sub(r'read\.xlsx\s*\(.*?\)', '', raw)

    # Remove gt functions incorrectly applied before as_gt() conversion
    # These must only be applied on gt objects, not tbl_summary objects
    invalid_fns_on_tbl = [
        r'\s*%>%\s*tab_style\s*\([^)]*\)',
        r'\s*%>%\s*tab_options\s*\([^)]*\)',
        r'\s*%>%\s*cols_move\s*\([^)]*\)',
        r'\s*%>%\s*cols_align\s*\([^)]*\)',
    ]
    # Only remove these if they appear before as_gt()
    as_gt_pos = raw.find('as_gt(')
    if as_gt_pos > 0:
        before_as_gt = raw[:as_gt_pos]
        after_as_gt  = raw[as_gt_pos:]
        for fn in invalid_fns_on_tbl:
            before_as_gt = re.sub(fn, '', before_as_gt)
        raw = before_as_gt + after_as_gt

    # Remove hallucinated gtsummary functions that don't exist
    invalid_fns = [
        r'modify_title\s*\([^)]*\)\s*%>%?',
        r'modify_title\s*\([^)]*\)',
        r'add_significance\s*\([^)]*\)\s*%>%?',
        r'bold_p\s*\([^)]*\)\s*%>%?',
        r'italicize_levels\s*\([^)]*\)\s*%>%?',
    ]
    for fn in invalid_fns:
        raw = re.sub(fn, '', raw)

# Remove repeated modify_header blocks — keep only the first occurrence
    # Strategy: find all modify_header calls and keep only the last one
    # (last is most likely the intended one from the new request)
    modify_header_pattern = re.compile(
        r'%>%\s*modify_header\s*\([^)]*\)',
        re.DOTALL
    )
    matches = list(modify_header_pattern.finditer(raw))
    if len(matches) > 1:
        # Keep only the last modify_header, remove all previous ones
        for match in matches[:-1]:
            raw = raw.replace(match.group(0), '', 1)

    # Also deduplicate any other repeated pipe steps line by line
    lines = raw.splitlines()
    deduped = []
    prev_stripped = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == prev_stripped:
            continue
        deduped.append(line)
        if stripped:
            prev_stripped = stripped
    raw = '\n'.join(deduped)
    return raw

# ─────────────────────────────────────────────
# R EXECUTOR
# ─────────────────────────────────────────────
def extract_existing_footnotes(code):
    """Extract existing footnote text from R code."""
    match = re.search(r"modify_footnote\s*\(\s*everything\(\)\s*~\s*['\"]([^'\"]+)['\"]", code)
    return match.group(1) if match else None
def execute_table(r_code, df, output_format):
    """Run R code, return (html_str, output_bytes, extension, stderr)."""
    ext = ".html"

    with tempfile.TemporaryDirectory() as d:
        inp_path    = os.path.join(d, "input.csv")
        out_path    = os.path.join(d, f"output_table{ext}")
        html_path   = os.path.join(d, "output_table.html")
        script_path = os.path.join(d, "script.R")

        df.to_csv(inp_path, index=False)

        full_script = "\n".join([
            "user_lib <- path.expand('~/R/library')",
            "if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))",
            "suppressPackageStartupMessages({",
            "  library(dplyr); library(gtsummary); library(flextable)",
            "  library(officer); library(gt); library(tidyr)",
            "})",
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE)',
            f'output_path <- "{out_path}"',
            f'html_path <- "{html_path}"',
            r_code,
        ])

        with open(script_path, "w") as f:
            f.write(full_script)

        res = subprocess.run(
            ["Rscript", script_path],
            capture_output=True, text=True, timeout=60
        )

        if res.returncode != 0:
            raise RuntimeError(f"R Error:\n{res.stderr}")

        if not os.path.exists(out_path):
            raise RuntimeError("Output file was not created.\n" + res.stderr)

        # Read HTML for screen display
        html_str = ""
        if os.path.exists(html_path):
            with open(html_path, "r") as f:
                html_str = f.read()

        with open(out_path, "rb") as f:
            return html_str, f.read(), ext, res.stderr


# ─────────────────────────────────────────────
# CODE DIFF DISPLAY  — reused from graph_builder
# ─────────────────────────────────────────────
def show_code_diff(old_code, new_code):
    import difflib
    diff = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm='')
    html = ["<pre style='font-family:monospace; font-size:13px; line-height:1.5;'>"]
    for line in diff:
        if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
            continue
        elif line.startswith('+'):
            html.append(f"<span style='background:#1a4a1a; color:#90ee90; display:block'>{line}</span>")
        elif line.startswith('-'):
            html.append(f"<span style='background:#4a1a1a; color:#ff9999; display:block; text-decoration:line-through'>{line}</span>")
        else:
            html.append(f"<span style='color:#ccc; display:block'>{line}</span>")
    html.append("</pre>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN TAB RENDERER
# ─────────────────────────────────────────────
def render_table_builder_tab():
    st.subheader("🏥 Clinical Table Builder")
    st.caption("Upload data → configure table → get R code + downloadable table")
    st.divider()

    # ── Session state init ──────────────────────────────────────────────
    # Sentinel prevents stale pending state on fresh load
    if "tbl_initialized" not in st.session_state:
        st.session_state["tbl_r_code_pending"]  = None
        st.session_state["tbl_r_code_original"] = None
        st.session_state["tbl_preview_bytes"]   = None
        st.session_state["_tbl_run_now"]        = False
        st.session_state["tbl_initialized"]     = True

    for key, default in {
        "tbl_df":              None,
        "tbl_r_code":          "",
        "tbl_output_bytes":    None,
        "tbl_html":            None,
        "tbl_output_ext":      ".docx",
        "tbl_log":             "",
        "tbl_error":           None,
        "tbl_r_code_pending":  None,
        "tbl_r_code_original": None,
        "tbl_preview_bytes":   None,
        "tbl_preview_html":    None,
        "_tbl_run_now":        False,
        "tbl_custom_text":     "",
        "tbl_output_format":   "Word (.docx)",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── R package check ─────────────────────────────────────────────────
    # ── R package check (runs once per session only) ─────────────────────
    if "tbl_pkgs_checked" not in st.session_state:
        st.info("🔧 Installing R packages on first run — this takes 2-5 minutes...")
        ok, err = ensure_r_packages()
        st.session_state["tbl_pkgs_checked"] = True
        if ok:
            st.success("✅ R packages ready!")
            st.rerun()
        else:
            st.warning(f"Some R packages may be missing:\n{err}")

    gemini_client, groq_client = _make_clients()

    # ── Data upload ─────────────────────────────────────────────────────
    st.subheader("📁 Upload Data")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="tbl_upload"
    )

    df = st.session_state.get("tbl_df")

    if uploaded:
        try:
            ext = os.path.splitext(uploaded.name)[1].lower()
            df = pd.read_excel(uploaded) if ext in (".xlsx", ".xls") else pd.read_csv(uploaded)
            st.session_state["tbl_df"] = df
            st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
            with st.expander("👁️ Preview Data", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    if df is None:
        st.info("👆 Upload a CSV or Excel file to get started.")
        return

    st.divider()

    # ── Configure table ─────────────────────────────────────────────────
    st.subheader("⚙️ Configure Table")
    cols = df.columns.tolist()
    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    all_with_none = ["None"] + cols

    # Row 1: table type, group col, subject col, output format
    r1a, r1b, r1c, r1d = st.columns(4)
    with r1a:
        table_type = st.selectbox("📋 Table Type", TABLE_TYPES)
    with r1b:
        group_col = st.selectbox("👥 Group / Treatment Col", all_with_none, index=0)
    with r1c:
        subj_col = st.selectbox("🔑 Subject ID Col", all_with_none, index=0)
    with r1d:
        output_format = st.selectbox("📄 Output Format", OUTPUT_FORMATS)
        st.session_state["tbl_output_format"] = output_format

    # Row 2 — Table 1 specific
    if "Table 1" in table_type:
        r2a, r2b, r2c = st.columns([2, 1, 1])
        with r2a:
            variables = st.multiselect(
                "📊 Variables to Summarise",
                cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
            )
        with r2b:
            stat_option = st.selectbox("📐 Continuous Stats", STAT_OPTIONS)
        with r2c:
            title = st.text_input("📝 Table Title", value="Table 1 — Baseline Characteristics")

        selections = {
            "table_type":    table_type,
            "variables":     variables,
            "group_col":     group_col if group_col != "None" else None,
            "subj_col":      subj_col  if subj_col  != "None" else None,
            "stat_option":   stat_option,
            "title":         title,
            "output_format": output_format,
        }

    # Row 2 — AE table specific
    elif "Adverse Events" in table_type:
        r2a, r2b, r2c, r2d = st.columns(4)
        with r2a:
            soc_col = st.selectbox("🏷️ SOC Column", cols)
        with r2b:
            pt_col  = st.selectbox("💊 PT Column", cols)
        with r2c:
            title   = st.text_input("📝 Table Title", value="Adverse Events Summary")
        with r2d:
            st.write("")  # spacer

        selections = {
            "table_type":    table_type,
            "soc_col":       soc_col,
            "pt_col":        pt_col,
            "group_col":     group_col if group_col != "None" else None,
            "subj_col":      subj_col  if subj_col  != "None" else "USUBJID",
            "title":         title,
            "output_format": output_format,
        }

    st.divider()

    # ── Output (above custom box, same as graph_builder) ────────────────
    if st.session_state.get("tbl_r_code"):
        st.subheader("📤 Output")
        out1, out2 = st.tabs(["📊 Table", "💻 R Code"])

        with out1:
            if st.session_state.get("tbl_html"):
                import streamlit.components.v1 as components
                components.html(st.session_state["tbl_html"], height=600, scrolling=True)
                st.download_button(
                    "⬇️ Download HTML",
                    data=st.session_state["tbl_output_bytes"],
                    file_name="clinical_table.html",
                    mime="text/html",
                    use_container_width=True
                )
            elif st.session_state.get("tbl_error"):
                st.error(st.session_state["tbl_error"])
            else:
                st.info("Click 'Generate Table' to produce output.")

        with out2:
            edited_code = st.text_area(
                "Edit R Code",
                value=st.session_state.get("tbl_r_code", ""),
                height=300,
                # key tied to code content so it refreshes when code changes
                key=f"tbl_edited_code_{hash(st.session_state.get('tbl_r_code', ''))}"
            )
            btn1, btn2 = st.columns(2)
            with btn1:
                run_edited = st.button("▶️ Run Edited Code", type="primary", use_container_width=True)
            with btn2:
                st.download_button(
                    "⬇️ Download R Code",
                    data=edited_code,
                    file_name="clinical_table.R",
                    mime="text/plain",
                    use_container_width=True
                )
            if run_edited:
                with st.spinner("Running updated code..."):
                    try:
                        html_str, out_bytes, ext, r_log = execute_table(
                            st.session_state["tbl_r_code"],
                            st.session_state["tbl_df"],
                            st.session_state["tbl_output_format"]
                        )
                        st.session_state["tbl_output_bytes"] = out_bytes
                        st.session_state["tbl_output_ext"]   = ext
                        st.session_state["tbl_html"]         = html_str
                        st.session_state["tbl_log"]          = r_log
                        st.session_state["tbl_error"]        = None
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))

            log = st.session_state.get("tbl_log", "")
            if log:
                with st.expander("📋 R Log"):
                    st.code(log, language="bash")

    st.divider()

    # ── Custom enhancement box ───────────────────────────────────────────
    # key bound to session_state so text survives reruns
    custom_request = st.text_area(
        "✨ Custom Enhancement (optional)",
        placeholder="e.g. Add footnote, bold p-values < 0.05, change font size, add spanning header...",
        height=80,
        key="tbl_custom_text",
    )

    # ── Generate button ──────────────────────────────────────────────────
    if st.button("🏥 Generate Table", type="primary", use_container_width=True):
        # Validation
        if "Table 1" in table_type and not selections.get("variables"):
            st.error("⚠️ Please select at least one variable to summarise.")
            st.stop()

        with st.spinner("🤖 Generating R code..."):
            try:
                if "Table 1" in table_type:
                    r_code = generate_table1_code(selections)
                else:
                    r_code = generate_ae_code(selections)

                # Use accepted code as base to preserve previous changes
                r_code_for_enhancement = st.session_state.get("tbl_r_code") or r_code

                if custom_request.strip():
                    prompt = build_enhance_prompt(r_code_for_enhancement, custom_request)
                    raw    = call_llm(prompt, groq_client, gemini_client)

                    if raw:
                        enhanced_code = clean_llm_output(raw)
                        # Merge footnotes from original to preserve all
                        enhanced_code = merge_footnotes(r_code_for_enhancement, enhanced_code)
                        st.session_state["tbl_r_code_pending"]  = enhanced_code
                        st.session_state["tbl_r_code_original"] = r_code_for_enhancement
                        # Don't overwrite tbl_r_code — keep accepted version intact
                        # so next enhancement still builds on accepted code
                        st.session_state["tbl_df"]              = df
                        st.session_state["tbl_preview_bytes"]   = None
                        st.rerun()
                    else:
                        st.warning("⚠️ Enhancement failed, using base code.")
                        st.session_state["tbl_r_code_pending"] = None
                        st.session_state["tbl_r_code"]         = r_code
                        st.session_state["tbl_df"]             = df
                        st.session_state["_tbl_run_now"]       = True

                else:
                    st.session_state["tbl_r_code_pending"] = None
                    st.session_state["tbl_r_code"]         = r_code
                    st.session_state["tbl_df"]             = df
                    st.session_state["_tbl_run_now"]       = True

            except Exception as e:
                import traceback
                st.error(f"Code generation error: {e}")
                st.code(traceback.format_exc())
                st.stop()

    # ── R execution block ─────────────────────────────────────────────────
    if st.session_state.get("_tbl_run_now") and not st.session_state.get("tbl_r_code_pending"):
        st.session_state["_tbl_run_now"] = False
        with st.spinner("⚙️ Running R..."):
            try:
                html_str, out_bytes, ext, r_log = execute_table(
                    st.session_state["tbl_r_code"],
                    st.session_state["tbl_df"],
                    st.session_state["tbl_output_format"]
                )
                st.session_state["tbl_output_bytes"] = out_bytes
                st.session_state["tbl_output_ext"]   = ext
                st.session_state["tbl_html"]         = html_str
                st.session_state["tbl_log"]          = r_log
                st.session_state["tbl_error"]        = None
            except RuntimeError as e:
                st.session_state["tbl_error"]        = str(e)
                st.session_state["tbl_output_bytes"] = None
        st.rerun()

    # ── Review block ──────────────────────────────────────────────────────
    if st.session_state.get("tbl_r_code_pending"):
        st.warning("⚠️ AI wants to modify your code. Review and confirm:")
        st.markdown("**Code Changes** (🟢 added | 🔴 removed):")
        show_code_diff(
            st.session_state["tbl_r_code_original"],
            st.session_state["tbl_r_code_pending"]
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("✅ Apply Changes", use_container_width=True, key="tbl_apply"):
                # Save the full accepted code — this is what next enhancement will build on
                accepted = st.session_state["tbl_r_code_pending"]
                st.session_state["tbl_r_code"]          = accepted
                st.session_state["tbl_r_code_original"] = None
                st.session_state["tbl_r_code_pending"]  = None
                st.session_state["tbl_preview_bytes"]   = None
                st.session_state["_tbl_run_now"]        = True
                st.rerun()

        with c2:
            if st.button("👁️ Preview", use_container_width=True, key="tbl_preview"):
                with st.spinner("Generating preview..."):
                    try:
                        prev_html, prev_bytes, _, _ = execute_table(
                            st.session_state["tbl_r_code_pending"],
                            st.session_state["tbl_df"],
                            st.session_state["tbl_output_format"]
                        )
                        st.session_state["tbl_preview_bytes"] = prev_bytes
                        st.session_state["tbl_preview_html"]  = prev_html
                        st.rerun()
                    except RuntimeError as e:
                        st.error(f"Preview failed: {e}")

        with c3:
            if st.button("❌ Reject Changes", use_container_width=True, key="tbl_reject"):
                # tbl_output_bytes already holds the last good table — nothing lost
                st.session_state["tbl_r_code_pending"] = None
                st.session_state["tbl_preview_bytes"]  = None
                st.rerun()

        # Side-by-side preview (current vs pending)
        if st.session_state.get("tbl_preview_html"):
            st.markdown("**👁️ Preview (not applied yet):**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current (accepted):**")
                if st.session_state.get("tbl_html"):
                    st.components.v1.html(
                    f"<div style='background:white; padding:10px;'>{st.session_state['tbl_html']}</div>",
                    height=400,
                    scrolling=True
                 )
            with col2:
                st.markdown("**Preview (pending):**")
                st.components.v1.html(
                    f"<div style='background:white; padding:10px;'>{st.session_state['tbl_preview_html']}</div>",
                    height=400,
                    scrolling=True
                )
