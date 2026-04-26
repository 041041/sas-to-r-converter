import os, re, subprocess, tempfile
import pandas as pd
import streamlit as st
from groq import Groq
from google import genai

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TABLE_TYPES  = ["Table 1 — Demographics & Baseline", "Adverse Events Summary"]
STAT_OPTIONS = ["Mean (SD)", "Median (IQR)", "Mean (SD) + Median (IQR)"]

# ─────────────────────────────────────────────
# API CLIENTS
# ─────────────────────────────────────────────
def _get_secret(key):F
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
REQUIRED_R_PACKAGES = ["gtsummary", "flextable", "officer", "dplyr", "gt", "broom", "htmltools", "tidyr"]

def ensure_r_packages():
    install_script = """
pkgs <- c("gtsummary", "flextable", "officer", "dplyr", "gt", "broom", "htmltools", "tidyr")
user_lib <- path.expand("~/R/library")
if (!dir.exists(user_lib)) dir.create(user_lib, recursive=TRUE)
.libPaths(c(user_lib, .libPaths()))
missing <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(missing) > 0) {
  message("Installing: ", paste(missing, collapse=", "))
  install.packages(missing, repos="https://cloud.r-project.org", lib=user_lib, quiet=FALSE)
}
if (packageVersion("broom") < "1.0.8") {
  install.packages("broom", repos="https://cloud.r-project.org", lib=user_lib, quiet=FALSE)
}
message("All packages ready")
"""
    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(install_script)
        script_path = f.name
    result = subprocess.run(["Rscript", script_path], capture_output=True, text=True, timeout=600)
    os.unlink(script_path)
    return result.returncode == 0, result.stderr

# ─────────────────────────────────────────────
# CODE GENERATORS
# ─────────────────────────────────────────────
def generate_table1_code(selections):
    vars_list   = selections["variables"]
    group_col   = selections.get("group_col")
    title       = selections.get("title", "Table 1 — Baseline Characteristics")
    stat_option = selections.get("stat_option", "Mean (SD)")
    subj_col    = selections.get("subj_col")

    clean_vars = [v for v in vars_list if v != subj_col]
    vars_r = "c(" + ", ".join(f'"{v}"' for v in clean_vars) + ")"

    if stat_option == "Mean (SD)":
        stat_str = '"{mean} ({sd})"'
    elif stat_option == "Median (IQR)":
        stat_str = '"{median} ({p25}, {p75})"'
    else:
        stat_str = '"{mean} ({sd}); {median} ({p25}, {p75})"'

    factor_line = "df <- df %>% mutate(across(where(is.character), as.factor))"

    if group_col:
        tbl_code = (
            f"{factor_line}\n\n"
            f"tbl <- df %>%\n"
            f'  select(all_of(c({vars_r}, "{group_col}"))) %>%\n'
            f"  tbl_summary(\n"
            f"    by = {group_col},\n"
            f"    statistic = list(all_continuous() ~ {stat_str},\n"
            f'                     all_categorical() ~ "{{n}} ({{p}}%)"),\n'
            f'    missing = "no"\n'
            f"  ) %>%\n"
            f"  add_overall(last = FALSE) %>%\n"
            f"  add_p() %>%\n"
            f"  bold_labels() %>%\n"
            f'  modify_caption("**{title}**")'
        )
    else:
        tbl_code = (
            f"{factor_line}\n\n"
            f"tbl <- df %>%\n"
            f"  select(all_of({vars_r})) %>%\n"
            f"  tbl_summary(\n"
            f"    statistic = list(all_continuous() ~ {stat_str},\n"
            f'                     all_categorical() ~ "{{n}} ({{p}}%)"),\n'
            f'    missing = "no"\n'
            f"  ) %>%\n"
            f"  bold_labels() %>%\n"
            f'  modify_caption("**{title}**")'
        )

    return f"""library(dplyr)
library(gtsummary)
library(gt)

# df is already loaded
{tbl_code}

gt_tbl <- as_gt(tbl)
html_content <- as_raw_html(gt_tbl)
writeLines(html_content, html_path)
writeLines(html_content, output_path)
cat("TABLE_DONE")
"""


def generate_ae_code(selections):
    soc_col   = selections.get("soc_col", "SOC")
    pt_col    = selections.get("pt_col", "PT")
    group_col = selections.get("group_col")
    subj_col  = selections.get("subj_col", "USUBJID")
    title     = selections.get("title", "Adverse Events Summary")

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

    return f"""library(dplyr)
library(tidyr)
library(gt)

# df is already loaded
{count_code}

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
cat("TABLE_DONE")
"""

# ─────────────────────────────────────────────
# FOOTNOTE HANDLING — pure Python, no LLM
# ─────────────────────────────────────────────
def extract_existing_footnotes(code):
    matches = re.findall(
        r'modify_footnote\s*\([^~]+~\s*[\'"]([^\'"]+)[\'"]',
        code, re.DOTALL
    )
    return matches


    
def extract_footnote_text_from_request(custom_request):
    quoted = re.search(r'["\']([^"\']+)["\']', custom_request)
    if quoted:
        return quoted.group(1).strip()
    text = custom_request
    for prefix in [
        "please add footnote", "add a footnote saying",
        "add footnote saying", "add footnote:",
        "add a footnote", "add footnote",
        "add note", "add annotation"
    ]:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE).strip()
    return text.strip('"\'').strip()


# ─────────────────────────────────────────────
# LLM ENHANCEMENT
# ─────────────────────────────────────────────
def build_enhance_prompt(current_code, custom_request):
    return (
        f"You are a clinical R table code editor. Apply ONLY the requested change.\n\n"
        f"EXISTING CODE:\n```r\n{current_code}\n```\n\n"
        f"REQUEST: {custom_request}\n\n"
        f"RULES:\n"
        f"1. Touch ONLY what the request asks. Preserve everything else exactly.\n"
        f"2. Never add data loading code (read.csv, read.xlsx, hardcoded data).\n"
        f"3. Never remove output_path, html_path, writeLines, or cat('TABLE_DONE').\n"
        f"4. Keep all existing library() calls.\n"
        f"5. Only use REAL gtsummary functions: modify_caption, modify_header, modify_footnote, "
        f"add_overall, add_p, bold_labels, bold_levels, italicize_labels.\n"
        f"6. gt functions (tab_style, tab_options, cols_move) ONLY after as_gt(tbl).\n"
        f"7. NEVER apply gt functions on tbl_summary objects.\n"
        f"8. NEVER invent function names.\n"
        f"9. Each function appears AT MOST once in the pipe — never repeat.\n"
        f"10. Do NOT touch any modify_footnote() calls — preserve them exactly.\n\n"
        f"Return ONLY complete R code. No explanations, no markdown fences."
    )


def call_llm(prompt, groq_client, gemini_client):
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
    raw = re.sub(r'```[rR]?\n?', '', raw)
    raw = re.sub(r'```', '', raw)
    raw = re.sub(r'read\.csv\s*\(.*?\)', '', raw)
    raw = re.sub(r'read\.xlsx\s*\(.*?\)', '', raw)

    as_gt_pos = raw.find('as_gt(')
    if as_gt_pos > 0:
        before = raw[:as_gt_pos]
        after  = raw[as_gt_pos:]
        for fn in [r'\s*%>%\s*tab_style\s*\([^)]*\)',
                   r'\s*%>%\s*tab_options\s*\([^)]*\)',
                   r'\s*%>%\s*cols_move\s*\([^)]*\)',
                   r'\s*%>%\s*cols_align\s*\([^)]*\)']:
            before = re.sub(fn, '', before)
        raw = before + after

    for fn in [r'modify_title\s*\([^)]*\)\s*%>%?',
               r'modify_title\s*\([^)]*\)',
               r'add_significance\s*\([^)]*\)\s*%>%?',
               r'bold_p\s*\([^)]*\)\s*%>%?']:
        raw = re.sub(fn, '', raw)

    lines = raw.splitlines()
    deduped, prev = [], None
    for line in lines:
        s = line.strip()
        if s and s == prev:
            continue
        deduped.append(line)
        if s:
            prev = s
    raw = '\n'.join(deduped)

    if 'TABLE_DONE' not in raw:
        raw = raw.rstrip() + '\ncat("TABLE_DONE")\n'

    return raw.strip()


# ─────────────────────────────────────────────
# R EXECUTOR
# ─────────────────────────────────────────────
def execute_table(r_code, df):
    with tempfile.TemporaryDirectory() as d:
        inp_path    = os.path.join(d, "input.csv")
        out_path    = os.path.join(d, "output_table.html")
        html_path   = os.path.join(d, "output_display.html")
        script_path = os.path.join(d, "script.R")

        df.to_csv(inp_path, index=False)

        full_script = "\n".join([
            "user_lib <- path.expand('~/R/library')",
            "if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))",
            "suppressPackageStartupMessages({",
            "  library(dplyr); library(gtsummary); library(gt); library(tidyr)",
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

        html_str = ""
        for path in [html_path, out_path]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    html_str = f.read()
                break

        if not html_str:
            raise RuntimeError("Output file was not created.\n" + res.stderr)

        return html_str, res.stderr


# ─────────────────────────────────────────────
# CODE DIFF DISPLAY
# ─────────────────────────────────────────────
def show_code_diff(old_code, new_code):
    import difflib
    diff = difflib.unified_diff(
        (old_code or "").splitlines(),
        (new_code or "").splitlines(),
        lineterm=''
    )
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

    # ── Session state init ───────────────────────────────────────────────
    if "tbl_initialized" not in st.session_state:
        st.session_state["tbl_r_code_pending"]  = None
        st.session_state["tbl_r_code_original"] = None
        st.session_state["tbl_preview_html"]    = None
        st.session_state["_tbl_run_now"]        = False
        st.session_state["tbl_initialized"]     = True

    # DEBUG — shows what code was used as base for last enhancement
    if st.session_state.get("_debug_enhancement_base"):
        st.sidebar.markdown("**Base used for enhancement:**")
        st.sidebar.code(st.session_state["_debug_enhancement_base"], language="r")

    for key, default in {
        "tbl_df":              None,
        "tbl_r_code":          "",
        "tbl_accepted_code":   "",
        "tbl_html":            None,
        "tbl_log":             "",
        "tbl_error":           None,
        "tbl_r_code_pending":  None,
        "tbl_r_code_original": None,
        "tbl_preview_html":    None,
        "_tbl_run_now":        False,
        "tbl_custom_text":     "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── R package check (once per session) ──────────────────────────────
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

    # ── Data upload ──────────────────────────────────────────────────────
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

    # ── Configure table ──────────────────────────────────────────────────
    st.subheader("⚙️ Configure Table")
    cols          = df.columns.tolist()
    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    all_with_none = ["None"] + cols

    r1a, r1b, r1c = st.columns(3)
    with r1a:
        table_type = st.selectbox("📋 Table Type", TABLE_TYPES)
    with r1b:
        group_col = st.selectbox("👥 Group / Treatment Col", all_with_none, index=0)
    with r1c:
        subj_col = st.selectbox("🔑 Subject ID Col", all_with_none, index=0)

    if "Table 1" in table_type:
        r2a, r2b, r2c = st.columns([2, 1, 1])
        with r2a:
            exclude = {group_col if group_col != "None" else "", subj_col if subj_col != "None" else ""}
            variables = st.multiselect(
                "📊 Variables to Summarise", cols,
                default=[c for c in cols if c not in exclude]
            )
        with r2b:
            stat_option = st.selectbox("📐 Continuous Stats", STAT_OPTIONS)
        with r2c:
            title = st.text_input("📝 Table Title", value="Table 1 — Baseline Characteristics")

        selections = {
            "table_type":  table_type,
            "variables":   variables,
            "group_col":   group_col if group_col != "None" else None,
            "subj_col":    subj_col  if subj_col  != "None" else None,
            "stat_option": stat_option,
            "title":       title,
        }

    elif "Adverse Events" in table_type:
        r2a, r2b, r2c = st.columns(3)
        with r2a:
            soc_col = st.selectbox("🏷️ SOC Column", cols)
        with r2b:
            pt_col  = st.selectbox("💊 PT Column", cols)
        with r2c:
            title   = st.text_input("📝 Table Title", value="Adverse Events Summary")

        selections = {
            "table_type": table_type,
            "soc_col":    soc_col,
            "pt_col":     pt_col,
            "group_col":  group_col if group_col != "None" else None,
            "subj_col":   subj_col  if subj_col  != "None" else "USUBJID",
            "title":      title,
        }

    st.divider()

    # ── Output (shown above custom box) ─────────────────────────────────
    if st.session_state.get("tbl_r_code"):
        st.subheader("📤 Output")
        out1, out2 = st.tabs(["📊 Table", "💻 R Code"])

        with out1:
            if st.session_state.get("tbl_html"):
                st.components.v1.html(
                    f"<div style='background:white; padding:10px;'>{st.session_state['tbl_html']}</div>",
                    height=600, scrolling=True
                )
                st.download_button(
                    "⬇️ Download HTML",
                    data=st.session_state["tbl_html"].encode("utf-8"),
                    file_name="clinical_table.html",
                    mime="text/html",
                )
            elif st.session_state.get("tbl_error"):
                st.error(st.session_state["tbl_error"])

        with out2:
            edited_code = st.text_area(
                "Edit R Code",
                value=st.session_state.get("tbl_r_code", ""),
                height=300,
                key=f"tbl_edited_{hash(st.session_state.get('tbl_r_code', ''))}"
            )
            b1, b2 = st.columns(2)
            with b1:
                run_edited = st.button("▶️ Run Edited Code", type="primary", use_container_width=True)
            with b2:
                st.download_button(
                    "⬇️ Download R Code", data=edited_code,
                    file_name="clinical_table.R", mime="text/plain",
                    use_container_width=True
                )
            if run_edited:
                with st.spinner("Running updated code..."):
                    try:
                        html_str, r_log = execute_table(edited_code, st.session_state["tbl_df"])
                        st.session_state["tbl_html"]   = html_str
                        st.session_state["tbl_log"]    = r_log
                        st.session_state["tbl_r_code"] = edited_code
                        st.session_state["tbl_error"]  = None
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))

            if st.session_state.get("tbl_log"):
                with st.expander("📋 R Log"):
                    st.code(st.session_state["tbl_log"], language="bash")

    st.divider()

    # ── Custom enhancement box ───────────────────────────────────────────
    custom_request = st.text_area(
        "✨ Custom Enhancement (optional)",
        placeholder="e.g. Add footnote 'Source: Clinical Study Report' | bold p-values | change font size",
        height=80,
        key="tbl_custom_text",
    )

    # ── Generate button ──────────────────────────────────────────────────
    if st.button("🏥 Generate Table", type="primary", use_container_width=True):

        if "Table 1" in table_type and not selections.get("variables"):
            st.error("⚠️ Please select at least one variable to summarise.")
            st.stop()

        with st.spinner("🤖 Generating R code..."):
            try:
                r_code = (
                    generate_table1_code(selections)
                    if "Table 1" in table_type
                    else generate_ae_code(selections)
                )

                existing = st.session_state.get("tbl_accepted_code", "")
                r_code_for_enhancement = existing if existing.strip() else r_code

                # Store what we're using for enhancement so we can inspect it
                st.session_state["_debug_enhancement_base"] = r_code_for_enhancement[:500]

                if custom_request.strip():
                    footnote_keywords = ["footnote", "foot note", "note", "annotation"]
                    is_footnote_request = any(k in custom_request.lower() for k in footnote_keywords)

                    if is_footnote_request:
                        footnote_text = extract_footnote_text_from_request(custom_request)
                        enhanced_code = apply_footnote_in_python(r_code_for_enhancement, footnote_text)
                        st.session_state["tbl_r_code_pending"]  = enhanced_code
                        st.session_state["tbl_r_code_original"] = r_code_for_enhancement
                        st.session_state["tbl_df"]              = df
                        st.session_state["tbl_preview_html"]    = None
                        st.rerun()

                    else:
                        prompt = build_enhance_prompt(r_code_for_enhancement, custom_request)
                        raw    = call_llm(prompt, groq_client, gemini_client)

                        if raw:
                            enhanced_code = clean_llm_output(raw)
                            st.session_state["tbl_r_code_pending"]  = enhanced_code
                            st.session_state["tbl_r_code_original"] = r_code_for_enhancement
                            st.session_state["tbl_df"]              = df
                            st.session_state["tbl_preview_html"]    = None
                            st.rerun()
                        else:
                            st.warning("⚠️ Enhancement failed, running base code instead.")
                            st.session_state["tbl_r_code_pending"] = None
                            st.session_state["tbl_r_code"]         = r_code
                            st.session_state["tbl_df"]             = df
                            st.session_state["_tbl_run_now"]       = True

                else:
                    # Fresh generate — reset accepted code and run base code
                    st.session_state["tbl_r_code_pending"] = None
                    st.session_state["tbl_r_code"]         = r_code
                    st.session_state["tbl_accepted_code"]  = ""
                    st.session_state["tbl_df"]             = df
                    st.session_state["_tbl_run_now"]       = True

            except Exception as e:
                import traceback
                st.error(f"Code generation error: {e}")
                st.code(traceback.format_exc())
                st.stop()

    # ── R execution block ────────────────────────────────────────────────
    # Outside Generate button block — fires on every rerun when flagged
    if st.session_state.get("_tbl_run_now") and not st.session_state.get("tbl_r_code_pending"):
        st.session_state["_tbl_run_now"] = False
        with st.spinner("⚙️ Running R..."):
            try:
                html_str, r_log = execute_table(
                    st.session_state["tbl_r_code"],
                    st.session_state["tbl_df"]
                )
                st.session_state["tbl_html"]  = html_str
                st.session_state["tbl_log"]   = r_log
                st.session_state["tbl_error"] = None
            except RuntimeError as e:
                st.session_state["tbl_error"] = str(e)
                st.session_state["tbl_html"]  = None
        st.rerun()

    # ── Review block ─────────────────────────────────────────────────────
    # Outside Generate button block — persists across reruns
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
                # Save to both tbl_r_code AND tbl_accepted_code
                # tbl_accepted_code is NEVER touched by Generate — only by Apply
                accepted = st.session_state["tbl_r_code_pending"]
                st.session_state["tbl_r_code"]          = accepted
                st.session_state["tbl_accepted_code"]   = accepted
                st.session_state["tbl_r_code_original"] = None
                st.session_state["tbl_r_code_pending"]  = None
                st.session_state["tbl_preview_html"]    = None
                st.session_state["_tbl_run_now"]        = True
                st.rerun()

        with c2:
            if st.button("👁️ Preview", use_container_width=True, key="tbl_preview"):
                with st.spinner("Generating preview..."):
                    try:
                        prev_html, _ = execute_table(
                            st.session_state["tbl_r_code_pending"],
                            st.session_state["tbl_df"]
                        )
                        st.session_state["tbl_preview_html"] = prev_html
                        st.rerun()
                    except RuntimeError as e:
                        st.error(f"Preview failed: {e}")

        with c3:
            if st.button("❌ Reject Changes", use_container_width=True, key="tbl_reject"):
                st.session_state["tbl_r_code_pending"] = None
                st.session_state["tbl_preview_html"]   = None
                st.rerun()

        # Side-by-side preview
        if st.session_state.get("tbl_preview_html"):
            st.markdown("**👁️ Preview (not applied yet):**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current (accepted):**")
                if st.session_state.get("tbl_html"):
                    st.components.v1.html(
                        f"<div style='background:white; padding:10px;'>{st.session_state['tbl_html']}</div>",
                        height=400, scrolling=True
                    )
            with col2:
                st.markdown("**Preview (pending):**")
                st.components.v1.html(
                    f"<div style='background:white; padding:10px;'>{st.session_state['tbl_preview_html']}</div>",
                    height=400, scrolling=True
                )
