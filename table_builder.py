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

OUTPUT_FORMATS = ["Word (.docx)", "PDF (.pdf)"]

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
REQUIRED_R_PACKAGES = ["gtsummary", "flextable", "officer", "dplyr", "webshot2", "gt"]

def ensure_r_packages():
    """Install missing R packages silently on first run."""
    install_script = """
pkgs <- c({pkgs})
user_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_lib)) dir.create(user_lib, recursive=TRUE)
.libPaths(c(user_lib, .libPaths()))
missing <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(missing) > 0) {{
  install.packages(missing, repos="https://cloud.r-project.org", lib=user_lib, quiet=TRUE)
}}
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

    if group_col:
        tbl_code = f"""
tbl <- df %>%
  select(all_of(c({vars_r}, "{group_col}"))) %>%
  tbl_summary(
    by = {group_col},
    statistic = list(all_continuous() ~ {stat_str},
                     all_categorical() ~ "{{n}} ({{p}}%)"),
    missing = "no"
  ) %>%
  add_overall() %>%
  add_p() %>%
  bold_labels() %>%
  modify_caption("**{title}**")
"""
    else:
        tbl_code = f"""
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

    if "Word" in output_fmt:
        export_code = f"""
ft <- as_flex_table(tbl)
save_as_docx(ft, path = output_path)
"""
    else:
        export_code = f"""
gt_tbl <- as_gt(tbl)
gtsave(gt_tbl, filename = output_path)
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

    if "Word" in output_fmt:
        export_code = f"""
ft <- flextable(ae_summary) %>%
  set_caption(caption = "{title}") %>%
  bold(part = "header") %>%
  autofit()
save_as_docx(ft, path = output_path)
"""
    else:
        export_code = f"""
gt_tbl <- gt(ae_summary) %>%
  tab_header(title = "{title}")
gtsave(gt_tbl, filename = output_path)
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
    return (
        f"You are a clinical R table code editor. Apply ONLY the requested change to the existing code.\n\n"
        f"EXISTING CODE:\n```r\n{current_code}\n```\n\n"
        f"REQUEST: {custom_request}\n\n"
        f"RULES:\n"
        f"- Touch ONLY what the request asks. Preserve everything else exactly.\n"
        f"- Never add read.csv, hardcoded data, or data loading code.\n"
        f"- Never remove or change the output_path variable — it is set externally.\n"
        f"- Keep all library() calls that are already present.\n"
        f"- Keep the cat('TABLE_DONE') line at the end.\n"
        f"- Return ONLY complete R code. No explanations, no markdown fences.\n"
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
    # Restore TABLE_DONE if LLM removed it
    if 'TABLE_DONE' not in raw:
        raw = raw.rstrip() + '\ncat("TABLE_DONE")\n'
    return raw.strip()


# ─────────────────────────────────────────────
# R EXECUTOR
# ─────────────────────────────────────────────
def execute_table(r_code, df, output_format):
    """Run R code, return (output_bytes, extension, stderr)."""
    ext = ".docx" if "Word" in output_format else ".pdf"

    with tempfile.TemporaryDirectory() as d:
        inp_path    = os.path.join(d, "input.csv")
        out_path    = os.path.join(d, f"output_table{ext}")
        script_path = os.path.join(d, "script.R")

        df.to_csv(inp_path, index=False)

        full_script = "\n".join([
            "user_lib <- Sys.getenv('R_LIBS_USER')",
            "if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))",
            "suppressPackageStartupMessages({",
            "  library(dplyr); library(gtsummary); library(flextable)",
            "  library(officer); library(gt); library(tidyr)",
            "})",
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE)',
            f'output_path <- "{out_path}"',
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

        with open(out_path, "rb") as f:
            return f.read(), ext, res.stderr


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
        "tbl_output_ext":      ".docx",
        "tbl_log":             "",
        "tbl_error":           None,
        "tbl_r_code_pending":  None,
        "tbl_r_code_original": None,
        "tbl_preview_bytes":   None,
        "_tbl_run_now":        False,
        "tbl_custom_text":     "",
        "tbl_output_format":   "Word (.docx)",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── R package check ─────────────────────────────────────────────────
    if "tbl_pkgs_checked" not in st.session_state:
        with st.spinner("🔧 Checking R packages..."):
            ok, err = ensure_r_packages()
            st.session_state["tbl_pkgs_checked"] = True
            if not ok:
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
            if st.session_state.get("tbl_output_bytes"):
                ext  = st.session_state.get("tbl_output_ext", ".docx")
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
                       if ext == ".docx" else "application/pdf"
                st.success("✅ Table generated successfully!")
                st.download_button(
                    f"⬇️ Download Table ({ext})",
                    data=st.session_state["tbl_output_bytes"],
                    file_name=f"clinical_table{ext}",
                    mime=mime,
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
                        out_bytes, ext, r_log = execute_table(
                            edited_code,
                            st.session_state["tbl_df"],
                            st.session_state["tbl_output_format"]
                        )
                        st.session_state["tbl_output_bytes"] = out_bytes
                        st.session_state["tbl_output_ext"]   = ext
                        st.session_state["tbl_log"]          = r_log
                        st.session_state["tbl_r_code"]       = edited_code
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
                # Generate base code from selections
                if "Table 1" in table_type:
                    r_code = generate_table1_code(selections)
                else:
                    r_code = generate_ae_code(selections)

                # Build on previously accepted code if it exists
                # (preserves cumulative custom changes)
                if st.session_state.get("tbl_r_code"):
                    r_code_for_enhancement = st.session_state["tbl_r_code"]
                else:
                    r_code_for_enhancement = r_code

                if custom_request.strip():
                    prompt = build_enhance_prompt(r_code_for_enhancement, custom_request)
                    raw    = call_llm(prompt, groq_client, gemini_client)

                    if raw:
                        enhanced_code = clean_llm_output(raw)
                        # Store pending — do NOT touch tbl_output_bytes
                        # so existing table stays visible during review
                        st.session_state["tbl_r_code_pending"]  = enhanced_code
                        st.session_state["tbl_r_code_original"] = r_code_for_enhancement
                        st.session_state["tbl_r_code"]          = r_code_for_enhancement
                        st.session_state["tbl_df"]              = df
                        st.session_state["tbl_preview_bytes"]   = None
                        st.rerun()
                    else:
                        st.warning("⚠️ Enhancement failed, using base code.")

                # No custom request — run immediately
                st.session_state["tbl_r_code_pending"] = None
                st.session_state["tbl_r_code"]         = r_code
                st.session_state["tbl_df"]             = df
                st.session_state["_tbl_run_now"]       = True

            except Exception as e:
                st.error(f"Code generation error: {e}")
                st.stop()

    # ── R execution block ─────────────────────────────────────────────────
    # Lives OUTSIDE generate button block so it runs on every rerun when flagged
    if st.session_state.get("_tbl_run_now") and not st.session_state.get("tbl_r_code_pending"):
        st.session_state["_tbl_run_now"] = False
        with st.spinner("⚙️ Running R..."):
            try:
                out_bytes, ext, r_log = execute_table(
                    st.session_state["tbl_r_code"],
                    st.session_state["tbl_df"],
                    st.session_state["tbl_output_format"]
                )
                st.session_state["tbl_output_bytes"] = out_bytes
                st.session_state["tbl_output_ext"]   = ext
                st.session_state["tbl_log"]          = r_log
                st.session_state["tbl_error"]        = None
            except RuntimeError as e:
                st.session_state["tbl_error"]        = str(e)
                st.session_state["tbl_output_bytes"] = None
        st.rerun()

    # ── Review block ──────────────────────────────────────────────────────
    # Lives OUTSIDE generate button block — persists across reruns
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
                st.session_state["tbl_r_code"]          = st.session_state["tbl_r_code_pending"]
                st.session_state["tbl_r_code_original"] = None   # reset diff baseline
                st.session_state["tbl_r_code_pending"]  = None
                st.session_state["tbl_preview_bytes"]   = None
                st.session_state["_tbl_run_now"]        = True
                st.rerun()

        with c2:
            if st.button("👁️ Preview", use_container_width=True, key="tbl_preview"):
                with st.spinner("Generating preview..."):
                    try:
                        prev_bytes, _, _ = execute_table(
                            st.session_state["tbl_r_code_pending"],
                            st.session_state["tbl_df"],
                            st.session_state["tbl_output_format"]
                        )
                        st.session_state["tbl_preview_bytes"] = prev_bytes
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
        if st.session_state.get("tbl_preview_bytes"):
            st.markdown("**👁️ Preview ready — download to compare:**")
            ext  = st.session_state.get("tbl_output_ext", ".docx")
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" \
                   if ext == ".docx" else "application/pdf"
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current (accepted):**")
                if st.session_state.get("tbl_output_bytes"):
                    st.download_button(
                        "⬇️ Download Current",
                        data=st.session_state["tbl_output_bytes"],
                        file_name=f"current_table{ext}",
                        mime=mime,
                        key="dl_current"
                    )
            with col2:
                st.markdown("**Preview (pending):**")
                st.download_button(
                    "⬇️ Download Preview",
                    data=st.session_state["tbl_preview_bytes"],
                    file_name=f"preview_table{ext}",
                    mime=mime,
                    key="dl_preview"
                )
