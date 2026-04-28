import os, re, subprocess, tempfile, io
import pandas as pd
import streamlit as st
from groq import Groq
from google import genai

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
def get_secret(key):
    try: return st.secrets[key]
    except Exception: return os.environ.get(key, "")

def _make_clients():
    gemini_client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    groq_client   = Groq(api_key=get_secret("GROQ_API_KEY"))
    return gemini_client, groq_client

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
OUTPUT_FORMATS = ["HTML", "Word (.docx)", "PDF"]

FLAG_COLORS = {
    "HIGH":   "#FFB3B3",
    "LOW":    "#B3D9FF",
    "NORMAL": "#B3FFB3",
    "ABNORMAL": "#FFD9B3",
}

# ─────────────────────────────────────────────
# R CODE GENERATOR
# ─────────────────────────────────────────────
def generate_listing_code(selections):
    """Generate R code for clinical listing using flextable."""

    cols          = selections["columns"]
    sort_cols     = selections.get("sort_cols", [])
    group_col     = selections.get("group_col")
    filter_col    = selections.get("filter_col")
    filter_val    = selections.get("filter_val", "")
    title         = selections.get("title", "Clinical Listing")
    footnote      = selections.get("footnote", "")
    flag_col      = selections.get("flag_col")
    output_fmt    = selections.get("output_format", "HTML")
    decimal_cols  = selections.get("decimal_cols", [])
    decimal_places = selections.get("decimal_places", 2)

    cols_r = "c(" + ", ".join(f'"{c}"' for c in cols) + ")"

    # Sort
    if sort_cols:
        sort_r = "c(" + ", ".join(f'"{c}"' for c in sort_cols) + ")"
        sort_code = f"df <- df %>% arrange(across(all_of({sort_r})))"
    else:
        sort_code = ""

    # Filter
    if filter_col and filter_val:
        filter_code = f'df <- df %>% filter({filter_col} == "{filter_val}")'
    else:
        filter_code = ""

    # Decimal formatting
    decimal_code = ""
    if decimal_cols:
        for col in decimal_cols:
            decimal_code += f'df${col} <- round(as.numeric(df${col}), {decimal_places})\n'

    # Group by (page break per subject)
    group_code = ""
    if group_col:
        group_code = f"""
# Add group separator
df <- df %>% arrange({group_col})
"""

    # Flag highlighting
    flag_code = ""
    if flag_col:
        flag_code = f"""
# Highlight flag column
ft <- ft %>%
  bg(i = ~ {flag_col} %in% c("HIGH", "ABNORMAL"), bg = "#FFB3B3", part = "body") %>%
  bg(i = ~ {flag_col} %in% c("LOW"), bg = "#B3D9FF", part = "body")
"""

    # Footnote
    footnote_code = ""
    if footnote:
        footnote_code = f'ft <- ft %>% add_footer_lines("{footnote}")'

    # Output format
    if output_fmt == "Word (.docx)":
        export_code = """
doc <- read_docx()
doc <- body_add_flextable(doc, ft)
print(doc, target = output_path)
writeLines(ft_html, html_path)
"""
    elif output_fmt == "PDF":
        export_code = """
# Save as HTML then note PDF needs further conversion
writeLines(ft_html, html_path)
writeLines(ft_html, output_path)
"""
    else:  # HTML
        export_code = """
writeLines(ft_html, html_path)
writeLines(ft_html, output_path)
"""

    code = f"""library(dplyr)
library(flextable)
library(officer)
library(htmltools)

# ── Data preparation ──
{filter_code}
{sort_code}
{decimal_code}
{group_code}

# ── Select columns ──
df <- df %>% select(all_of({cols_r}))

# ── Build flextable ──
ft <- flextable(df)
ft <- set_caption(ft, caption = "{title}")
ft <- theme_vanilla(ft)
ft <- autofit(ft)
ft <- bold(ft, part = "header")
ft <- align(ft, align = "left", part = "all")
ft <- fontsize(ft, size = 10, part = "all")
ft <- fontsize(ft, size = 11, part = "header")
ft <- bg(ft, bg = "#F5F5F5", part = "header")
ft <- border_outer(ft, part = "all")
ft <- border_inner_h(ft, part = "body")

{flag_code}
{footnote_code}

# ── Export ──
ft_html <- as_html(ft)
{export_code}
cat("LISTING_DONE")
"""
    return code

# ─────────────────────────────────────────────
# R EXECUTOR
# ─────────────────────────────────────────────
def execute_listing(r_code, df, output_format):
    """Run R listing code and return HTML + bytes."""
    with tempfile.TemporaryDirectory() as d:
        inp_path    = os.path.join(d, "input.csv")
        html_path   = os.path.join(d, "listing.html")
        out_path    = os.path.join(d, "listing.docx" if "Word" in output_format else "listing.html")
        script_path = os.path.join(d, "script.R")

        df.to_csv(inp_path, index=False)

        full_script = f"""
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(flextable))
suppressPackageStartupMessages(library(officer))

html_path   <- "{html_path}"
output_path <- "{out_path}"
df <- read.csv("{inp_path}", stringsAsFactors=FALSE, check.names=FALSE)

{r_code}
"""
        with open(script_path, "w") as f:
            f.write(full_script)

        res = subprocess.run(
            ["Rscript", script_path],
            capture_output=True, text=True, timeout=60
        )

        if res.returncode != 0:
            raise RuntimeError(f"R Error:\n{res.stderr}")

        html_str = ""
        if os.path.exists(html_path):
            with open(html_path, "r") as f:
                html_str = f.read()

        out_bytes = None
        ext = ".docx" if "Word" in output_format else ".html"
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                out_bytes = f.read()

        return html_str, out_bytes, ext, res.stderr

# ─────────────────────────────────────────────
# LLM ENHANCEMENT
# ─────────────────────────────────────────────
def call_llm(prompt, groq_client, gemini_client):
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = res.choices[0].message.content
    except Exception:
        try:
            raw = gemini_client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            ).text
        except Exception:
            return None
    # Clean code blocks
    raw = re.sub(r'```[rR]?\n?', '', raw)
    raw = re.sub(r'```', '', raw)
    return raw.strip()

def build_listing_enhance_prompt(current_code, custom_request, available_cols=None):
    col_info = f"\nAVAILABLE COLUMNS: {', '.join(available_cols)}\n" if available_cols else ""
    return (
        f"You are a clinical R listing code editor. Apply ONLY the requested change.\n\n"
        f"EXISTING CODE:\n```r\n{current_code}\n```\n"
        f"{col_info}"
        f"REQUEST: {custom_request}\n\n"
        f"RULES:\n"
        f"1. Touch ONLY what the request asks.\n"
        f"2. Never add data loading code.\n"
        f"3. Never remove html_path, output_path, cat('LISTING_DONE').\n"
        f"4. Only use flextable functions: bg, color, bold, italic, fontsize, align, border, add_footer_lines.\n"
        f"5. Never invent column names — only use AVAILABLE COLUMNS.\n"
        f"6. Return ONLY complete R code. No explanations.\n"
    )

# ─────────────────────────────────────────────
# DIFF VIEWER
# ─────────────────────────────────────────────
def show_code_diff(old_code, new_code):
    import difflib
    old_lines = (old_code or "").splitlines()
    new_lines = (new_code or "").splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, lineterm='')
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
# CLEAR
# ─────────────────────────────────────────────
def clear_listing():
    for key in ["lst_df", "lst_r_code", "lst_html", "lst_output_bytes",
                "lst_error", "lst_log", "lst_r_code_pending",
                "lst_r_code_original", "lst_preview_html",
                "lst_accepted_code", "_lst_run_now"]:
        st.session_state[key] = None
    st.session_state["lst_r_code"] = ""
    st.session_state["_lst_run_now"] = False

# ─────────────────────────────────────────────
# MAIN RENDERER
# ─────────────────────────────────────────────
def render_listing_builder_tab():
    st.title("📋 Clinical Listings")
    st.caption("Upload data → configure listing → get formatted R code + downloadable listing")
    st.divider()

    # ── Session state init ──
    for key, default in {
        "lst_df": None,
        "lst_r_code": "",
        "lst_html": None,
        "lst_output_bytes": None,
        "lst_output_ext": ".html",
        "lst_error": None,
        "lst_log": "",
        "lst_r_code_pending": None,
        "lst_r_code_original": None,
        "lst_preview_html": None,
        "lst_accepted_code": "",
        "_lst_run_now": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    gemini_client, groq_client = _make_clients()

    # ── Data upload ──
    st.subheader("📁 Upload Data")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="lst_upload"
    )

    df = st.session_state.get("lst_df")

    if uploaded:
        try:
            ext = os.path.splitext(uploaded.name)[1].lower()
            df = pd.read_excel(uploaded) if ext in (".xlsx", ".xls") else pd.read_csv(uploaded)
            st.session_state["lst_df"] = df
            st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
            with st.expander("👁️ Preview Data", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    with st.expander("Or paste CSV text manually"):
        manual_csv = st.text_area("Paste CSV here", height=100, key="lst_manual_csv")
        if manual_csv:
            try:
                df = pd.read_csv(io.StringIO(manual_csv))
                st.session_state["lst_df"] = df
                st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Parse error: {e}")

    if df is None:
        st.info("👆 Upload a CSV or Excel file or paste CSV text to get started.")
        return

    st.divider()

    # ── Configure listing ──
    st.subheader("⚙️ Configure Listing")
    cols = df.columns.tolist()
    all_with_none = ["None"] + cols

    # Row 1
    r1a, r1b, r1c = st.columns(3)
    with r1a:
        output_format = st.selectbox("📄 Output Format", OUTPUT_FORMATS, key="lst_output_fmt")
        st.session_state["lst_output_format"] = output_format
    with r1b:
        group_col = st.selectbox("👥 Group / Page Break By", all_with_none, index=0, key="lst_group_col")
    with r1c:
        flag_col = st.selectbox("🚩 Flag Column", all_with_none, index=0, key="lst_flag_col")

    # Row 2
    r2a, r2b = st.columns(2)
    with r2a:
        selected_cols = st.multiselect(
            "📊 Columns to Display",
            cols,
            default=cols[:6] if len(cols) >= 6 else cols,
            key="lst_cols"
        )
    with r2b:
        sort_cols = st.multiselect(
            "📏 Sort By",
            cols,
            default=[cols[0]] if cols else [],
            key="lst_sort_cols"
        )

    # Row 3
    r3a, r3b, r3c = st.columns(3)
    with r3a:
        filter_col = st.selectbox("🔍 Filter Column", all_with_none, index=0, key="lst_filter_col")
    with r3b:
        if filter_col and filter_col != "None":
            unique_vals = [""] + sorted(df[filter_col].dropna().unique().astype(str).tolist())
            filter_val = st.selectbox("🔍 Filter Value", unique_vals, key="lst_filter_val")
        else:
            filter_val = ""
            st.selectbox("🔍 Filter Value", ["—"], disabled=True, key="lst_filter_val_disabled")
    with r3c:
        title = st.text_input("📝 Listing Title", value="Clinical Data Listing", key="lst_title")

    # Row 4
    r4a, r4b, r4c = st.columns(3)
    with r4a:
        decimal_cols = st.multiselect("🔢 Round Numeric Cols", cols, key="lst_decimal_cols")
    with r4b:
        decimal_places = st.number_input("Decimal Places", min_value=0, max_value=6, value=2, key="lst_decimal_places")
    with r4c:
        footnote = st.text_input("📝 Footnote", value="", placeholder="Optional footnote text", key="lst_footnote")

    if not selected_cols:
        st.warning("Please select at least one column to display.")
        return

    selections = {
        "columns":        selected_cols,
        "sort_cols":      sort_cols,
        "group_col":      group_col if group_col != "None" else None,
        "flag_col":       flag_col if flag_col != "None" else None,
        "filter_col":     filter_col if filter_col != "None" else None,
        "filter_val":     filter_val,
        "title":          title,
        "footnote":       footnote,
        "decimal_cols":   decimal_cols,
        "decimal_places": decimal_places,
        "output_format":  output_format,
    }

    st.divider()

    # ── Output ──
    if st.session_state.get("lst_r_code"):
        st.subheader("📤 Output")
        out1, out2 = st.tabs(["📋 Listing", "💻 R Code"])

        with out1:
            if st.session_state.get("lst_html"):
                import streamlit.components.v1 as components
                components.html(
                    f"<div style='background:white; padding:10px;'>{st.session_state['lst_html']}</div>",
                    height=600,
                    scrolling=True
                )
                if st.session_state.get("lst_output_bytes"):
                    ext  = st.session_state.get("lst_output_ext", ".html")
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext == ".docx" else "text/html"
                    st.download_button(
                        f"⬇️ Download Listing",
                        data=st.session_state["lst_output_bytes"],
                        file_name=f"listing{ext}",
                        mime=mime,
                        use_container_width=True
                    )
            elif st.session_state.get("lst_error"):
                st.error(st.session_state["lst_error"])

        with out2:
            edited_code = st.text_area(
                "Edit R Code",
                value=st.session_state.get("lst_r_code", ""),
                height=300,
                key=f"lst_edited_{hash(st.session_state.get('lst_r_code', ''))}"
            )
            btn1, btn2 = st.columns(2)
            with btn1:
                run_edited = st.button("▶️ Run Edited Code", type="primary", use_container_width=True)
            with btn2:
                st.download_button(
                    "⬇️ Download R Code",
                    data=edited_code,
                    file_name="listing.R",
                    mime="text/plain",
                    use_container_width=True
                )
            if run_edited:
                with st.spinner("Running..."):
                    try:
                        html_str, out_bytes, ext, r_log = execute_listing(
                            edited_code,
                            st.session_state["lst_df"],
                            st.session_state.get("lst_output_format", "HTML")
                        )
                        st.session_state["lst_html"]         = html_str
                        st.session_state["lst_output_bytes"] = out_bytes
                        st.session_state["lst_output_ext"]   = ext
                        st.session_state["lst_log"]          = r_log
                        st.session_state["lst_r_code"]       = edited_code
                        st.session_state["lst_accepted_code"] = edited_code
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))

            log = st.session_state.get("lst_log", "")
            if log:
                with st.expander("📋 R Log"):
                    st.code(log, language="bash")

    st.divider()

    # ── Custom enhancement ──
    custom_request = st.text_area(
        "✨ Custom Enhancement (optional)",
        placeholder="e.g. Highlight rows where FLAG='HIGH' in red, bold subject column, add border between groups...",
        height=80,
        key="lst_custom_text"
    )

    # ── Generate button ──
    btn_gen, btn_clr = st.columns([4, 1])
    with btn_gen:
        generate = st.button("📋 Generate Listing", type="primary", use_container_width=True)
    with btn_clr:
        st.button("🗑️ Clear", on_click=clear_listing, use_container_width=True)

    if generate:
        with st.spinner("🤖 Generating R code..."):
            try:
                r_code = generate_listing_code(selections)
                r_code_for_enhancement = st.session_state.get("lst_accepted_code") or r_code

                if custom_request.strip():
                    prompt = build_listing_enhance_prompt(
                        r_code_for_enhancement, custom_request, df.columns.tolist()
                    )
                    raw = call_llm(prompt, groq_client, gemini_client)

                    if raw:
                        # Validate columns
                        available_cols = df.columns.tolist()
                        all_quoted = re.findall(r'"([^"]+)"', raw)
                        KNOWN_NON_COLS = {'LISTING_DONE', 'TRUE', 'FALSE', 'NULL', 'NA',
                                         'left', 'right', 'center', 'all', 'body', 'header',
                                         'vanilla', 'booktabs'}
                        real_invalid = [
                            c for c in all_quoted
                            if c not in available_cols
                            and c not in KNOWN_NON_COLS
                            and '/' not in c
                            and '.' not in c
                            and len(c) < 30
                            and (c.isupper() or (c[0].isupper() and ' ' not in c and '{' not in c))
                        ]
                        if real_invalid:
                            st.warning(f"⚠️ AI tried to use columns that don't exist: {real_invalid}. Request ignored.")
                            st.stop()

                        st.session_state["lst_r_code_pending"]  = raw
                        st.session_state["lst_r_code_original"] = r_code_for_enhancement
                        st.session_state["lst_df"]              = df
                        st.session_state["lst_preview_html"]    = None
                        st.rerun()
                    else:
                        st.warning("⚠️ Enhancement failed, running base code.")
                        st.session_state["lst_r_code_pending"] = None
                        st.session_state["lst_r_code"]         = r_code
                        st.session_state["lst_df"]             = df
                        st.session_state["_lst_run_now"]       = True
                else:
                    st.session_state["lst_r_code_pending"] = None
                    st.session_state["lst_r_code"]         = r_code
                    st.session_state["lst_accepted_code"]  = ""
                    st.session_state["lst_df"]             = df
                    st.session_state["_lst_run_now"]       = True

            except Exception as e:
                import traceback
                st.error(f"Code generation error: {e}")
                st.code(traceback.format_exc())
                st.stop()

    # ── R execution block ──
    if st.session_state.get("_lst_run_now") and not st.session_state.get("lst_r_code_pending"):
        st.session_state["_lst_run_now"] = False
        with st.spinner("⚙️ Running R..."):
            try:
                html_str, out_bytes, ext, r_log = execute_listing(
                    st.session_state["lst_r_code"],
                    st.session_state["lst_df"],
                    st.session_state.get("lst_output_format", "HTML")
                )
                st.session_state["lst_html"]         = html_str
                st.session_state["lst_output_bytes"] = out_bytes
                st.session_state["lst_output_ext"]   = ext
                st.session_state["lst_log"]          = r_log
                st.session_state["lst_error"]        = None
                st.session_state["lst_accepted_code"] = st.session_state["lst_r_code"]
            except RuntimeError as e:
                st.session_state["lst_error"] = str(e)
                st.session_state["lst_html"]  = None
        st.rerun()

    # ── Review block ──
    if st.session_state.get("lst_r_code_pending"):
        st.warning("⚠️ AI wants to modify your code. Review and confirm:")
        st.markdown("**Code Changes** (🟢 added | 🔴 removed):")
        show_code_diff(
            st.session_state["lst_r_code_original"],
            st.session_state["lst_r_code_pending"]
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ Apply Changes", use_container_width=True, key="lst_apply"):
                accepted = st.session_state["lst_r_code_pending"]
                st.session_state["lst_r_code"]         = accepted
                st.session_state["lst_accepted_code"]  = accepted
                st.session_state["lst_r_code_pending"] = None
                st.session_state["lst_preview_html"]   = None
                st.session_state["_lst_run_now"]       = True
                st.rerun()
        with c2:
            if st.button("👁️ Preview", use_container_width=True, key="lst_preview"):
                with st.spinner("Generating preview..."):
                    try:
                        prev_html, _, _, _ = execute_listing(
                            st.session_state["lst_r_code_pending"],
                            st.session_state["lst_df"],
                            st.session_state.get("lst_output_format", "HTML")
                        )
                        st.session_state["lst_preview_html"] = prev_html
                        st.rerun()
                    except RuntimeError as e:
                        st.error(f"Preview failed: {e}")
        with c3:
            if st.button("❌ Reject Changes", use_container_width=True, key="lst_reject"):
                st.session_state["lst_r_code_pending"] = None
                st.session_state["lst_preview_html"]   = None
                st.rerun()

        if st.session_state.get("lst_preview_html"):
            st.markdown("**👁️ Preview (not applied yet):**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current:**")
                if st.session_state.get("lst_html"):
                    import streamlit.components.v1 as components
                    components.html(
                        f"<div style='background:white; padding:10px;'>{st.session_state['lst_html']}</div>",
                        height=400, scrolling=True
                    )
            with col2:
                st.markdown("**Preview (pending):**")
                import streamlit.components.v1 as components
                components.html(
                    f"<div style='background:white; padding:10px;'>{st.session_state['lst_preview_html']}</div>",
                    height=400, scrolling=True
                )
