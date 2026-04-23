import os, re, subprocess, tempfile
import pandas as pd
import streamlit as st
from groq import Groq
from google import genai

def clear_graph():
    for key in ["graph_df", "graph_r_code", "graph_png", "graph_png_accepted",
                "graph_log", "graph_error", "graph_preview_png",
                "graph_r_code_pending", "graph_r_code_original"]:
        st.session_state[key] = None
    st.session_state["graph_r_code"] = ""

def show_code_diff(old_code, new_code):
    import difflib
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()
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

# --- CLIENTS ---
def get_secret(key):
    try: return st.secrets[key]
    except Exception: return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

CHART_TYPES = ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart", "Area Chart", "Heatmap"]
THEMES = ["minimal", "classic", "dark", "light", "bw", "void"]
PALETTES = ["default", "Blues", "Reds", "Greens", "Spectral", "Set1", "Set2"]

def generate_graph_code(selections, df_preview, col_types):
    chart_type  = selections['chart_type']
    x_col       = selections['x_col']
    y_col       = selections.get('y_col')
    color_col   = selections.get('color_col')
    title       = selections.get('title', '')
    theme       = selections.get('theme', 'minimal')
    orientation = selections.get('orientation', 'vertical')
    show_values = selections.get('show_values', False)
    sort_order  = selections.get('sort_order', 'none')
    palette     = selections.get('palette', 'default')

    if sort_order == 'asc' and y_col:
        x_aes = f"reorder({x_col}, {y_col})"
    elif sort_order == 'desc' and y_col:
        x_aes = f"reorder({x_col}, -{y_col})"
    else:
        x_aes = x_col

    if color_col:
        aes_str = f"aes(x={x_aes}, y={y_col}, fill={color_col})" if y_col else f"aes(x={x_aes}, fill={color_col})"
    else:
        aes_str = f"aes(x={x_aes}, y={y_col})" if y_col else f"aes(x={x_aes})"

    if chart_type == "Bar Chart":
        geom = "geom_bar(stat='identity', position='dodge')" if color_col else "geom_bar(stat='identity', fill='steelblue')"
    elif chart_type == "Line Chart":
        geom = f"geom_line(aes(group={color_col}), size=1)" if color_col else "geom_line(size=1, color='steelblue')"
        geom += f"\n  + geom_point(size=2)"
    elif chart_type == "Scatter Plot":
        geom = "geom_point(size=3)" if color_col else "geom_point(size=3, color='steelblue')"
    elif chart_type == "Histogram":
        geom = "geom_histogram(bins=30, fill='steelblue', color='white')"
    elif chart_type == "Box Plot":
        geom = "geom_boxplot()" if color_col else "geom_boxplot(fill='steelblue')"
    elif chart_type == "Area Chart":
        geom = f"geom_area(aes(group={color_col}), alpha=0.6)" if color_col else "geom_area(fill='steelblue', alpha=0.6)"
    elif chart_type == "Pie Chart":
        geom = "geom_bar(stat='identity', width=1)\n  + coord_polar('y')"
    else:
        geom = "geom_bar(stat='identity', fill='steelblue')"

    palette_line = f" +\n  scale_fill_brewer(palette='{palette}')" if (color_col and palette != "default") else ""

    values_line = ""
    if show_values and y_col:
        if color_col:
            values_line = f" +\n  geom_text(aes(label={y_col}), position=position_stack(vjust=0.5), size=3, color='white')"
        else:
            values_line = f" +\n  geom_text(aes(label={y_col}), vjust=-0.5, size=3)"

    flip_line = "\n  + coord_flip()" if orientation == "horizontal" else ""

    code = f"""library(ggplot2)

p <- ggplot(df, {aes_str}) +
  {geom}{palette_line}{values_line} +
  labs(title='{title}',
       x='{x_col}',
       y='{y_col if y_col else ''}') +
  theme_{theme}() +
  theme(plot.background = element_rect(fill='white'),
        panel.background = element_rect(fill='white')){flip_line}
p
"""
    return code

def execute_graph(r_code, df):
    with tempfile.TemporaryDirectory() as d:
        inp_path    = os.path.join(d, "input.csv")
        plot_path   = os.path.join(d, "output_plot.png")
        script_path = os.path.join(d, "script.R")

        df.to_csv(inp_path, index=False)

        if 'ggsave' in r_code:
            r_code = re.sub(r'\s*\+\s*\n?\s*(?=ggsave)', '', r_code)
            r_code_clean = r_code.split('ggsave')[0].strip().rstrip('+').strip()
        else:
            r_code_clean = r_code.strip()

        r_code_clean = r_code_clean.rstrip()
        while r_code_clean.endswith('+'):
            r_code_clean = r_code_clean[:-1].rstrip()

        full_script = "\n".join([
            "suppressPackageStartupMessages(library(ggplot2))",
            "suppressPackageStartupMessages(library(dplyr))",
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE)',
            r_code_clean,
            f'ggsave("{plot_path}", width=10, height=6, dpi=150)'
        ])
        with open(script_path, "w") as f:
            f.write(full_script)

        res = subprocess.run(["Rscript", script_path], capture_output=True, text=True, timeout=30)

        if res.returncode != 0:
            raise RuntimeError(f"R Error:\n{res.stderr}")

        if os.path.exists(plot_path):
            with open(plot_path, "rb") as f:
                return f.read(), res.stderr
        else:
            raise RuntimeError("Plot file was not created.")

def render_graph_builder_tab():
    st.title("📊 R Graph Builder")
    st.caption("Upload data → Configure chart → Generate ggplot2 code → Edit & Download")
    st.divider()

    # --- SESSION STATE INIT ---
    for key, default in {
        "graph_df": None,
        "graph_r_code": "",
        "graph_png": None,
        "graph_png_accepted": None,
        "graph_png_before_preview": None,
        "graph_log": "",
        "graph_error": None,
        "graph_preview_png": None,
        "graph_r_code_pending": None,
        "graph_r_code_original": None,
        "custom_request_text": "",
        "_run_r_now": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- DATA UPLOAD ---
    st.subheader("📁 Upload Data")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="graph_upload"
    )

    df = st.session_state.get("graph_df", None)

    if uploaded:
        try:
            ext = os.path.splitext(uploaded.name)[1].lower()
            df = pd.read_excel(uploaded) if ext in (".xlsx", ".xls") else pd.read_csv(uploaded)
            st.session_state["graph_df"] = df
            st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    with st.expander("Or paste CSV text manually"):
        manual_csv = st.text_area("Paste CSV here", height=100, key="graph_manual_csv")
        if manual_csv:
            try:
                import io
                df = pd.read_csv(io.StringIO(manual_csv))
                st.session_state["graph_df"] = df
                st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Parse error: {e}")

    if df is None:
        st.info("👆 Upload a CSV or Excel file or paste CSV text to get started.")
        return

    st.divider()

    # --- CONFIGURE CHART ---
    st.subheader("⚙️ Configure Chart")
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols_with_none = ["None"] + cols

    r1a, r1b, r1c, r1d, r1e = st.columns(5)
    with r1a:
        chart_type = st.selectbox("📊 Chart Type", CHART_TYPES)
    with r1b:
        x_col = st.selectbox("📋 X Axis", cols)
    with r1c:
        numeric_default = next(
            (all_cols_with_none.index(c) for c in numeric_cols if c in all_cols_with_none), 0
        )
        y_col = st.selectbox("📈 Y Axis", all_cols_with_none, index=numeric_default)
    with r1d:
        color_col = st.selectbox("🎨 Color By", all_cols_with_none, index=0)
    with r1e:
        orientation = st.selectbox("📐 Orientation", ["vertical", "horizontal"])

    r2a, r2b, r2c, r2d, r2e = st.columns(5)
    with r2a:
        title = st.text_input("📝 Title", value=f"{chart_type} of {x_col}")
    with r2b:
        theme = st.selectbox("🎨 Theme", THEMES)
    with r2c:
        palette = st.selectbox("🖌️ Palette", PALETTES)
    with r2d:
        sort_order = st.selectbox("📏 Sort Bars", ["none", "asc", "desc"])
    with r2e:
        st.write("")
        show_values = st.checkbox("🔢 Show Values", value=False)

    selections = {
        "chart_type":   chart_type,
        "x_col":        x_col,
        "y_col":        y_col if y_col != "None" else None,
        "color_col":    color_col if color_col != "None" else None,
        "title":        title,
        "theme":        theme,
        "orientation":  orientation,
        "palette":      palette,
        "show_values":  show_values,
        "sort_order":   sort_order,
    }

    col_types  = {c: str(df[c].dtype) for c in cols}
    df_preview = df.head(3).to_string()

    st.divider()

    # --- OUTPUT ---
    if st.session_state.get("graph_r_code"):
        st.subheader("📤 Output")
        out1, out2 = st.tabs(["📊 Graph", "💻 R Code"])
        with out1:
            img_to_show = st.session_state.get("graph_png_accepted") or st.session_state.get("graph_png")
            if img_to_show:
                st.image(img_to_show, use_container_width=True)
                st.download_button(
                    "⬇️ Download PNG",
                    data=img_to_show,
                    file_name="graph.png",
                    mime="image/png"
                )
            elif st.session_state.get("graph_error"):
                st.error(st.session_state["graph_error"])

        with out2:
            edited_code = st.text_area(
                "Edit R Code",
                value=st.session_state.get("graph_r_code", ""),
                height=300,
                key=f"edited_r_code_{hash(st.session_state.get('graph_r_code', ''))}"
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                run_edit = st.button("▶️ Run Edited Code", type="primary", use_container_width=True)
            with btn_col2:
                st.download_button(
                    "⬇️ Download R Code",
                    data=edited_code,
                    file_name="graph.R",
                    mime="text/plain",
                    use_container_width=True
                )
            if run_edit:
                with st.spinner("Running updated code..."):
                    try:
                        png_bytes, r_log = execute_graph(edited_code, st.session_state.get("graph_df"))
                        st.session_state["graph_png"] = png_bytes
                        st.session_state["graph_png_accepted"] = png_bytes
                        st.session_state["graph_log"] = r_log
                        st.session_state["graph_r_code"] = edited_code
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))
            log = st.session_state.get("graph_log", "")
            if log:
                with st.expander("📋 R Log"):
                    st.code(log, language="bash")

    # --- CUSTOM ENHANCEMENT ---
    custom_request = st.text_area(
        "✨ Custom Enhancement (optional)",
        placeholder="e.g. Add trend line, move legend to bottom, use dark theme...",
        height=80,
        key="custom_request_text",
    )

    # --- GENERATE BUTTON ---
    btn1, btn2 = st.columns([4, 1])
    with btn1:
        generate = st.button("🎨 Generate Graph", type="primary", use_container_width=True)
    with btn2:
        st.button("🗑️ Clear", on_click=clear_graph, use_container_width=True)

    if generate:
        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Area Chart"] and not selections.get("y_col"):
            st.error("⚠️ Please select a Y axis column for this chart type!")
            st.stop()

        with st.spinner("🤖 Generating ggplot2 code..."):
            try:
                r_code = generate_graph_code(selections, df_preview, col_types)

                if custom_request.strip():
                    enhance_prompt = (
                        f"Modify this ggplot2 R code based on this request: '{custom_request}'\n\n"
                        f"CURRENT CODE:\n{r_code}\n\n"
                        f"CRITICAL RULES:\n"
                        f"1. NEVER create or modify the data frame df\n"
                        f"2. NEVER add read.csv() or any data loading code\n"
                        f"3. NEVER invent or hardcode any data values\n"
                        f"4. ONLY modify ggplot2 visual elements (themes, labels, colors, geoms)\n"
                        f"5. NEVER change aes() mappings — keep fill, x, y exactly as is\n"
                        f"6. NEVER remove fill= or color= from aes()\n"
                        f"7. NEVER change geom type or position\n"
                        f"8. ONLY modify theme(), labs(), legend.position\n"
                        f"9. Return complete modified R code\n"
                        f"10. No explanations, just code\n"
                        f"11. Do NOT add ggsave\n"
                        f"12. Only use base ggplot2 — NO cowplot, NO ggthemes\n"
                    )
                    raw = None
                    try:
                        res = groq_client.chat.completions.create(
                            model='llama-3.3-70b-versatile',
                            messages=[{'role': 'user', 'content': enhance_prompt}],
                            temperature=0
                        )
                        raw = res.choices[0].message.content
                    except Exception:
                        try:
                            raw = gemini_client.models.generate_content(
                                model='gemini-2.0-flash', contents=enhance_prompt
                            ).text
                        except Exception:
                            st.warning("⚠️ Enhancement failed, using base code.")

                    if raw:
                        raw = re.sub(r'```[rR]?\n?', '', raw)
                        raw = re.sub(r'```', '', raw)
                        raw = re.sub(r'\+?\s*ggsave\s*\(.*?\)', '', raw, flags=re.DOTALL)
                        raw = re.sub(r'panel_border\([^)]*\)\s*\+?', '', raw)
                        raw = re.sub(r'library\(cowplot\)', '', raw)
                        raw = re.sub(r'library\(ggthemes\)', '', raw)
                        enhanced_code = raw.strip()
                        st.session_state["graph_r_code_pending"]  = enhanced_code
                        st.session_state["graph_r_code_original"] = r_code
                        st.session_state["graph_r_code"]          = r_code
                        st.session_state["graph_df"]              = df
                        st.session_state["graph_preview_png"]     = None
                        st.rerun()

                st.session_state["graph_r_code_pending"] = None
                st.session_state["graph_r_code"]         = r_code
                st.session_state["graph_df"]             = df
                st.session_state["_run_r_now"]           = True

            except Exception as e:
                st.error(f"Code generation error: {e}")
                st.stop()

    # --- RUN R ---
    if st.session_state.get("_run_r_now") and not st.session_state.get("graph_r_code_pending"):
        st.session_state["_run_r_now"] = False
        with st.spinner("⚙️ Running R..."):
            try:
                png_bytes, r_log = execute_graph(
                    st.session_state["graph_r_code"],
                    st.session_state["graph_df"]
                )
                st.session_state["graph_png"] = png_bytes
                st.session_state["graph_png_accepted"] = png_bytes
                st.session_state["graph_log"] = r_log
                st.session_state["graph_error"] = None
            except RuntimeError as e:
                st.session_state["graph_error"] = str(e)
                st.session_state["graph_png"] = None
        st.rerun()

    # --- REVIEW PENDING CHANGES ---
    if st.session_state.get("graph_r_code_pending"):
        st.warning("⚠️ AI wants to modify your code. Review and confirm:")
        st.markdown("**Code Changes** (🟢 added | 🔴 removed):")
        show_code_diff(
            st.session_state["graph_r_code_original"],
            st.session_state["graph_r_code_pending"]
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ Apply Changes", use_container_width=True):
                st.session_state["graph_r_code"]         = st.session_state["graph_r_code_pending"]
                st.session_state["graph_r_code_pending"] = None
                st.session_state["graph_preview_png"]    = None
                st.session_state["_run_r_now"]           = True
                st.rerun()
        with c2:
            if st.button("👁️ Preview", use_container_width=True):
                with st.spinner("Generating preview..."):
                    try:
                        preview_png, _ = execute_graph(
                            st.session_state["graph_r_code_pending"],
                            st.session_state["graph_df"]
                        )
                        st.session_state["graph_preview_png"] = preview_png
                        # Always use accepted graph as the "before" reference
                        before = st.session_state.get("graph_png_accepted")
                        if before is None:
                            before = st.session_state.get("graph_png")
                        st.session_state["graph_png_before_preview"] = before
                        st.rerun()
                    except RuntimeError as e:
                        st.error(f"Preview failed: {e}")
        with c3:
            if st.button("❌ Reject Changes", use_container_width=True):
                st.session_state["graph_r_code_pending"] = None
                st.session_state["graph_preview_png"]    = None
                st.rerun()

        if st.session_state.get("graph_preview_png"):
            st.markdown("**👁️ Preview (not applied yet):**")
            col_old, col_new = st.columns(2)
            with col_old:
                st.markdown("**Current Graph:**")
                before = st.session_state.get("graph_png_accepted") or st.session_state.get("graph_png_before_preview") or st.session_state.get("graph_png")
                if before:
                    st.image(before, use_container_width=True)
            with col_new:
                st.markdown("**Preview (pending):**")
                if st.session_state.get("graph_preview_png"):
                    st.image(st.session_state["graph_preview_png"], use_container_width=True)
