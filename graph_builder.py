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
            "user_lib <- path.expand('~/R/library')",
            "if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))",
            # Prevent survminer from loading — it breaks with newer ggplot2
            "if ('survminer' %in% loadedNamespaces()) unloadNamespace('survminer')",
            "options(warn = -1)",
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE)',
            r_code_clean,
            f'suppressMessages(ggsave("{plot_path}", width=10, height=6, dpi=150))',
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
        "graph_chart_type": "Bar Chart",
        "graph_x_col": None,
        "graph_y_col": None,
        "graph_color_col": None,
        "graph_orientation": "vertical",
        "graph_title": "",
        "graph_theme": "minimal",
        "graph_palette": "default",
        "graph_sort_order": "none",
        "graph_show_values": False,
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
            df.columns = df.columns.str.strip()
            st.session_state["graph_df"] = df
            st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    with st.expander("📋 Or paste CSV text manually"):
        manual_csv = st.text_area("Paste CSV here", height=100, key="graph_manual_csv")
        if manual_csv:
            try:
                import io
                df = pd.read_csv(io.StringIO(manual_csv))
                df.columns = df.columns.str.strip()
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
    # Restore previous selections if columns match
    saved_x = st.session_state.get("graph_x_col")
    saved_y = st.session_state.get("graph_y_col") 
    saved_color = st.session_state.get("graph_color_col")
    
    x_default = cols.index(saved_x) if saved_x in cols else 0
    y_default = all_cols_with_none.index(saved_y) if saved_y in all_cols_with_none else (
        next((all_cols_with_none.index(c) for c in numeric_cols if c in all_cols_with_none), 0)
    )
    color_default = all_cols_with_none.index(saved_color) if saved_color in all_cols_with_none else 0
    st.write("DEBUG saved_color:", saved_color, "in list:", saved_color in all_cols_with_none)
    r1a, r1b, r1c, r1d, r1e = st.columns(5)
    with r1a:
        chart_type = st.selectbox("📊 Chart Type", CHART_TYPES, key="graph_chart_type")
    with r1b:
        x_col = st.selectbox("📋 X Axis", cols, index=x_default)
        st.session_state["graph_x_col"] = x_col
    with r1c:
        y_col = st.selectbox("📈 Y Axis", all_cols_with_none, index=y_default)
        st.session_state["graph_y_col"] = y_col
    with r1d:
        color_col = st.selectbox("🎨 Color By", all_cols_with_none, 
                                  index=color_default,
                                  key="graph_color_col_widget")
        st.session_state["graph_color_col"] = color_col
    with r1e:
        orientation = st.selectbox("📐 Orientation", ["vertical", "horizontal"], key="graph_orientation")

    r2a, r2b, r2c, r2d, r2e = st.columns(5)
    with r2a:
        default_title = st.session_state.get("graph_title") or f"{chart_type} of {x_col}"
        title = st.text_input("📝 Title", value=default_title)
        st.session_state["graph_title"] = title
    with r2b:
        theme = st.selectbox("🎨 Theme", THEMES, key="graph_theme")
    with r2c:
        palette = st.selectbox("🖌️ Palette", PALETTES, key="graph_palette")
    with r2d:
        sort_order = st.selectbox("📏 Sort Bars", ["none", "asc", "desc"], key="graph_sort_order")
    with r2e:
        st.write("")
        show_values = st.checkbox("🔢 Show Values", value=False, key="graph_show_values")

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

                # Build on top of previously accepted code so cumulative changes are preserved
                if st.session_state.get("graph_r_code"):
                    r_code_for_enhancement = st.session_state["graph_r_code"]
                else:
                    r_code_for_enhancement = r_code

                if custom_request.strip():
                    enhance_prompt = (
                        f"You are a ggplot2 code editor. Apply ONLY the requested change to the existing code.\n\n"
                        f"EXISTING CODE:\n```r\n{r_code_for_enhancement}\n```\n\n"
                        f"REQUEST: {custom_request}\n\n"
                        f"RULES:\n"
                        f"- Touch ONLY what the request asks. Preserve everything else exactly as in EXISTING CODE.\n"
                        f"- MERGE new settings into existing theme() block — never rewrite or replace the whole theme().\n"
                        f"- Keep all aes(), geom type, labs(), legend, colors and style from EXISTING CODE unless request explicitly changes them.\n"
                        f"- For structural changes (facet, panel, flip, new geom): still preserve all theme() and labs() settings from EXISTING CODE.\n"
                        f"- Never add: read.csv, hardcoded data, ggsave, cowplot, ggthemes.\n"
                        f"- Never remove fill= or color= from aes().\n"
                        f"- Before outputting, verify: is every theme(), legend, labs() setting from EXISTING CODE still present?\n"
                        f"- Return ONLY complete R code. No explanations, no markdown fences.\n"
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
                        st.session_state["graph_r_code_original"] = r_code_for_enhancement
                        st.session_state["graph_r_code"]          = r_code
                        st.session_state["graph_df"]              = df
                        st.session_state["graph_preview_png"]     = None
                        # Snapshot current graph for preview comparison
                        if st.session_state.get("graph_png") and not st.session_state.get("graph_png_accepted"):
                            st.session_state["graph_png_accepted"] = st.session_state["graph_png"]
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
                st.session_state["graph_r_code"]          = st.session_state["graph_r_code_pending"]
                st.session_state["graph_r_code_original"] = None
                st.session_state["graph_r_code_pending"]  = None
                st.session_state["graph_preview_png"]     = None
                st.session_state["_run_r_now"]            = True
                st.rerun()
        with c2:
            if st.button("👁️ Preview", use_container_width=True):
                with st.spinner("Generating preview..."):
                    try:
                        st.code(st.session_state["graph_r_code_pending"], language="r")
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


# ═══════════════════════════════════════════════════════════════
# CLINICAL GRAPHS — completely separate, does not touch anything above
# ═══════════════════════════════════════════════════════════════

CLINICAL_CHART_TYPES = [
    "Kaplan-Meier Survival Curve",
    "Forest Plot (Subgroup Analysis)",
    "Waterfall Plot (Tumor Response)",
    "Box Plot by Visit (Lab Values)",
    "Spaghetti Plot (Patient Trajectories)",
    "Mean ± SD Plot by Visit",
    "Dot Plot (Biomarker)",
    "Swimmer Plot (Patient Timeline)",
]

CLINICAL_R_PACKAGES = ["survival", "survminer", "ggplot2", "dplyr", "tidyr", "scales"]


def ensure_clinical_packages():
    install_script = """
pkgs <- c("survival", "survminer", "ggplot2", "dplyr", "tidyr", "scales")
user_lib <- path.expand("~/R/library")
if (!dir.exists(user_lib)) dir.create(user_lib, recursive=TRUE)
.libPaths(c(user_lib, .libPaths()))
missing <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(missing) > 0) {
  message("Installing: ", paste(missing, collapse=", "))
  install.packages(missing, repos="https://cloud.r-project.org", lib=user_lib, quiet=FALSE)
}
message("All clinical packages ready")
"""
    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(install_script)
        path = f.name
    res = subprocess.run(["Rscript", path], capture_output=True, text=True, timeout=600)
    os.unlink(path)
    return res.returncode == 0, res.stderr


def generate_clinical_code(chart_type, selections):
    """Generate R code for each clinical chart type — pure Python, no LLM."""

    time_col    = selections.get("time_col", "TIME")
    event_col   = selections.get("event_col", "EVENT")
    group_col   = selections.get("group_col")
    value_col   = selections.get("value_col", "VALUE")
    visit_col   = selections.get("visit_col", "VISIT")
    subj_col    = selections.get("subj_col", "USUBJID")
    response_col= selections.get("response_col", "RESPONSE")
    low_col     = selections.get("low_col")
    high_col    = selections.get("high_col")
    est_col     = selections.get("est_col", "ESTIMATE")
    label_col   = selections.get("label_col", "LABEL")
    start_col   = selections.get("start_col", "START")
    end_col     = selections.get("end_col", "END")
    event2_col  = selections.get("event2_col")
    title       = selections.get("title", chart_type)
    theme       = selections.get("theme", "minimal")

    base_libs = """
user_lib <- path.expand("~/R/library")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
})
"""

    if chart_type == "Kaplan-Meier Survival Curve":
        group_formula = group_col if group_col else "1"
        color_aes     = ", color = strata" if group_col else ""
        fill_aes      = ", fill = strata" if group_col else ""
        code = f"""
{base_libs}
suppressPackageStartupMessages(library(survival))

df${event_col} <- as.numeric(df${event_col})
df${time_col}  <- as.numeric(df${time_col})

fit <- survfit(Surv({time_col}, {event_col}) ~ {group_formula}, data = df)

fit_sum <- summary(fit)
fit_df  <- data.frame(
  time   = fit_sum$time,
  surv   = fit_sum$surv,
  upper  = fit_sum$upper,
  lower  = fit_sum$lower,
  strata = if (!is.null(fit_sum$strata)) as.character(fit_sum$strata) else "Overall"
)

p <- ggplot(fit_df, aes(x = time, y = surv{color_aes})) +
  geom_step(size = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper{fill_aes}),
              alpha = 0.15, linetype = 0) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title = "{title}",
       x = "Time",
       y = "Survival Probability",
       color = "{group_col if group_col else ''}",
       fill  = "{group_col if group_col else ''}") +
  theme_{theme}() +
  theme(plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""
    elif chart_type == "Forest Plot (Subgroup Analysis)":
        ci_low  = low_col  if low_col  else f"({est_col} - 0.2)"
        ci_high = high_col if high_col else f"({est_col} + 0.2)"
        code = f"""
{base_libs}

df${est_col}  <- as.numeric(df${est_col})
df${label_col} <- factor(df${label_col}, levels = rev(unique(df${label_col})))

p <- ggplot(df, aes(x = {est_col}, y = {label_col})) +
  geom_point(size = 3, color = "steelblue") +
  geom_errorbarh(aes(xmin = {ci_low}, xmax = {ci_high}), height = 0.2, color = "steelblue") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red", size = 0.8) +
  labs(title = "{title}", x = "Hazard Ratio (95% CI)", y = "") +
  theme_{theme}() +
  theme(plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""

    elif chart_type == "Waterfall Plot (Tumor Response)":
        color_aes = f"fill = {group_col}" if group_col else f"fill = {response_col} > 0"
        code = f"""
{base_libs}

df <- df %>% arrange({response_col})
df$patient_order <- factor(seq_len(nrow(df)), levels = seq_len(nrow(df)))

p <- ggplot(df, aes(x = patient_order, y = {response_col}, {color_aes})) +
  geom_bar(stat = "identity") +
  geom_hline(yintercept = -30, linetype = "dashed", color = "red",   size = 0.8) +
  geom_hline(yintercept =  20, linetype = "dashed", color = "orange",size = 0.8) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "{title}", x = "Patient", y = "Best % Change from Baseline") +
  theme_{theme}() +
  theme(axis.text.x     = element_blank(),
        axis.ticks.x    = element_blank(),
        plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""

    elif chart_type == "Box Plot by Visit (Lab Values)":
        color_aes = f", fill = {group_col}" if group_col else ""
        code = f"""
{base_libs}

df${visit_col} <- factor(df${visit_col}, levels = unique(df${visit_col}))

p <- ggplot(df, aes(x = {visit_col}, y = {value_col}{color_aes})) +
  geom_boxplot(outlier.shape = 21, outlier.size = 2, alpha = 0.7) +
  geom_jitter(width = 0.15, alpha = 0.3, size = 1.5) +
  labs(title = "{title}", x = "Visit", y = "{value_col}") +
  theme_{theme}() +
  theme(axis.text.x     = element_text(angle = 45, hjust = 1),
        plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""

    elif chart_type == "Spaghetti Plot (Patient Trajectories)":
        color_aes = f", color = {group_col}" if group_col else ""
        code = f"""
{base_libs}

df${visit_col} <- factor(df${visit_col}, levels = unique(df${visit_col}))
df$visit_num   <- as.numeric(df${visit_col})

p <- ggplot(df, aes(x = visit_num, y = {value_col}, group = {subj_col}{color_aes})) +
  geom_line(alpha = 0.4, size = 0.6) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq_along(levels(df${visit_col})),
                     labels = levels(df${visit_col})) +
  labs(title = "{title}", x = "Visit", y = "{value_col}") +
  theme_{theme}() +
  theme(axis.text.x     = element_text(angle = 45, hjust = 1),
        plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""

    elif chart_type == "Mean ± SD Plot by Visit":
        color_aes = f", color = {group_col}, group = {group_col}" if group_col else ", group = 1"
        code = f"""
{base_libs}

df${visit_col} <- factor(df${visit_col}, levels = unique(df${visit_col}))
summ <- df %>%
  group_by({visit_col}{(", " + group_col) if group_col else ""}) %>%
  summarise(mean_val = mean({value_col}, na.rm=TRUE),
            sd_val   = sd({value_col},   na.rm=TRUE),
            .groups  = "drop")

p <- ggplot(summ, aes(x = {visit_col}, y = mean_val{color_aes})) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_val - sd_val, ymax = mean_val + sd_val), width = 0.2) +
  labs(title = "{title}", x = "Visit", y = "Mean ± SD of {value_col}") +
  theme_{theme}() +
  theme(axis.text.x     = element_text(angle = 45, hjust = 1),
        plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""

    elif chart_type == "Dot Plot (Biomarker)":
        color_aes = f", color = {group_col}" if group_col else ""
        code = f"""
{base_libs}

p <- ggplot(df, aes(x = {group_col if group_col else "1"}, y = {value_col}{color_aes})) +
  geom_jitter(width = 0.2, size = 3, alpha = 0.7) +
  stat_summary(fun = median, geom = "crossbar", width = 0.4,
               color = "black", fatten = 2) +
  labs(title = "{title}",
       x = "{group_col if group_col else ''}",
       y = "{value_col}") +
  theme_{theme}() +
  theme(plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""

    elif chart_type == "Swimmer Plot (Patient Timeline)":
        event_line = (
            f"""
  geom_point(data = df %>% filter(!is.na({event2_col})),
             aes(x = {event2_col}, y = {subj_col}),
             shape = 23, size = 4, fill = "red", color = "black") +"""
            if event2_col else ""
        )
        code = f"""
{base_libs}

df${subj_col} <- factor(df${subj_col}, levels = df${subj_col}[order(df${end_col})])

p <- ggplot(df, aes(y = {subj_col})) +
  geom_segment(aes(x = {start_col}, xend = {end_col},
                   yend = {subj_col}{(", color = " + group_col) if group_col else ""}),
               size = 5, lineend = "round") +{event_line}
  labs(title = "{title}", x = "Time (Days)", y = "Patient") +
  theme_{theme}() +
  theme(plot.background  = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
p
"""
    else:
        code = f"""
{base_libs}
p <- ggplot(df, aes(x = 1, y = 1)) +
  geom_text(label = "Chart type not implemented") +
  theme_{theme}()
p
"""

    return code.strip()


def execute_clinical_graph(r_code, df):
    """Run R code, return PNG bytes. Same pattern as execute_graph."""
    with tempfile.TemporaryDirectory() as d:
        inp_path    = os.path.join(d, "input.csv")
        plot_path   = os.path.join(d, "output_plot.png")
        script_path = os.path.join(d, "script.R")

        df.to_csv(inp_path, index=False)

        # Strip any trailing + or ggsave
        r_code_clean = r_code.strip()
        while r_code_clean.endswith('+'):
            r_code_clean = r_code_clean[:-1].rstrip()

        full_script = "\n".join([
            "user_lib <- path.expand('~/R/library')",
            "if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))",
            # Prevent survminer from loading — it breaks with newer ggplot2
            "if ('survminer' %in% loadedNamespaces()) unloadNamespace('survminer')",
            "options(warn = -1)",
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE)',
            r_code_clean,
            f'suppressMessages(ggsave("{plot_path}", width=10, height=6, dpi=150))',
        ])

        with open(script_path, "w") as f:
            f.write(full_script)

        res = subprocess.run(
            ["Rscript", script_path],
            capture_output=True, text=True, timeout=60
        )

        if res.returncode != 0:
            raise RuntimeError(f"R Error:\n{res.stderr}")

        if os.path.exists(plot_path):
            with open(plot_path, "rb") as f:
                return f.read(), res.stderr
        else:
            raise RuntimeError("Plot file was not created.\n" + res.stderr)


def render_clinical_graphs_tab():
    st.subheader("🏥 Clinical Graphs")
    st.caption("Upload data → select clinical chart type → generate R code + graph")
    st.divider()
    # ── Session state — all keys prefixed cg_ to avoid any collision ────
    if "cg_initialized" not in st.session_state:
        st.session_state["cg_r_code_pending"]  = None
        st.session_state["cg_r_code_original"] = None
        st.session_state["cg_preview_png"]     = None
        st.session_state["_cg_run_now"]        = False
        st.session_state["cg_initialized"]     = True
        

    for key, default in {
        "cg_df":              None,
        "cg_r_code":          "",
        "cg_png":             None,
        "cg_png_accepted":    None,
        "cg_log":             "",
        "cg_error":           None,
        "cg_r_code_pending":  None,
        "cg_r_code_original": None,
        "cg_preview_png":     None,
        "cg_custom_text":     "",
        "_cg_run_now":        False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── R package check (once per session) ──────────────────────────────
    if "cg_pkgs_checked" not in st.session_state:
        st.info("🔧 Installing clinical R packages — this takes 2-5 minutes on first run...")
        ok, err = ensure_clinical_packages()
        st.session_state["cg_pkgs_checked"] = True
        if not ok:
            st.warning(f"Some packages may be missing:\n{err}")

    # ── Data upload ──────────────────────────────────────────────────────
    st.subheader("📁 Upload Data")
    with st.expander("📋 Or paste CSV text manually"):
        manual_csv = st.text_area("Paste CSV here", height=100, key="cg_manual_csv")
        if manual_csv:
            try:
                import io
                df = pd.read_csv(io.StringIO(manual_csv))
                st.session_state["cg_df"] = df
                st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
                st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Parse error: {e}")

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="cg_upload"
    )
    df = st.session_state.get("cg_df")
    if uploaded:
        try:
            ext = os.path.splitext(uploaded.name)[1].lower()
            df = pd.read_excel(uploaded) if ext in (".xlsx", ".xls") else pd.read_csv(uploaded)
            st.session_state["cg_df"] = df
            st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
            with st.expander("👁️ Preview Data", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    if df is None:
        st.info("👆 Upload a CSV or Excel file or paste CSV text to get started.")
        return

    st.divider()

    # ── Configure chart ──────────────────────────────────────────────────
    st.subheader("⚙️ Configure Clinical Chart")
    cols          = df.columns.tolist()
    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    all_with_none = ["None"] + cols

    # Row 1: chart type + theme + title
    r1a, r1b, r1c = st.columns([2, 1, 2])
    with r1a:
        chart_type = st.selectbox("📊 Chart Type", CLINICAL_CHART_TYPES, key="cg_chart_type")
    with r1b:
        theme = st.selectbox("🎨 Theme", THEMES, key="cg_theme")
    with r1c:
        title = st.text_input("📝 Title", value=chart_type, key="cg_title")

    # Row 2: column selectors — shown contextually based on chart type
    selections = {"title": title, "theme": theme}

    if chart_type == "Kaplan-Meier Survival Curve":
        c1, c2, c3 = st.columns(3)
        with c1:
            selections["time_col"]  = st.selectbox("⏱️ Time Column", cols, key="cg_time")
        with c2:
            selections["event_col"] = st.selectbox("💀 Event Column (0/1)", cols, key="cg_event")
        with c3:
            selections["group_col"] = st.selectbox("👥 Group Column", all_with_none, key="cg_group")
            if selections["group_col"] == "None":
                selections["group_col"] = None

    elif chart_type == "Forest Plot (Subgroup Analysis)":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selections["label_col"] = st.selectbox("🏷️ Label Column", cols, key="cg_label")
        with c2:
            selections["est_col"]   = st.selectbox("📍 Estimate Column", cols, key="cg_est")
        with c3:
            selections["low_col"]   = st.selectbox("⬇️ CI Lower", all_with_none, key="cg_low")
            if selections["low_col"] == "None": selections["low_col"] = None
        with c4:
            selections["high_col"]  = st.selectbox("⬆️ CI Upper", all_with_none, key="cg_high")
            if selections["high_col"] == "None": selections["high_col"] = None

    elif chart_type == "Waterfall Plot (Tumor Response)":
        c1, c2 = st.columns(2)
        with c1:
            selections["response_col"] = st.selectbox("📊 Response % Column", numeric_cols or cols, key="cg_response")
        with c2:
            selections["group_col"] = st.selectbox("👥 Group Column", all_with_none, key="cg_group")
            if selections["group_col"] == "None": selections["group_col"] = None

    elif chart_type in ["Box Plot by Visit (Lab Values)", "Spaghetti Plot (Patient Trajectories)", "Mean ± SD Plot by Visit"]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selections["visit_col"] = st.selectbox("📅 Visit Column", cols, key="cg_visit")
        with c2:
            selections["value_col"] = st.selectbox("🔢 Value Column", numeric_cols or cols, key="cg_value")
        with c3:
            selections["group_col"] = st.selectbox("👥 Group Column", all_with_none, key="cg_group")
            if selections["group_col"] == "None": selections["group_col"] = None
        with c4:
            selections["subj_col"]  = st.selectbox("🔑 Subject ID", all_with_none, key="cg_subj")
            if selections["subj_col"] == "None": selections["subj_col"] = "USUBJID"

    elif chart_type == "Dot Plot (Biomarker)":
        c1, c2 = st.columns(2)
        with c1:
            selections["value_col"] = st.selectbox("🔢 Value Column", numeric_cols or cols, key="cg_value")
        with c2:
            selections["group_col"] = st.selectbox("👥 Group Column", all_with_none, key="cg_group")
            if selections["group_col"] == "None": selections["group_col"] = None

    elif chart_type == "Swimmer Plot (Patient Timeline)":
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            selections["subj_col"]   = st.selectbox("🔑 Subject ID", cols, key="cg_subj")
        with c2:
            selections["start_col"]  = st.selectbox("▶️ Start", cols, key="cg_start")
        with c3:
            selections["end_col"]    = st.selectbox("⏹️ End", cols, key="cg_end")
        with c4:
            selections["group_col"]  = st.selectbox("👥 Group", all_with_none, key="cg_group")
            if selections["group_col"] == "None": selections["group_col"] = None
        with c5:
            selections["event2_col"] = st.selectbox("💊 Event Marker", all_with_none, key="cg_event2")
            if selections["event2_col"] == "None": selections["event2_col"] = None

    st.divider()

    # ── Output ───────────────────────────────────────────────────────────
    if st.session_state.get("cg_r_code"):
        st.subheader("📤 Output")
        out1, out2 = st.tabs(["📊 Graph", "💻 R Code"])

        with out1:
            img = st.session_state.get("cg_png_accepted") or st.session_state.get("cg_png")
            if img:
                st.image(img, use_container_width=True)
                st.download_button(
                    "⬇️ Download PNG", data=img,
                    file_name="clinical_graph.png", mime="image/png"
                )
            elif st.session_state.get("cg_error"):
                st.error(st.session_state["cg_error"])

        with out2:
            edited = st.text_area(
                "Edit R Code",
                value=st.session_state.get("cg_r_code", ""),
                height=300,
                key=f"cg_edited_{hash(st.session_state.get('cg_r_code', ''))}"
            )
            b1, b2 = st.columns(2)
            with b1:
                run_edit = st.button("▶️ Run Edited Code", type="primary",
                                     use_container_width=True, key="cg_run_edit")
            with b2:
                st.download_button(
                    "⬇️ Download R Code", data=edited,
                    file_name="clinical_graph.R", mime="text/plain",
                    use_container_width=True
                )
            if run_edit:
                with st.spinner("Running..."):
                    try:
                        png, log = execute_clinical_graph(edited, st.session_state["cg_df"])
                        st.session_state["cg_png"]          = png
                        st.session_state["cg_png_accepted"] = png
                        st.session_state["cg_log"]          = log
                        st.session_state["cg_r_code"]       = edited
                        st.session_state["cg_error"]        = None
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))

            if st.session_state.get("cg_log"):
                with st.expander("📋 R Log"):
                    st.code(st.session_state["cg_log"], language="bash")

    st.divider()

    # ── Custom enhancement ───────────────────────────────────────────────
    custom_request = st.text_area(
        "✨ Custom Enhancement (optional)",
        placeholder="e.g. Add confidence interval, change color palette, add annotation...",
        height=80,
        key="cg_custom_text",
    )

    # ── Generate button ──────────────────────────────────────────────────
    def clear_clinical_graph():
        for key in ["cg_df", "cg_r_code", "cg_png", "cg_png_accepted",
                    "cg_log", "cg_error", "cg_preview_png",
                    "cg_r_code_pending", "cg_r_code_original"]:
            st.session_state[key] = None
        st.session_state["cg_r_code"] = ""

    btn1, btn2 = st.columns([4, 1])
    with btn1:
        generate_cg = st.button("🏥 Generate Clinical Graph", type="primary", use_container_width=True)
    with btn2:
        st.button("🗑️ Clear", on_click=clear_clinical_graph, use_container_width=True, key="cg_clear")

    if generate_cg:
        with st.spinner("🤖 Generating R code..."):
            try:
                r_code = generate_clinical_code(chart_type, selections)

                # Build on accepted code for cumulative enhancements
                existing = st.session_state.get("cg_r_code", "")
                r_code_for_enhancement = existing if existing.strip() else r_code

                if custom_request.strip():
                    enhance_prompt = (
                        f"You are a ggplot2 clinical graph code editor. Apply ONLY the requested change.\n\n"
                        f"EXISTING CODE:\n```r\n{r_code_for_enhancement}\n```\n\n"
                        f"REQUEST: {custom_request}\n\n"
                        f"RULES:\n"
                        f"- Touch ONLY what the request asks. Preserve everything else exactly.\n"
                        f"- Never add read.csv, hardcoded data, or ggsave.\n"
                        f"- Keep all aes(), geom, labs(), theme() settings unless request changes them.\n"
                        f"- MERGE new theme settings — never rewrite the whole theme() block.\n"
                        f"- Return ONLY complete R code. No explanations, no markdown fences.\n"
                    )
                    raw = None
                    try:
                        res = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": enhance_prompt}],
                            temperature=0
                        )
                        raw = res.choices[0].message.content
                    except Exception:
                        try:
                            raw = gemini_client.models.generate_content(
                                model="gemini-2.0-flash", contents=enhance_prompt
                            ).text
                        except Exception:
                            st.warning("⚠️ Enhancement failed, using base code.")

                    if raw:
                        raw = re.sub(r'```[rR]?\n?', '', raw)
                        raw = re.sub(r'```', '', raw)
                        raw = re.sub(r'\+?\s*ggsave\s*\(.*?\)', '', raw, flags=re.DOTALL)
                        enhanced_code = raw.strip()
                        st.session_state["cg_r_code_pending"]  = enhanced_code
                        st.session_state["cg_r_code_original"] = r_code_for_enhancement
                        st.session_state["cg_r_code"]          = r_code_for_enhancement
                        st.session_state["cg_df"]              = df
                        st.session_state["cg_preview_png"]     = None
                        st.rerun()

                # No custom — run immediately
                st.session_state["cg_r_code_pending"] = None
                st.session_state["cg_r_code"]         = r_code
                st.session_state["cg_df"]             = df
                st.session_state["_cg_run_now"]       = True

            except Exception as e:
                import traceback
                st.error(f"Code generation error: {e}")
                st.code(traceback.format_exc())
                st.stop()

    # ── R execution block ─────────────────────────────────────────────────
    if st.session_state.get("_cg_run_now") and not st.session_state.get("cg_r_code_pending"):
        st.session_state["_cg_run_now"] = False
        with st.spinner("⚙️ Running R..."):
            try:
                png, log = execute_clinical_graph(
                    st.session_state["cg_r_code"],
                    st.session_state["cg_df"]
                )
                st.session_state["cg_png"]          = png
                st.session_state["cg_png_accepted"] = png
                st.session_state["cg_log"]          = log
                st.session_state["cg_error"]        = None
            except RuntimeError as e:
                st.session_state["cg_error"] = str(e)
                st.session_state["cg_png"]   = None
        st.rerun()

    # ── Review block ─────────────────────────────────────────────────────
    if st.session_state.get("cg_r_code_pending"):
        st.warning("⚠️ AI wants to modify your code. Review and confirm:")
        st.markdown("**Code Changes** (🟢 added | 🔴 removed):")
        show_code_diff(
            st.session_state["cg_r_code_original"],
            st.session_state["cg_r_code_pending"]
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ Apply Changes", use_container_width=True, key="cg_apply"):
                st.session_state["cg_r_code"]          = st.session_state["cg_r_code_pending"]
                st.session_state["cg_r_code_original"] = None
                st.session_state["cg_r_code_pending"]  = None
                st.session_state["cg_preview_png"]     = None
                st.session_state["_cg_run_now"]        = True
                st.rerun()
        with c2:
            if st.button("👁️ Preview", use_container_width=True, key="cg_preview_btn"):
                with st.spinner("Generating preview..."):
                    try:
                        prev_png, _ = execute_clinical_graph(
                            st.session_state["cg_r_code_pending"],
                            st.session_state["cg_df"]
                        )
                        st.session_state["cg_preview_png"] = prev_png
                        st.rerun()
                    except RuntimeError as e:
                        st.error(f"Preview failed: {e}")
        with c3:
            if st.button("❌ Reject Changes", use_container_width=True, key="cg_reject"):
                st.session_state["cg_r_code_pending"] = None
                st.session_state["cg_preview_png"]   = None
                st.rerun()

        if st.session_state.get("cg_preview_png"):
            st.markdown("**👁️ Preview (not applied yet):**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current Graph:**")
                current = st.session_state.get("cg_png_accepted") or st.session_state.get("cg_png")
                if current:
                    st.image(current, use_container_width=True)
            with col2:
                st.markdown("**Preview (pending):**")
                st.image(st.session_state["cg_preview_png"], use_container_width=True)
