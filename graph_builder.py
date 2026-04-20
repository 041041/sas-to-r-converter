import os, re, subprocess, tempfile
import pandas as pd
import streamlit as st
from groq import Groq
from google import genai

# --- CLIENTS (reuse from app.py) ---
def get_secret(key):
    try: return st.secrets[key]
    except Exception: return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

# --- CHART TYPES ---
CHART_TYPES = [
    "Bar Chart",
    "Line Chart", 
    "Scatter Plot",
    "Histogram",
    "Box Plot",
    "Pie Chart",
    "Area Chart",
    "Heatmap"
]

THEMES = ["minimal", "classic", "dark", "light", "bw", "void"]
PALETTES = ["default", "Blues", "Reds", "Greens", "Spectral", "Set1", "Set2"]

# --- GENERATE GGPLOT CODE VIA LLM ---
def generate_graph_code(selections, df_preview, col_types):
    """Sends graph selections to LLM and gets ggplot2 code back."""
    
    prompt = (
        f"Generate R ggplot2 code for the following chart.\n"
        f"DATA PREVIEW:\n{df_preview}\n"
        f"COLUMN TYPES:\n{col_types}\n"
        f"CHART SETTINGS:\n"
        f"- Chart type: {selections['chart_type']}\n"
        f"- X axis: {selections['x_col']}\n"
        f"- Y axis: {selections.get('y_col', 'None')}\n"
        f"- Color by: {selections.get('color_col', 'None')}\n"
        f"- Title: {selections.get('title', '')}\n"
        f"- Theme: theme_{selections.get('theme', 'minimal')}()\n"
        f"- Orientation: {selections.get('orientation', 'vertical')}\n"
        f"- Show values: {selections.get('show_values', False)}\n"
        f"- Sort: {selections.get('sort_order', 'none')}\n"
        f"- Color palette: {selections.get('palette', 'default')}\n\n"
        f"STRICT RULES:\n"
        f"1. Use ggplot2 only. Load with library(ggplot2).\n"
        f"2. Data frame is named 'df'.\n"
        f"3. Save plot as: ggsave('output_plot.png', width=10, height=6, dpi=150)\n"
        f"4. CRITICAL: ALWAYS use geom_bar(stat='identity') — NEVER geom_bar() alone.\n"
        f"5. ALWAYS include aes(x={selections['x_col']}, y={selections.get('y_col', '')}) in ggplot().\n"
        f"6. CRITICAL: If color_by is not None, ALWAYS put fill={selections.get('color_col','NULL')} inside aes().\n"
        f"7. ALWAYS use theme_{selections.get('theme','minimal')}() — NEVER theme_dark().\n"
        f"8. ALWAYS add labs(title='{selections.get('title','')}', x='{selections['x_col']}', y='{selections.get('y_col','')}').\n"
        f"9. Use position='dodge' for grouped bar charts.\n"
        f"10. If show values is TRUE add geom_text() with labels.\n"
        f"11. If sort is asc/desc, use reorder() on x axis.\n"
        f"12. Always add proper labs(title=, x=, y=, fill=) labels.\n"
        f"13. No explanations. Just R code.\n"
    )
    
    try:
        res = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        raw = res.choices[0].message.content
    except Exception:
        raw = gemini_client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt
        ).text
    
    # clean code blocks
    backticks = "\x60\x60\x60"
    if backticks in raw:
        pattern = backticks + r"(?:r|R)?\n(.*?)\n" + backticks
        import re
        blocks = re.findall(pattern, raw, re.DOTALL)
        if blocks: raw = "\n".join(blocks)
    
    return raw.strip()

# --- EXECUTE R AND RETURN PNG ---
def execute_graph(r_code, df):
    """Runs ggplot2 code via Rscript and returns PNG path."""
    with tempfile.TemporaryDirectory() as d:
        inp_path    = os.path.join(d, "input.csv")
        plot_path   = os.path.join(d, "output_plot.png")
        script_path = os.path.join(d, "script.R")

        df.to_csv(inp_path, index=False)

        import re
        # Step 1 - remove ggsave from LLM code
        r_code_clean = re.sub(r'\+?\s*ggsave\s*\(.*?\)', '', r_code, flags=re.DOTALL).strip()
        # Step 2 - force stat='identity' in geom_bar
        r_code_clean = re.sub(r'geom_bar\(\)', "geom_bar(stat='identity')", r_code_clean)

        full_script = "\n".join([
            "suppressPackageStartupMessages(library(ggplot2))",
            "suppressPackageStartupMessages(library(dplyr))",
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE)',
            r_code_clean,
            f'ggsave("{plot_path}", width=10, height=6, dpi=150)'
        ])
        with open(script_path, "w") as f:
            f.write(full_script)

        res = subprocess.run(
            ["Rscript", script_path],
            capture_output=True, text=True, timeout=30
        )

        if res.returncode != 0:
            raise RuntimeError(f"R Error:\n{res.stderr}")

        if os.path.exists(plot_path):
            with open(plot_path, "rb") as f:
                return f.read(), res.stderr
        else:
            raise RuntimeError("Plot file was not created.")

# --- MAIN TAB RENDERER ---
def render_graph_builder_tab():
    st.subheader("📊 R Graph Builder")
    st.caption("Upload data → configure chart → get ggplot2 code + graph")
    st.divider()

    # --- DATA UPLOAD ---
    st.subheader("📁 Upload Data")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="graph_upload"
    )

    df = None
    if uploaded:
        try:
            ext = os.path.splitext(uploaded.name)[1].lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return

    if df is None:
        st.info("👆 Upload a CSV or Excel file to get started.")
        return

    st.divider()

    # --- CHART CONFIGURATION ---
    st.subheader("⚙️ Configure Chart")

    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols_with_none = ["None"] + cols

    c1, c2 = st.columns(2)
    with c1:
        chart_type  = st.selectbox("📊 Chart Type", CHART_TYPES)
        x_col       = st.selectbox("📋 X Axis", cols)
        title       = st.text_input("📝 Title", value=f"{chart_type} of {x_col}")
        theme       = st.selectbox("🎨 Theme", THEMES)
        sort_order  = st.selectbox("📏 Sort Bars", ["none", "asc", "desc"])

    with c2:
        y_col       = st.selectbox("📈 Y Axis", all_cols_with_none, index=0)
        color_col   = st.selectbox("🎨 Color By", all_cols_with_none, index=0)
        orientation = st.selectbox("📐 Orientation", ["vertical", "horizontal"])
        palette     = st.selectbox("🖌️ Color Palette", PALETTES)
        show_values = st.checkbox("🔢 Show Values on Chart", value=False)

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

    # column types for LLM context
    col_types = {c: str(df[c].dtype) for c in cols}

    # df preview for LLM
    df_preview = df.head(3).to_string()

    st.divider()

    # --- GENERATE BUTTON ---
    if st.button("🎨 Generate Graph", type="primary", use_container_width=True):
        with st.spinner("🤖 Generating ggplot2 code..."):
            try:
                r_code = generate_graph_code(selections, df_preview, col_types)
            except Exception as e:
                st.error(f"LLM error: {e}")
                return

        st.session_state["graph_r_code"] = r_code
        st.session_state["graph_df"]     = df

        with st.spinner("⚙️ Running R..."):
            try:
                png_bytes, r_log = execute_graph(r_code, df)
                st.session_state["graph_png"] = png_bytes
                st.session_state["graph_log"] = r_log
                st.session_state["graph_error"] = None
            except RuntimeError as e:
                st.session_state["graph_error"] = str(e)
                st.session_state["graph_png"]   = None

    # --- OUTPUT ---
    if st.session_state.get("graph_r_code"):
        st.divider()
        st.subheader("📤 Output")

        out1, out2 = st.tabs(["📊 Graph", "💻 R Code"])

        with out1:
            if st.session_state.get("graph_png"):
                st.image(st.session_state["graph_png"], use_container_width=True)
                st.download_button(
                    "⬇️ Download PNG",
                    data=st.session_state["graph_png"],
                    file_name="graph.png",
                    mime="image/png"
                )
            elif st.session_state.get("graph_error"):
                st.error(st.session_state["graph_error"])

        with out2:
            st.code(st.session_state.get("graph_r_code", ""), language="r")
            st.download_button(
                "⬇️ Download R Code",
                data=st.session_state.get("graph_r_code", ""),
                file_name="graph.R",
                mime="text/plain"
            )
            log = st.session_state.get("graph_log", "")
            if log:
                with st.expander("📋 R Log"):
                    st.code(log, language="bash")
