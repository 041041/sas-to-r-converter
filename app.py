# ============================================================
# SAS to R Converter — Streamlit App
# Converted from Google Colab by Claude
# ============================================================

import os
import re
import subprocess
import tempfile

import pandas as pd
import streamlit as st

# ── API clients ──────────────────────────────────────────────
from google import genai
from groq import Groq

# Load keys — works both on Streamlit Cloud (st.secrets) and locally (env vars)
def get_secret(key):
    try:
        return st.secrets[key]          # Streamlit Cloud / secrets.toml
    except Exception:
        return os.environ.get(key, "")  # Local machine

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

# ── R execution mode: rpy2 (preferred) or subprocess fallback ─
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    USE_RPY2 = True
except ImportError:
    USE_RPY2 = False


# ============================================================
# YOUR ORIGINAL HELPER FUNCTIONS (unchanged)
# ============================================================

def clean_r_code(text):
    if "```" in text:
        code_blocks = re.findall(r"```(?:r|python|R)?\n(.*?)\n```", text, re.DOTALL)
        if code_blocks:
            text = "\n".join(code_blocks)

    def fix_all_args(text):
        result = ""
        stack = 0
        i = 0
        while i < len(text):
            char = text[i]
            if char == '(':
                stack += 1
                result += char
            elif char == ')':
                stack -= 1
                result += char
            elif stack > 0 and text[i:i+2] == '<-':
                result += '='
                i += 1
            else:
                result += char
            i += 1
        return result

    text = fix_all_args(text)

    lines = text.split("\n")
    code_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(('#', '```', 'library', 'print', 'display')):
            continue
        if any(x in line.lower() for x in ["explanation:", "sas code:", "run;", "data.frame()"]):
            continue
        if "(" not in line and "=" not in line:
            line = re.sub(r'\s+=\s+', ' <- ', line)
        code_lines.append(line)

    cleaned = "\n".join(code_lines)
    if "df" not in cleaned.split()[-1:]:
        cleaned += "\ndf"
    return cleaned


def enforce_df_usage(code, df_cols):
    if not code.strip():
        return "df"
    lines = code.split("\n")
    out = []
    for line in lines:
        if line.strip() == "df":
            out.append(line)
            continue
        temp = line
        for col in df_cols:
            temp = re.sub(rf"(?<!df\$)\b{col}\b", f"df${col}", temp)
        if "<-" in temp:
            parts = temp.split("<-", 1)
            lhs = parts[0]
            rhs = parts[1]
            if "df$" not in lhs and lhs.strip() not in ["df"]:
                temp = f"df${lhs.strip()} <- {rhs.strip()}"
        out.append(temp)
    return "\n".join(out)


def call_llm_api(step, df_cols):
    is_proc      = 'proc ' in step.lower()
    is_transpose = 'transpose' in step.lower()

    prompt = (
        f"TASK: Convert SAS to Base R. Input data is 'df'. "
        f"Final line MUST be 'df'. NO EXPLANATIONS. SAS:\n{step}"
    )

    if is_transpose:
        prompt += """\nCRITICAL: If SAS contains PREFIX= or SUFFIX=, the R code MUST:
1. Store the unique levels of the ID/Time variable in a vector BEFORE calling reshape().
2. Perform reshape().
3. Use those stored levels to rename columns using paste0(PREFIX, levels, SUFFIX).
4. Arguments inside reshape() MUST use '='.
"""

    try:
        raw = gemini_client.models.generate_content(
            model='gemini-2.0-flash', contents=prompt
        ).text
    except Exception:
        res = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        raw = res.choices[0].message.content

    cleaned = clean_r_code(raw)
    return cleaned if is_proc else enforce_df_usage(cleaned, df_cols)


def run_r_code_subprocess(r_code: str, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run R code via subprocess (works even without rpy2).
    Writes input CSV → runs R script → reads output CSV.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv  = os.path.join(tmpdir, "input.csv")
        output_csv = os.path.join(tmpdir, "output.csv")
        r_script   = os.path.join(tmpdir, "script.R")

        input_df.to_csv(input_csv, index=False)

        full_r = f"""
df <- read.csv("{input_csv}", stringsAsFactors=FALSE)
{r_code}
write.csv(df, "{output_csv}", row.names=FALSE)
"""
        with open(r_script, "w") as f:
            f.write(full_r)

        result = subprocess.run(
            ["Rscript", r_script],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        return pd.read_csv(output_csv)


def run_pipeline(sas_code, sas_outputs, log_fn=None):
    """
    Main pipeline. log_fn is an optional callable for streaming logs to UI.
    """
    steps = re.findall(
        r"((?:data|proc)\s+.*?;.*?run;)",
        sas_code, re.DOTALL | re.IGNORECASE
    )
    env = {}

    for step in steps:
        name_match = re.search(r"(?:^data\s+|out\s*=\s*)(\w+)", step, re.I)
        if not name_match:
            continue
        name = name_match.group(1)
        if name not in sas_outputs:
            continue
        if 'datalines' in step.lower():
            env[name] = sas_outputs[name]
            continue

        inp = list(env.values())[-1] if env else None
        if inp is None:
            continue

        r_code = call_llm_api(step, inp.columns.tolist())

        if log_fn:
            log_fn(name, r_code)

        try:
            if USE_RPY2:
                ro.globalenv['df'] = pandas2ri.py2rpy(inp)
                out = pandas2ri.rpy2py(ro.r(f"local({{ {r_code} }})"))
            else:
                out = run_r_code_subprocess(r_code, inp)

            env[name] = out

        except Exception as e:
            if log_fn:
                log_fn(name, r_code, error=str(e))

    return env


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="SAS → R Converter",
    page_icon="🔄",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────
st.title("🔄 SAS to R Converter")
st.caption(
    "Powered by **Gemini 2.0 Flash** with **Groq / Llama 3.3 70B** as fallback  "
    f"| R engine: {'rpy2 (fast)' if USE_RPY2 else 'Rscript subprocess'}"
)
st.divider()

# ── API key check ─────────────────────────────────────────────
if not GEMINI_API_KEY or not GROQ_API_KEY:
    st.error(
        "⚠️ API keys not found. Set them in your terminal before running:\n\n"
        "```bash\n"
        "export GEMINI_API_KEY='your_key'\n"
        "export GROQ_API_KEY='your_key'\n"
        "```"
    )
    st.stop()

# ── Two-column layout ─────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📋 Input: SAS Code")
    sas_input = st.text_area(
        label="Paste your SAS code here",
        height=400,
        placeholder="""data LABS;
    input patient_id $ glucose age;
    datalines;
101 110 45
102 140 62
;
run;

data LAB_RESULTS;
    set LABS;
    adj_glucose = glucose + (age * 0.1);
    if adj_glucose > 130 then status = 'HIGH';
    else status = 'NORMAL';
run;
""",
        label_visibility="collapsed"
    )

    convert_btn = st.button("⚡ Convert to R", type="primary", use_container_width=True)

with col_right:
    st.subheader("📤 Output: Generated R Code")
    output_placeholder = st.empty()

# ── On button click ───────────────────────────────────────────
if convert_btn:
    if not sas_input.strip():
        st.warning("Please paste some SAS code first.")
    else:
        steps = re.findall(
            r"((?:data|proc)\s+.*?;.*?run;)",
            sas_input, re.DOTALL | re.IGNORECASE
        )

        if not steps:
            st.error("No valid SAS DATA or PROC steps found. Check your code format.")
        else:
            st.divider()
            st.subheader("🔄 Conversion Results")

            all_r_code_parts = []

            # ── Step-by-step conversion display ──────────────
            for i, step in enumerate(steps, 1):
                name_match = re.search(r"(?:^data\s+|out\s*=\s*)(\w+)", step, re.I)
                step_name  = name_match.group(1) if name_match else f"Step {i}"

                with st.expander(f"Step {i}: `{step_name}`", expanded=True):
                    tab_sas, tab_r = st.tabs(["SAS Input", "Generated R"])

                    with tab_sas:
                        st.code(step.strip(), language="sas")

                    with tab_r:
                        with st.spinner(f"Converting {step_name} via LLM..."):
                            try:
                                # Use dummy cols for proc steps
                                r_code = call_llm_api(step, [])
                                st.code(r_code, language="r")
                                all_r_code_parts.append(f"# --- {step_name} ---\n{r_code}")
                                st.success(f"✅ {step_name} converted")
                            except Exception as e:
                                st.error(f"❌ LLM error: {e}")

            # ── Combined R script download ────────────────────
            if all_r_code_parts:
                st.divider()
                full_r_script = "\n\n".join(all_r_code_parts)

                st.subheader("📥 Full R Script")
                st.code(full_r_script, language="r")

                st.download_button(
                    label="⬇️ Download R Script (.R)",
                    data=full_r_script,
                    file_name="converted_script.R",
                    mime="text/plain",
                    use_container_width=True
                )

# ── Sidebar: how to use ───────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown("""
1. Paste your SAS code in the left panel
2. Click **Convert to R**
3. View generated R code step-by-step
4. Download the full R script

---
**Supported SAS steps:**
- `DATA` step with `SET`, `IF/ELSE`, derived columns
- `PROC SORT`
- `PROC TRANSPOSE` (with PREFIX/SUFFIX)
- `PROC MEANS`
- `PROC FREQ`

---
**R execution:**
- rpy2 is used if installed
- Falls back to `Rscript` (subprocess) automatically
""")

    st.divider()
    st.caption("Built with Gemini 2.0 Flash + Groq")
