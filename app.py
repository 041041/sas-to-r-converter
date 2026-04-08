import os, re, subprocess, tempfile, io
import pandas as pd
import streamlit as st
from google import genai
from groq import Groq

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart SAS to R Converter", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    .step-card { border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- API CLIENT SETUP ---
def get_secret(key):
    try: return st.secrets[key]
    except Exception: return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")

if not GEMINI_API_KEY or not GROQ_API_KEY:
    st.error("API keys missing! Please add GEMINI_API_KEY and GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

# --- CLEANING & UTILS ---

def safe_read_csv(file_obj):
    """Attempts multiple methods to read a CSV gracefully."""
    try:
        # Attempt 1: Standard UTF-8 comma-separated
        file_obj.seek(0)
        return pd.read_csv(file_obj)
    except Exception:
        # Attempt 2: Fallback to latin1, auto-detect separator, and skip bad lines
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding='latin1', sep=None, engine='python', on_bad_lines='skip')
        except Exception as e:
            raise RuntimeError(f"Could not parse CSV file. Error: {str(e)}")

def clean_r_code(text):
    """Strips LLM conversational filler and ensures the code returns 'df'."""
    # Using hex escape \x60 to prevent markdown parser truncation
    backticks = "\x60\x60\x60"
    if backticks in text:
        pattern = backticks + r"(?:r|python|R)?\n(.*?)\n" + backticks
        blocks = re.findall(pattern, text, re.DOTALL)
        if blocks: text = "\n".join(blocks)
    
    lines = text.split("\n")
    out = []
    # Keywords often included by LLMs that break raw execution
    forbidden = ["explanation:", "sas code:", "run;", "data.frame()", "library("]
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line or clean_line.startswith(('#', backticks)): continue
        if any(x in clean_line.lower() for x in forbidden): continue
        # Replace arrow with equals if it's inside a function call (common LLM error)
        if "(" in clean_line and "<-" in clean_line:
            clean_line = clean_line.replace("<-", "=")
        out.append(clean_line)
    
    cleaned = "\n".join(out)
    if not cleaned.strip().endswith("df"): cleaned += "\ndf"
    return cleaned

def call_llm_api(step, df_cols):
    """Calls Gemini with a Groq fallback."""
    prompt = (
        f"TASK: Convert this SAS step to Base R code.\n"
        f"INPUT: A dataframe named 'df' with columns: {df_cols}\n"
        f"OUTPUT: Your code must manipulate 'df'. The last line MUST be exactly 'df'.\n"
        f"STRICT: No explanations, no markdown, no comments. Just executable R code.\n\n"
        f"SAS STEP:\n{step}"
    )
    try:
        raw = gemini_client.models.generate_content(model='gemini-2.0-flash', contents=prompt).text
    except Exception:
        res = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role':'user','content':prompt}], 
            temperature=0
        )
        raw = res.choices[0].message.content
    return clean_r_code(raw)

def run_r_subprocess(r_code, input_df):
    """Executes the generated R code in a controlled environment."""
    with tempfile.TemporaryDirectory() as d:
        inp_path = os.path.join(d, "input.csv")
        out_path = os.path.join(d, "output.csv")
        script_path = os.path.join(d, "script.R")
        
        input_df.to_csv(inp_path, index=False)
        
        # Build the wrapper script
        full_script = [
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE, check.names=FALSE)',
            r_code,
            f'write.csv(df, "{out_path}", row.names=FALSE)'
        ]
        
        with open(script_path, "w") as f:
            f.write("\n".join(full_script))
            
        res = subprocess.run(["Rscript", script_path], capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(f"R Error: {res.stderr}\nCode Attempted:\n{r_code}")
            
        return pd.read_csv(out_path)

def compare_dfs(sas_df, r_df, tol=1e-3):
    """Smart comparison: handles case-sensitivity and whitespace."""
    if sas_df is None or r_df is None: 
        return {"match": False, "details": "Comparison data missing."}
    
    s_df = sas_df.copy().reset_index(drop=True)
    r_df_comp = r_df.copy().reset_index(drop=True)
    
    # Normalize Columns
    s_df.columns = s_df.columns.str.upper().str.strip()
    r_df_comp.columns = r_df_comp.columns.str.upper().str.strip()

    if s_df.shape != r_df_comp.shape:
        return {"match": False, "details": f"Shape mismatch: SAS {s_df.shape} vs R {r_df_comp.shape}"}

    # Ensure column order matches
    try:
        r_df_comp = r_df_comp[s_df.columns]
    except KeyError as e:
        return {"match": False, "details": f"Missing column in R output: {e}"}

    mismatches = []
    for col in s_df.columns:
        for i in range(len(s_df)):
            val_s = s_df[col].iloc[i]
            val_r = r_df_comp[col].iloc[i]
            
            # Numeric comparison
            try:
                if abs(float(val_s) - float(val_r)) > tol:
                    mismatches.append({"col": col, "row": i, "sas": val_s, "r": val_r})
            except (ValueError, TypeError):
                # String comparison
                if str(val_s).strip().upper() != str(val_r).strip().upper():
                    mismatches.append({"col": col, "row": i, "sas": val_s, "r": val_r})
                    
    return {
        "match": len(mismatches) == 0, 
        "details": "All values match!" if not mismatches else f"{len(mismatches)} values differ.",
        "mismatches": mismatches
    }

def parse_datalines(step):
    """Extracts raw data from SAS datalines/cards block."""
    try:
        inp_match = re.search(r'input\s+(.*?);', step, re.I | re.DOTALL)
        raw_cols = inp_match.group(1).split()
        cols = [c.replace('$', '').strip() for c in raw_cols if c.strip() != '$']
        
        dl_match = re.search(r'datalines\s*;(.*?)\s*;', step, re.I | re.DOTALL)
        if not dl_match: dl_match = re.search(r'cards\s*;(.*?)\s*;', step, re.I | re.DOTALL)
        
        raw_lines = [l.strip() for l in dl_match.group(1).strip().split('\n') if l.strip()]
        rows = [line.split() for line in raw_lines]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return None

# --- PIPELINE LOGIC ---

def run_chain_pipeline(sas_code, uploaded_outputs):
    """Processes SAS steps as a continuous chain."""
    steps = re.findall(r"((?:data|proc)\s+.*?;.*?run;)", sas_code, re.DOTALL | re.I)
    work_library = {} # Dictionary to store intermediate DataFrames
    pipeline_results = []
    
    # Pre-scan for the final dataset name to highlight it
    all_out_names = re.findall(r"(?:^data\s+|out\s*=\s*)(\w+)", sas_code, re.I)
    final_ds_name = all_out_names[-1].upper() if all_out_names else None

    for i, step in enumerate(steps):
        # 1. Identify Target Name
        out_name_match = re.search(r"(?:^data\s+|out\s*=\s*)(\w+)", step, re.I)
        target_name = out_name_match.group(1).upper() if out_name_match else f"STEP_{i+1}"
        
        # 2. Identify Input Source
        set_match = re.search(r"set\s+(\w+)", step, re.I)
        source_name = set_match.group(1).upper() if set_match else None
        
        # Find the correct DataFrame to pass to the LLM/R
        if 'datalines' in step.lower() or 'cards' in step.lower():
            active_df = None
        elif source_name and source_name in work_library:
            active_df = work_library[source_name]
        elif source_name and source_name in uploaded_outputs:
            active_df = uploaded_outputs[source_name]
        elif work_library:
            active_df = list(work_library.values())[-1]
        else:
            active_df = None

        res_entry = {
            "name": target_name,
            "step": step,
            "r_code": None,
            "r_output": None,
            "error": None,
            "comparison": None,
            "is_final": (target_name == final_ds_name)
        }

        # 3. Execution Engine
        try:
            if 'datalines' in step.lower() or 'cards' in step.lower():
                out_df = parse_datalines(step)
                if out_df is None: raise ValueError("Failed to parse datalines.")
            else:
                if active_df is None:
                    raise ValueError(f"Input dataset '{source_name or 'WORK'}' not found.")
                r_code = call_llm_api(step, active_df.columns.tolist())
                res_entry["r_code"] = r_code
                out_df = run_r_subprocess(r_code, active_df)
            
            work_library[target_name] = out_df
            res_entry["r_output"] = out_df
            
            if target_name in uploaded_outputs:
                res_entry["comparison"] = compare_dfs(uploaded_outputs[target_name], out_df)
            elif target_name == final_ds_name:
                res_entry["comparison"] = {"match": None, "details": "Final output reached. Upload expected CSV to validate."}
                
        except Exception as e:
            res_entry["error"] = str(e)
            
        pipeline_results.append(res_entry)
        
    return pipeline_results

# --- STREAMLIT UI ---

st.title("🔄 Smart SAS to R Converter")
st.markdown("Convert complex SAS scripts. **Data chains automatically** between steps.")

with st.sidebar:
    st.header("1. Options")
    mode = st.radio("Workflow", ["Manual Step-by-Step", "Auto-Chaining Pipeline"])
    st.divider()
    st.info("💡 **Chaining Tip:** You don't need to upload CSVs for every step. Uploading just the **final** output CSV is enough.")

st.subheader("SAS Source Code")
sas_script = st.text_area("Paste SAS script here...", height=280, placeholder="DATA A; SET B; ... RUN;")

uploaded_csvs = {}
if mode == "Auto-Chaining Pipeline":
    st.divider()
    st.subheader("Expected Outputs (Optional)")
    files = st.file_uploader("Upload CSVs (name must match SAS dataset name)", type="csv", accept_multiple_files=True)
    if files:
        cols = st.columns(3)
        for idx, f in enumerate(files):
            name = os.path.splitext(f.name)[0].upper()
            try:
                uploaded_csvs[name] = safe_read_csv(f)
                with cols[idx % 3]:
                    st.success(f"Loaded: {name}")
            except Exception as e:
                with cols[idx % 3]:
                    st.error(f"Failed to load {name}: {str(e)}")

if st.button("⚡ Run Conversion & Pipeline", type="primary", use_container_width=True):
    if not sas_script.strip():
        st.warning("Please provide SAS code.")
    else:
        if mode == "Manual Step-by-Step":
            steps = re.findall(r"((?:data|proc)\s+.*?;.*?run;)", sas_script, re.DOTALL | re.I)
            for i, s in enumerate(steps):
                with st.expander(f"Step {i+1} Conversion"):
                    st.code(call_llm_api(s, ["unknown_cols"]), language="r")
        else:
            with st.spinner("Processing chain: LLM Conversion ➡️ R Execution..."):
                results = run_chain_pipeline(sas_script, uploaded_csvs)
            
            for res in results:
                status_icon = "⚪"
                comp = res["comparison"]
                if res["error"]: status_icon = "⚠️"
                elif comp:
                    if comp["match"] is True: status_icon = "✅"
                    elif comp["match"] is False: status_icon = "❌"
                    else: status_icon = "🏁"
                
                header = f"{status_icon} Dataset: {res['name']}"
                if res["is_final"]: header += " (FINAL)"

                with st.expander(header, expanded=res["is_final"] or res["error"] is not None):
                    if res["error"]:
                        st.error(f"Pipeline broke here: {res['error']}")
                    tab_sas, tab_r, tab_val = st.tabs(["SAS Code", "R Output", "Validation"])
                    with tab_sas:
                        st.code(res["step"], language="sas")
                        if res["r_code"]: st.code(res["r_code"], language="r")
                    with tab_r:
                        if res["r_output"] is not None: st.dataframe(res["r_output"], use_container_width=True)
                    with tab_val:
                        if comp:
                            if comp["match"] is True: st.success(comp["details"])
                            elif comp["match"] is False:
                                st.error(comp["details"])
                                if comp["mismatches"]: st.table(pd.DataFrame(comp["mismatches"]).head(10))
                            else: st.warning(comp["details"])
                        else: st.info("Intermediate step: No validation CSV provided.")

            st.divider()
            valid_steps = [r for r in results if r["comparison"] and r["comparison"]["match"] is not None]
            matches = [r for r in valid_steps if r["comparison"]["match"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Steps", len(results))
            c2.metric("Validated", len(valid_steps))
            c3.metric("Passed ✅", len(matches))
