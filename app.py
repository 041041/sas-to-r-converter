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
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj)
    except Exception:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding='latin1', sep=None, engine='python', on_bad_lines='skip')
        except Exception as e:
            raise RuntimeError(f"Could not parse CSV file. Error: {str(e)}")

def clean_r_code(text):
    """Strips LLM conversational filler, fixes dangling pipes & empty functions, and returns 'df'."""
    backticks = "\x60\x60\x60"
    if backticks in text:
        pattern = backticks + r"(?:r|python|R)?\n(.*?)\n" + backticks
        blocks = re.findall(pattern, text, re.DOTALL)
        if blocks: text = "\n".join(blocks)
    
    lines = text.split("\n")
    out = []
    forbidden = ["explanation:", "sas code:", "run;", "data.frame()", "library("]
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line or clean_line.startswith(('#', backticks)): continue
        if "data.frame(" in clean_line and "c(" in clean_line and "df =" in clean_line.lower(): continue 
        if any(x in clean_line.lower() for x in forbidden if x != "data.frame()"): continue
        if "(" in clean_line and "<-" in clean_line:
            clean_line = clean_line.replace("<-", "=")
        out.append(clean_line)
    
    cleaned = "\n".join(out)
    
    # --- SAFETY NETS ---
    cleaned = re.sub(r"%>%\s*$", "", cleaned.strip()) # Catch dangling pipes
    cleaned = re.sub(r"%>%\s*select\(\)\s*$", "", cleaned.strip()) # Catch empty selects
    cleaned = re.sub(r"%>%\s*mutate\(\)\s*$", "", cleaned.strip()) # Catch empty mutates
    
    # De-duplicate assigning pipelines if AI hallucinates repeating the exact same block
    if cleaned.count("df <- ") > 1:
        parts = cleaned.split("df <- ")
        cleaned = "df <- " + parts[-1]
        
    if not cleaned.strip().endswith("df"): cleaned += "\ndf"
    return cleaned

def call_llm_api(step, df_cols, env_names=None, dialect="Base R"):
    env_info = f"\nAvailable tables in R environment: {', '.join(env_names)}" if env_names else ""
    
    if not df_cols:
        input_context = "Convert this step. You have access to the tables listed below."
    else:
        input_context = f"A dataframe named 'df' with columns: {df_cols}"
        
    if dialect == "Modern R (dplyr)":
        rule_set = (
            f"1. Use modern R (tidyverse). Use the pipe operator (%>%) for chaining.\n"
            f"2. IF SAS uses DATALINES: ONLY create the data.frame using `data.frame(...)`. STOP immediately.\n"
            f"3. IF SAS reads an existing table: start the pipeline exactly with `df <- TABLE_NAME %>%`.\n"
            f"4. FOR DATA STEPS: Create new variables inside a populated `mutate(...)`. NEVER write an empty mutate() or select().\n"
            f"5. FOR PROC SQL: Chain multiple tables using `left_join()`. Use `select()` at the end for columns.\n"
            f"6. FIRST. LOGIC: Translate SAS 'first.variable' using `group_by(var) %>% slice(1) %>% ungroup()`.\n"
        )
    else:
        rule_set = (
            f"1. Use ONLY pure Base R. DO NOT use dplyr, tidyr, or pipes (%>%).\n"
            f"2. For aggregate(), ALWAYS use the formula interface.\n"
            f"3. IF SAS uses DATALINES: ONLY create the data.frame. STOP immediately.\n"
            f"4. IF SAS reads an existing table: start your code exactly with `df <- TABLE_NAME`.\n"
            f"5. CRITICAL: Keep all original columns. NO MATH inside aggregate() or cbind().\n"
            f"6. FIRST. LOGIC: Translate SAS 'first.variable' using `!duplicated(df$var)`.\n"
        )

    prompt = (
        f"TASK: Convert this SAS step to R code.\n"
        f"INPUT CONTEXT: {input_context}{env_info}\n"
        f"OUTPUT: Your code must result in a final dataframe named 'df'. The last line MUST be exactly 'df'.\n"
        f"STRICT RULES:\n{rule_set}"
        f"FINAL RULE: No explanations, no markdown. Just executable R code.\n\n"
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

def run_r_subprocess(r_code, input_df, env_dict=None):
    with tempfile.TemporaryDirectory() as d:
        inp_path = os.path.join(d, "input.csv")
        out_path = os.path.join(d, "output.csv")
        script_path = os.path.join(d, "script.R")
        
        input_df.to_csv(inp_path, index=False)
        
        full_script = [
            'suppressWarnings(suppressMessages(library(tidyverse)))',
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE, check.names=FALSE)'
        ]
        
        if env_dict:
            for name, df_mem in env_dict.items():
                mem_path = os.path.join(d, f"{name}.csv")
                df_mem.to_csv(mem_path, index=False)
                full_script.append(f'{name} <- read.csv("{mem_path}", stringsAsFactors=FALSE, check.names=FALSE)')
                
        full_script.append(r_code)
        full_script.append(f'write.csv(df, "{out_path}", row.names=FALSE)')
        
        with open(script_path, "w") as f:
            f.write("\n".join(full_script))
            
        res = subprocess.run(["Rscript", script_path], capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(f"R Error: {res.stderr}\nCode Attempted:\n{r_code}")
            
        return pd.read_csv(out_path)

def compare_dfs(sas_df, r_df, tol=1e-3):
    if sas_df is None or r_df is None: return {"match": False, "details": "Comparison data missing.", "mismatches": []}
    
    s_df = sas_df.copy().reset_index(drop=True)
    r_df_comp = r_df.copy().reset_index(drop=True)
    s_df.columns = s_df.columns.str.upper().str.strip()
    r_df_comp.columns = r_df_comp.columns.str.upper().str.strip()

    if s_df.shape != r_df_comp.shape:
        return {"match": False, "details": f"Shape mismatch: SAS {s_df.shape} vs R {r_df_comp.shape}", "mismatches": []}

    try:
        r_df_comp = r_df_comp[s_df.columns]
    except KeyError as e:
        return {"match": False, "details": f"Missing column in R output: {e}", "mismatches": []}

    mismatches = []
    for col in s_df.columns:
        for i in range(len(s_df)):
            val_s = s_df[col].iloc[i]
            val_r = r_df_comp[col].iloc[i]
            try:
                if abs(float(val_s) - float(val_r)) > tol:
                    mismatches.append({"col": col, "row": i, "sas": val_s, "r": val_r})
            except (ValueError, TypeError):
                if str(val_s).strip().upper() != str(val_r).strip().upper():
                    mismatches.append({"col": col, "row": i, "sas": val_s, "r": val_r})
                    
    return {
        "match": len(mismatches) == 0, 
        "details": "All values match!" if not mismatches else f"{len(mismatches)} values differ.",
        "mismatches": mismatches
    }

def parse_datalines(step):
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

def run_chain_pipeline(sas_code, uploaded_outputs, dialect):
    steps = re.findall(r"((?:data|proc)\s+.*?;.*?(?:run|quit);)", sas_code, re.DOTALL | re.I)
    work_library = {}
    pipeline_results = []
    
    all_out_names = re.findall(r"(?:^data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", sas_code, re.I)
    final_ds_name = all_out_names[-1].split('.')[-1].upper().strip() if all_out_names else None

    for i, step in enumerate(steps):
        # NEW LOGIC: Correctly map PROC SORT in-place operations to the same target name
        out_name_match = re.search(r"(?:^data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", step, re.I)
        sort_inplace_match = re.search(r"proc\s+sort\s+data\s*=\s*([\w.]+)", step, re.I)
        
        if out_name_match:
            target_name = out_name_match.group(1).split('.')[-1].upper().strip()
        elif sort_inplace_match and not re.search(r"out\s*=", step, re.I):
            target_name = sort_inplace_match.group(1).split('.')[-1].upper().strip()
        else:
            target_name = f"STEP_{i+1}"
        
        # Look for source datasets using SET, FROM, JOIN, or DATA=
        set_match = re.search(r"(?:set|from|join|data\s*=)\s+([\w.]+)", step, re.I)
        source_name = set_match.group(1).split('.')[-1].upper().strip() if set_match else None
        
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

        try:
            if 'datalines' in step.lower() or 'cards' in step.lower():
                out_df = parse_datalines(step)
                if out_df is None: raise ValueError("Failed to parse datalines.")
            else:
                if active_df is None:
                    raise ValueError(f"Input dataset '{source_name or 'WORK'}' not found.")
                r_code = call_llm_api(step, active_df.columns.tolist(), list(work_library.keys()), dialect)
                res_entry["r_code"] = r_code
                out_
