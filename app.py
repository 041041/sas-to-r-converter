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
        file_obj.seek(0)
        return pd.read_csv(file_obj)
    except Exception:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding='latin1', sep=None, engine='python', on_bad_lines='skip')
        except Exception as e:
            raise RuntimeError(f"Could not parse CSV file. Error: {str(e)}")

def clean_r_code(text):
    """Strips LLM conversational filler and ensures the code returns 'df'."""
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
        # Only ban data.frame() if it looks like mock data generation
        if "data.frame(" in clean_line and "c(" in clean_line and "df =" in clean_line.lower(): continue 
        if any(x in clean_line.lower() for x in forbidden if x != "data.frame()"): continue
        if "(" in clean_line and "<-" in clean_line:
            clean_line = clean_line.replace("<-", "=")
        out.append(clean_line)
    
    cleaned = "\n".join(out)
    if not cleaned.strip().endswith("df"): cleaned += "\ndf"
    return cleaned

def call_llm_api(step, df_cols, env_names=None):
    """Calls Gemini with a Groq fallback. Injects available table names for SQL Joins."""
    env_info = f"\nAvailable tables in R environment: {', '.join(env_names)}" if env_names else ""
    
    if not df_cols:
        input_context = "Convert this step. You have access to the tables listed below."
    else:
        input_context = f"A dataframe named 'df' with columns: {df_cols}"
        
    prompt = (
        f"TASK: Convert this SAS step to pure Base R code.\n"
        f"INPUT CONTEXT: {input_context}{env_info}\n"
        f"OUTPUT: Your code must result in a final dataframe named 'df'. The last line MUST be exactly 'df'.\n"
        f"STRICT RULES:\n"
        f"1. Use ONLY pure Base R.\n"
        f"2. DO NOT use dplyr, tidyr, or pipes (%>%).\n"
        f"3. For aggregate(), ALWAYS use the formula interface (e.g., `aggregate(total_qty ~ product, data = df, FUN = sum)`) to keep clean column names. NEVER use `by = list(...)` which creates ugly names like 'Group.1' and 'x'.\n"
        f"4. ABSOLUTELY NO MATH inside aggregate() or cbind(). If SAS does sum(price*qty), do `df$new_col <- df$price * df$qty` BEFORE calling aggregate().\n"
        f"5. IF the SAS code uses DATALINES/CARDS, build the data.frame from the raw data.\n"
        f"6. IF the SAS code reads from an existing table (e.g., FROM SALES, SET WORK.SALES), start your code exactly with `df <- SALES`. NEVER generate mock/dummy data.\n"
        f"7. No explanations, no markdown. Just executable R code.\n\n"
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
    """Executes the generated R code in a controlled environment."""
    with tempfile.TemporaryDirectory() as d:
        inp_path = os.path.join(d, "input.csv")
        out_path = os.path.join(d, "output.csv")
        script_path = os.path.join(d, "script.R")
        
        input_df.to_csv(inp_path, index=False)
        
        full_script = [
            f'df <- read.csv("{inp_path}", stringsAsFactors=FALSE, check.names=FALSE)'
        ]
        
        # Inject the entire Work Library into R for SQL Joins
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
    """Smart comparison: handles case-sensitivity and whitespace."""
    if sas_df is None or r_df is None: 
        return {"match": False, "details": "Comparison data missing.", "mismatches": []}
    
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
    """Extracts raw data from SAS datalines/cards block."""
    try:
        inp_match = re.search(r'input\s+(.*?);', step, re.I | re.DOTALL)
        raw_cols = inp_match.group(1).split()
        cols = [c.replace('$', '').strip() for c in raw_cols if c.strip() != '$
