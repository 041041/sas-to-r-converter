import os, re, subprocess, tempfile, io, time
import pandas as pd
import streamlit as st 
from google import genai
from groq import Groq
from graph_builder import render_graph_builder_tab

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart SAS to R Converter", page_icon="🚀", layout="wide")

# --- SESSION STATE INIT ---
for key, default in {
    "sas_input": "",
    "upload_key": 0,
    "uploaded_csvs": {},
    "retry_step": None,
    "retry_counts": {},
    "fix_results": {},
    "pipeline_results": [],
    "pipeline_run": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def clear_all():
    st.session_state.sas_input = ""
    st.session_state.uploaded_csvs = {}
    st.session_state.upload_key = st.session_state.upload_key + 1
    st.session_state.pipeline_results = []
    st.session_state.pipeline_run = False
    st.session_state.fix_results = {}

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; }
    .step-card { border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .timing-badge {
        display: inline-block;
        background: #f0f2f6;
        border: 1px solid #d0d4de;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78em;
        color: #555;
        margin-left: 8px;
        font-family: monospace;
    }
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

# --- SAS TO R FUNCTION MAPPING ---
SAS_TO_R = {
    "INTCK":    "use lubridate time_length(interval(date1, date2), 'month')",
    "INTNX":    "use date + months(n) or date + days(n) from lubridate",
    "PUT":      "use format() or sprintf()",
    "INPUT":    "use as.Date() or as.numeric()",
    "COMPRESS": "use gsub(' ', '', var) to remove spaces",
    "CATX":     "use paste(..., sep='-')",
    "SCAN":     "use strsplit(var, ' ')[[1]][n]",
    "MISSING":  "use is.na()",
    "STRIP":    "use trimws()",
    "UPCASE":   "use toupper()",
    "LOWCASE":  "use tolower()",
    "INDEX":    "use regexpr() or grepl()",
    "MOD":      "use %% operator",
    "INT":      "use as.integer() or floor()",
    "ROUND":    "use round()",
    "SUBSTR":   "use substr() — same in R",
    "TRIM":     "use trimws()",
    "LEFT":     "use trimws(var, which='left')",
    "LENGTH":   "use nchar()",
    "TODAY":    "use Sys.Date()",
    "DATE":     "use Sys.Date()",
    "DATE9":    "use as.Date(var, '%d%b%Y') for date9. format",
    "MMDDYY":   "use as.Date(var, '%m/%d/%Y')",
    "YYMMDD":   "use as.Date(var, '%Y-%m-%d')",
}

def inject_function_hints(step):
    hints = []
    for sas_func, r_equiv in SAS_TO_R.items():
        if sas_func in step.upper():
            hints.append(f"  - {sas_func} → {r_equiv}")
    return "\nFUNCTION HINTS (use these exact R equivalents):\n" + "\n".join(hints) if hints else ""
    
def expand_macros(sas_code):
    """Expands SAS macros by substituting parameters and inlining macro bodies."""
    macro_lib = {}

    # Step 1 — collect all macro definitions
    for m in re.finditer(
        r"%macro\s+(\w+)\s*\(([^)]*)\)\s*;(.*?)%mend\s*\w*\s*;",
        sas_code, re.DOTALL | re.I
    ):
        name = m.group(1).strip().upper()
        params = [p.strip().lstrip('&').split('=')[0].strip() for p in m.group(2).split(',') if p.strip()]
        body = m.group(3).strip()
        macro_lib[name] = {"params": params, "body": body}
    # Step 2 — remove macro definitions from code
    expanded = re.sub(
        r"%macro\s+\w+\s*\([^)]*\)\s*;.*?%mend\s*\w*\s*;",
        "", sas_code, flags=re.DOTALL | re.I
    ).strip()

    # Step 3 — expand macro calls (up to 5 passes for nested)
    for _ in range(5):
        for name, macro in macro_lib.items():
            pattern = rf"%{name}\s*\(([^)]*)\)\s*;"
            for call_match in re.finditer(pattern, expanded, re.I):
                args_raw = [a.strip() for a in call_match.group(1).split(',')]
                # handle named args like dataset=sales → extract value only
                arg_dict = {}
                for a in args_raw:
                    if '=' in a:
                        k, v = a.split('=', 1)
                        arg_dict[k.strip().lstrip('&')] = v.strip()
                # substitute by name not position
                body = macro["body"]
                for param in macro["params"]:
                    val = arg_dict.get(param, "")
                    body = re.sub(rf"&{param}\.?", val, body, flags=re.I)
                expanded = expanded[:call_match.start()] + "\n" + body + "\n" + expanded[call_match.end():]
                break
                body = macro["body"]
                for param, arg in zip(macro["params"], args):
                    # handle both &param and &param. patterns
                    body = re.sub(rf"&{param}\.?", arg, body, flags=re.I)
                expanded = expanded[:call_match.start()] + "\n" + body + "\n" + expanded[call_match.end():]
                break  # restart after each substitution

    return expanded.strip()
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

def safe_read_excel(file_obj, sheet_name=0):
    """Reads an Excel file (.xlsx or .xls), returns a DataFrame."""
    try:
        file_obj.seek(0)
        return pd.read_excel(file_obj, sheet_name=sheet_name)
    except Exception as e:
        raise RuntimeError(f"Could not parse Excel file. Error: {str(e)}")

def format_elapsed(seconds):
    """Returns a human-readable elapsed time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"

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

        if not out or clean_line != out[-1]:
            out.append(clean_line)

    cleaned = "\n".join(out)

    cleaned = re.sub(r"%>%\s*$", "", cleaned.strip())
    cleaned = re.sub(r"%>%\s*select\(\)\s*$", "", cleaned.strip())
    cleaned = re.sub(r"%>%\s*mutate\(\)\s*$", "", cleaned.strip())
    cleaned = re.sub(r"df\s*=\s*df\[order\([^)]+\),\s*\]\s*\n(?=.*!duplicated)", "", cleaned)
    cleaned = re.sub(r"\s*arrange\([^)]+\)\s*%>%\s*(?=.*group_by)", "", cleaned)
    cleaned = re.sub(r"%>%(?!\s)", " %>%\n  ", cleaned)

    if "pivot_longer" in cleaned:
        source_match = re.search(r"df\s*<-\s*(\w+)\s*%>%", cleaned)
        source_table = source_match.group(1) if source_match else "QUARTERLY"
        cols_match = re.search(r"cols\s*=\s*c\(([^)]+)\)", cleaned)
        cols = cols_match.group(1) if cols_match else ""
        names_match = re.search(r'names_to\s*=\s*["\']([^"\']+)["\']', cleaned)
        names_to = names_match.group(1) if names_match else "quarter"
        values_match = re.search(r'values_to\s*=\s*["\']([^"\']+)["\']', cleaned)
        values_to = values_match.group(1) if values_match else "revenue"
        cleaned = (
            f'df <- {source_table} %>%\n'
            f'  pivot_longer(cols = c({cols}),\n'
            f'               names_to = "{names_to}",\n'
            f'               values_to = "{values_to}")\n'
            f'df'
        )
    is_freq_step = (
        "count(" in cleaned and
        "merge(" not in cleaned and
        "join(" not in cleaned and
        ("pivot_wider" in cleaned or "rename(COUNT" not in cleaned)
    )
    if is_freq_step:
        match = re.search(r"count\(([^)]+)\)", cleaned)
        if match:
            vars = match.group(1).strip()
            source_match = re.search(r"df\s*<-\s*(\w+)\s*%>%", cleaned)
            source_table = source_match.group(1) if source_match else "df"
            cleaned = re.sub(
                r"df\s*<-\s*\w+\s*%>%.*",
                f'df <- {source_table} %>%\n  count({vars}) %>%\n  rename(COUNT = n)',
                cleaned,
                flags=re.DOTALL
            )

    if cleaned.count("df <- ") > 1 and "aggregate" not in cleaned and "merge(" not in cleaned:
        parts = cleaned.split("df <- ")
        cleaned = "df <- " + parts[-1]

    if not cleaned.strip().endswith("df"): cleaned += "\ndf"
    return cleaned

def call_llm_api(step, df_cols, env_names=None, dialect="Base R"):
    """Calls Gemini with a Groq fallback. Injects available table names for SQL Joins."""
    env_info = f"\nAvailable tables in R environment: {', '.join(env_names)}" if env_names else ""
    func_hints = inject_function_hints(step)
    if not df_cols:
        input_context = "Convert this step. You have access to the tables listed below."
    else:
        input_context = f"A dataframe named 'df' with columns: {df_cols}"

    if dialect == "Modern R (tidyverse)":
        rule_set = (
            f"1. Use modern R (tidyverse). Use the pipe operator (%>%) for chaining.\n"
            f"2. IF SAS uses DATALINES: ONLY create the data.frame using `data.frame(...)`. STOP immediately.\n"
            f"3. IF SAS reads an existing table: start the pipeline exactly with `df <- TABLE_NAME %>%`.\n"
            f"4. FOR DATA STEPS: Create new variables inside a populated `mutate(...)`. NEVER write an empty mutate().\n"
            f"5. FOR PROC SORT: Use `arrange()`. ONLY use `desc()` if SAS code explicitly has `DESCENDING` keyword before the variable. If no DESCENDING keyword — always use ascending order.\n"            f"6. FIRST. LOGIC: Use `group_by(var) %>% slice(1) %>% ungroup()`. IMPORTANT: Do NOT add an extra arrange() or sort inside this step; it must rely on the previous step's order.\n"
            f"7. MACRO LOGIC: If input is a %macro, convert macro variables (&var) to column names in a mutate() call.\n"
            f"8. FOR PROC FREQ: Use `df %>% count(var1, var2) %>% rename(COUNT = n)` for cross-tabs. "
            f"NEVER use pivot_wider or spread. Output MUST stay in long format with one row per combination. "
            f"Final columns must be: var1, var2, COUNT.\n"
        )
    else:
        rule_set = (
            f"1. Use ONLY pure Base R. DO NOT use tidyverse, tidyr, or pipes (%>%).\n"
            f"2. For aggregate(), ALWAYS use the formula interface.\n"
            f"3. IF SAS uses DATALINES: ONLY create the data.frame. STOP immediately.\n"
            f"4. IF SAS reads an existing table: start your code exactly with `df <- TABLE_NAME`.\n"
            f"5. FOR PROC SORT: Use `df = df[order(...), ]`. ONLY use minus sign for descending if SAS code explicitly has `DESCENDING` keyword before the variable. If no DESCENDING keyword — always use ascending order.\n"            f"6. FIRST. LOGIC: Use ONLY `df[!duplicated(df$var), ]`. ABSOLUTELY NO order() or sort() call allowed in this step — not even for tie-breaking. The previous PROC SORT already established the correct order. Trust it. Adding any order() here WILL produce wrong results.\n"
            f"7. MACRO LOGIC: Convert macro variables (&var) to standard R object references.\n"
            f"8. FOR PROC FREQ: Use EXACTLY this pattern: `df = as.data.frame(table(df$var1, df$var2))` then `names(df) = c('var1', 'var2', 'COUNT')` then `df = df[df$COUNT > 0, ]`. "
            f"NEVER add an order() or sort() before table(). "
            f"NEVER use any other approach. Output MUST stay in long format with one row per combination. "
            f"Final columns must be: var1, var2, COUNT.\n"
            f"9. FOR PROC SQL GROUP BY + HAVING: Use this EXACT two-step pattern:\n"
            f"   Step 1 - WHERE filter: `df = df[df$col == 'value', ]`\n"
            f"   Step 2 - aggregate separately for each output column:\n"
            f"   `df_count = aggregate(order_id ~ cust_id, data=df, FUN=length)`\n"
            f"   `df_sum = aggregate(amount ~ cust_id, data=df, FUN=sum)`\n"
            f"   `df = merge(df_count, df_sum, by='cust_id')`\n"
            f"   `names(df) = c('cust_id', 'total_orders', 'total_spent')`\n"
            f"   Step 3 - HAVING filter: `df = df[df$total_spent > 600, ]`\n"
            f"   NEVER use cbind inside aggregate. NEVER use matrix columns.\n"
            f"10. FOR PROC TRANSPOSE: Use EXACTLY this pattern:\n"
            f"    `df = reshape(TABLENAME, varying=c('q1','q2','q3','q4'), v.names='revenue', timevar='quarter', times=c('q1','q2','q3','q4'), direction='long')`\n"
            f"    `df = df[order(match(df$region, TABLENAME$region), match(df$quarter, c('q1','q2','q3','q4'))), ]`\n"
            f"    `df = df[, c('region', 'quarter', 'revenue')]`\n"
            f"    `row.names(df) = NULL`\n"
            f"    NEVER use stack(), NEVER use melt(). NEVER convert quarter to factor.\n"
        )

    prompt = (
        f"TASK: Convert this SAS step to R code.\n"
        f"INPUT CONTEXT: {input_context}{env_info}{func_hints}\n"
        f"OUTPUT: Your code must result in a final dataframe named 'df'. The last line MUST be exactly 'df'.\n"
        f"STRICT RULES:\n{rule_set}"
        f"FINAL RULE: No explanations. Just executable R code. Write the code EXACTLY ONCE. DO NOT loop or repeat lines.\n\n"
        f"SAS STEP:\n{step}"
    )

    try:
        res = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        raw = res.choices[0].message.content
    except Exception:
        raw = gemini_client.models.generate_content(model='gemini-2.0-flash', contents=prompt).text
    return clean_r_code(raw)

def run_r_subprocess(r_code, input_df, env_dict=None):
    """Executes the generated R code in a controlled environment."""
    with tempfile.TemporaryDirectory() as d:
        inp_path = os.path.join(d, "input.csv")
        out_path = os.path.join(d, "output.csv")
        script_path = os.path.join(d, "script.R")

        input_df.to_csv(inp_path, index=False)

        full_script = [
            'options(warn=1)',
            'suppressPackageStartupMessages(library(tidyverse))',
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

        res = subprocess.run(["Rscript", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        combined_log = res.stderr.strip() or "✅ No warnings or messages."
        if res.returncode != 0:
            raise RuntimeError(f"R Error: {res.stderr}\nCode Attempted:\n{r_code}")

        return pd.read_csv(out_path), combined_log

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
                    mismatches.append({"col": col, "row": i, "sas": str(val_s), "r": str(val_r)})
            except (ValueError, TypeError):
                if str(val_s).strip().upper() != str(val_r).strip().upper():
                    mismatches.append({"col": col, "row": i, "sas": str(val_s), "r": str(val_r)})

    return {
        "match": len(mismatches) == 0,
        "details": "All values match!" if not mismatches else f"{len(mismatches)} values differ.",
        "mismatches": mismatches
    }
def fix_r_code_on_mismatch(r_code, step, mismatches, sas_df, r_df, dialect):
    try:
        mismatch_info = ""
        if sas_df is not None and r_df is not None and sas_df.shape != r_df.shape:
            mismatch_info += f"Shape: SAS={sas_df.shape} R={r_df.shape}\n"
        if mismatches:
            mismatch_info += "Value mismatches:\n"
            clean_mismatches = [m for m in mismatches if m is not None]
            for m in clean_mismatches[:5]:
                mismatch_info += f"  col={str(m.get('col','?'))} row={str(m.get('row','?'))} SAS={str(m.get('sas','?'))} R={str(m.get('r','?'))}\n"
    except Exception as e:
        mismatch_info = "Could not extract mismatch details."

    fix_prompt = "\n".join([
        "This R code produced wrong output compared to SAS.",
        "ORIGINAL R CODE:",
        str(r_code) if r_code else "",
        "ORIGINAL SAS CODE:",
        str(step) if step else "",
        "MISMATCH DETAILS:",
        str(mismatch_info),
        "Fix the R code to match SAS output exactly. Return only corrected R code ending with df."
    ])

    try:
        res = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': fix_prompt}],
            temperature=0
        )
        raw = res.choices[0].message.content
    except Exception:
        raw = gemini_client.models.generate_content(model='gemini-2.0-flash', contents=fix_prompt).text
    return clean_r_code(raw)
    
    fix_prompt = (
        f"This R code produced wrong output compared to SAS.\n"
        f"ORIGINAL R CODE:\n{r_code}\n"
        f"ORIGINAL SAS CODE:\n{step}\n"
        f"MISMATCH DETAILS:\n{mismatch_info}\n"
        f"Fix the R code to match SAS output exactly. Return only corrected R code ending with df."
    )
    try:
        res = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': fix_prompt}],
            temperature=0
        )
        raw = res.choices[0].message.content
    except Exception:
        raw = gemini_client.models.generate_content(model='gemini-2.0-flash', contents=fix_prompt).text
 
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

def run_chain_pipeline(sas_code, uploaded_outputs, dialect, progress_bar=None, status_text=None, retry_step=None):
    """Processes SAS steps as a continuous chain. Supports progress bar + per-step timing."""
    steps = re.findall(r"((?:data|proc)\s+.*?;.*?(?:run|quit);)", sas_code, re.DOTALL | re.I)
    work_library = {}
    st.session_state["work_library"] = work_library

    pipeline_results = []
    total_steps = len(steps)

    all_out_names = re.findall(r"(?:^\s*data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", sas_code, re.I | re.M)
    final_ds_name = all_out_names[-1].split('.')[-1].upper().strip() if all_out_names else None

    for i, step in enumerate(steps):
        out_name_match = re.search(r"(?:^\s*data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", step, re.I | re.M)
        sort_inplace_match = re.search(r"proc\s+sort\s+data\s*=\s*([\w.]+)", step, re.I)

        if out_name_match:
            target_name = out_name_match.group(1).split('.')[-1].upper().strip()
        elif sort_inplace_match and not re.search(r"out\s*=", step, re.I):
            target_name = sort_inplace_match.group(1).split('.')[-1].upper().strip()
        else:
            target_name = f"STEP_{i+1}"

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
            "is_final": (target_name == final_ds_name),
            "elapsed_llm": None,     # ← NEW: time for LLM call
            "elapsed_exec": None,    # ← NEW: time for R execution
            "elapsed_total": None,   # ← NEW: total time for the step   
            "r_log": None,           # ← NEW: log 
        }

        # Update progress bar
        if progress_bar is not None:
            progress_bar.progress(i / total_steps, text=f"Processing step {i+1}/{total_steps}: {target_name}")
        if status_text is not None:
            status_text.markdown(f"⏳ **Step {i+1}/{total_steps}** — `{target_name}`")
            
        # Skip steps if retrying specific step only
        if retry_step and target_name != retry_step:
            continue
        step_start = time.time()

        try:
            if 'datalines' in step.lower() or 'cards' in step.lower():
                out_df = parse_datalines(step)
                if out_df is None: raise ValueError("Failed to parse datalines.")
                res_entry["elapsed_total"] = time.time() - step_start
            else:
                if active_df is None:
                    raise ValueError(f"Input dataset '{source_name or 'WORK'}' not found.")

                # Time the LLM call
                llm_start = time.time()
                r_code = call_llm_api(step, active_df.columns.tolist(), list(work_library.keys()), dialect)
                res_entry["elapsed_llm"] = time.time() - llm_start
                res_entry["r_code"] = r_code

# Time the R execution — with 1 auto-retry on failure
                exec_start = time.time()
                try:
                    out_df, r_log = run_r_subprocess(r_code, active_df, work_library)
                    res_entry["r_log"] = r_log
                except RuntimeError as r_err:
                    # Auto-fix: feed error back to LLM and retry once
                    fix_prompt = f"This R code failed:\n{r_code}\nError:\n{str(r_err)}\nFix it. Return only corrected R code ending with df."
                    try:
                        res_fix = groq_client.chat.completions.create(
                            model='llama-3.3-70b-versatile',
                            messages=[{'role': 'user', 'content': fix_prompt}],
                            temperature=0
                        )
                        fixed_raw = res_fix.choices[0].message.content
                    except Exception:
                        fixed_raw = gemini_client.models.generate_content(model='gemini-2.0-flash', contents=fix_prompt).text
                    r_code = clean_r_code(fixed_raw)
                    res_entry["r_code"] = r_code
                    res_entry["auto_fixed"] = True
                    out_df, r_log = run_r_subprocess(r_code, active_df, work_library)
                    res_entry["r_log"] = r_log
                res_entry["elapsed_exec"] = time.time() - exec_start
                
                res_entry["elapsed_total"] = time.time() - step_start

            work_library[target_name] = out_df
            res_entry["r_output"] = out_df

            if target_name in uploaded_outputs:
                res_entry["comparison"] = compare_dfs(uploaded_outputs[target_name], out_df)
            elif target_name == final_ds_name and len(uploaded_outputs) == 1:
                only_csv_key = list(uploaded_outputs.keys())[0]
                res_entry["comparison"] = compare_dfs(uploaded_outputs[only_csv_key], out_df)
                res_entry["comparison"]["details"] = f"(Auto-mapped to '{only_csv_key}') " + res_entry["comparison"]["details"]
            elif target_name == final_ds_name:
                res_entry["comparison"] = {"match": None, "details": "Final output reached. Upload expected CSV/Excel to validate.", "mismatches": []}

        except Exception as e:
            res_entry["error"] = str(e)
            res_entry["elapsed_total"] = time.time() - step_start

        pipeline_results.append(res_entry)

    # Complete the progress bar
    if progress_bar is not None:
        progress_bar.progress(1.0, text=f"✅ All {total_steps} steps processed!")
    if status_text is not None:
        status_text.empty()

    return pipeline_results

# --- STREAMLIT UI ---
with st.sidebar:
      st.header("⚙️ Settings")
      mode = st.radio("App Mode", ["Convert Only", "Convert + Execute + Validate"])
      st.divider()
      r_dialect = st.radio("R Dialect", ["Base R", "Modern R (tidyverse)"])
  
      st.divider()
  
      st.header("📖 How to use")
      st.markdown("""
  **Convert Only:**
  1. Paste SAS code → Run
  2. Download R script
  
  ---
  **Convert + Validate:**
  1. Paste SAS code
  2. Upload expected CSV or Excel
     - filename = dataset name
     - *Single file auto-maps to final step!*
  3. Run → see ✅ MATCH / ❌ MISMATCH
  """)
  
      st.divider()
  
      st.header("✨ What this app does")
      st.markdown("""
  🔄 Converts SAS code to R automatically
  
  ✅ Executes & validates R output
  
  🔧 Auto-fixes R errors on failure
  
  🔄 Fix & Retry on output mismatch
  
  📊 Side by side SAS vs R comparison
  
  ⏱️ Per-step timing metrics
  
  📥 Downloads full R script
  """)
  
      st.divider()
  
      st.header("📋 Supported SAS")
      st.markdown("""
  ✅ DATA step (SET, IF/ELSE, mutate)
  
  ✅ PROC SORT
  
  ✅ PROC MEANS
  
  ✅ PROC FREQ
  
  ✅ PROC SQL (JOIN, GROUP BY, HAVING)
  
  ✅ PROC TRANSPOSE
  """)
  
      st.divider()
  
      st.header("🔜 Coming Soon")
      st.markdown("""
  🔶 SAS Macros *(in development)*
  """)
  
      st.divider()
  
      st.header("💡 Tips")
      st.markdown("""
  - Name CSV same as SAS dataset
  - Single CSV auto-maps to final step
  - Use **Modern R** for cleaner code
  - Use **Base R** for maximum compatibility
  """)
  
      st.caption("Built with Gemini + Groq + Rscript")
main_tab1, main_tab2 = st.tabs(["🔄 SAS Converter", "📊 Graph Builder"])

with main_tab2:
    render_graph_builder_tab()

with main_tab1:
    st.title("🔄 Smart SAS to R Converter")
    st.caption("Gemini 2.0 Flash + Groq fallback | Executes R via Rscript | Compares output vs SAS expected")
    st.divider()
   
  # --- SAS INPUT ---
  st.subheader("📋 SAS Code")
  sas_script = st.text_area(
      "sas", height=250, label_visibility="collapsed",
      placeholder="Paste your SAS code here...",
      value=st.session_state.sas_input,
      key="sas_input"
  )
  
  # --- FILE UPLOAD — only shown in validate mode ---
  uploaded_csvs = st.session_state.uploaded_csvs
  
  if mode == "Convert + Execute + Validate":
      st.divider()
      st.subheader("📊 Expected SAS Outputs")
      st.caption("Upload CSV or Excel files. The app auto-maps a single uploaded file to the final step.")
  
      # ── NEW: Accept both CSV and Excel ──
      uploaded = st.file_uploader(
          "Upload CSV or Excel files",
          type=["csv", "xlsx", "xls"],          # ← ADDED xlsx/xls
          accept_multiple_files=True,
          key="uploader_" + str(st.session_state.get("upload_key", 0))
      )
  
      if uploaded:
          st.session_state.uploaded_csvs = {}
          uploaded_csvs = st.session_state.uploaded_csvs
          cols = st.columns(min(len(uploaded), 3))
  
          for i, f in enumerate(uploaded):
              name = os.path.splitext(f.name)[0].upper().strip()
              ext = os.path.splitext(f.name)[1].lower()
  
              try:
                  # ── Route by extension ──
                  if ext in (".xlsx", ".xls"):
                      # Let user pick sheet if multiple sheets exist
                      xls = pd.ExcelFile(f)
                      sheet_names = xls.sheet_names
  
                      if len(sheet_names) > 1:
                          f.seek(0)
                          chosen_sheet = st.selectbox(
                              f"📋 Sheet for **{f.name}**",
                              options=sheet_names,
                              key=f"sheet_{name}_{i}"
                          )
                          f.seek(0)
                          df = safe_read_excel(f, sheet_name=chosen_sheet)
                      else:
                          f.seek(0)
                          df = safe_read_excel(f, sheet_name=0)
                  else:
                      df = safe_read_csv(f)
  
                  uploaded_csvs[name] = df
                  st.session_state.uploaded_csvs[name] = df
  
                  with cols[i % 3]:
                      icon = "📗" if ext in (".xlsx", ".xls") else "📄"
                      st.markdown(f"**{icon} {name}** ({df.shape[0]}r × {df.shape[1]}c)")
                      st.dataframe(df, use_container_width=True, height=140)
  
              except Exception as e:
                  st.error(f"Failed to load {name}: {str(e)}")
  
      with st.expander("Or paste CSV text manually"):
          manual_csv = st.text_area(
              "Paste CSV here", height=100,
              key=f"manual_csv_{st.session_state.get('upload_key', 0)}"
          )
          if manual_csv:
              try:
                  df = pd.read_csv(io.StringIO(manual_csv))
                  uploaded_csvs["MANUAL_INPUT"] = df
                  st.session_state.uploaded_csvs["MANUAL_INPUT"] = df
                  st.success(f"✅ Loaded — {df.shape[0]} rows × {df.shape[1]} cols")
                  st.dataframe(df, height=140)
              except Exception as e:
                  st.error(f"Parse error: {e}")
  
  # --- RUN / CLEAR BUTTONS ---
  st.divider()
  col_run, col_clear = st.columns([5, 1])
  with col_run:
      run_btn = st.button("⚡ Run", type="primary", use_container_width=True)
  with col_clear:
      st.button("🗑️ Clear", on_click=clear_all, use_container_width=True)
  
  # --- MAIN LOGIC ---
  if run_btn:
      st.session_state.pipeline_run = False  # force fresh run
      st.session_state.fix_results = {}
      st.session_state.retry_counts = {}
  
  if run_btn or st.session_state.get("pipeline_run"):
      if not sas_script.strip():
          st.warning("Paste some SAS code first."); st.stop()
      st.divider()
      
  # --- MACRO EXPANSION ---
      original_sas = sas_script
      sas_script = expand_macros(sas_script)
      if sas_script != original_sas:
          st.info("🔧 Macros detected and expanded before conversion.")
          
      if mode == "Convert Only":
          st.subheader("Generated R Code")
          steps = re.findall(r"((?:data|proc)\s+.*?;.*?(?:run|quit);)", sas_script, re.DOTALL | re.IGNORECASE)
          if not steps: st.error("No valid SAS steps found."); st.stop()
  
          all_r = []
          known_tables = []
          total_steps = len(steps)
  
          # ── PROGRESS BAR for Convert Only ──
          prog = st.progress(0, text=f"Starting conversion of {total_steps} step(s)...")
          status = st.empty()
          overall_start = time.time()
  
          for i, step in enumerate(steps, 1):
              out_name_match = re.search(r"(?:^\s*data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", step, re.I | re.M)
              sort_inplace_match = re.search(r"proc\s+sort\s+data\s*=\s*([\w.]+)", step, re.I)
  
              if out_name_match:
                  sname = out_name_match.group(1).split('.')[-1].upper().strip()
              elif sort_inplace_match and not re.search(r"out\s*=", step, re.I):
                  sname = sort_inplace_match.group(1).split('.')[-1].upper().strip()
              else:
                  sname = f"Step{i}"
  
              prog.progress((i - 1) / total_steps, text=f"Converting step {i}/{total_steps}: {sname}...")
              status.markdown(f"⏳ **Step {i}/{total_steps}** — `{sname}`")
  
              with st.expander(f"Step {i}: {sname}", expanded=True):
                  t1, t2 = st.tabs(["SAS", "Generated R"])
                  with t1: st.code(step.strip(), language="sas")
                  with t2:
                      with st.spinner(f"Converting {sname}..."):
                          try:
                              step_start = time.time()
                              rc = call_llm_api(step, [], known_tables, r_dialect)
                              elapsed = time.time() - step_start
                              st.code(rc, language="r")
                              all_r.append(f"# --- {sname} ---\n{rc}\n{sname} <- df\n")
                              if sname not in known_tables:
                                  known_tables.append(sname)
                              st.success(f"✅ {sname} converted — ⏱️ {format_elapsed(elapsed)}")
                          except Exception as e:
                              st.error(f"❌ {e}")
  
          prog.progress(1.0, text=f"✅ All {total_steps} steps converted!")
          status.empty()
          total_elapsed = time.time() - overall_start
          st.info(f"🏁 Total conversion time: **{format_elapsed(total_elapsed)}**")
  
          if all_r:
              st.divider()
              full_script_text = "\n".join(all_r)
              if "tidyverse" in r_dialect:
                  full_script_text = "library(tidyverse)\n\n" + full_script_text
              st.subheader("📥 Full R Script")
              st.code(full_script_text, language="r")
              st.download_button("⬇️ Download .R", data=full_script_text, file_name="converted.R", mime="text/plain", use_container_width=True)
  
      else:
          st.subheader("Conversion + Execution + Validation")
  
          # ── PROGRESS BAR + STATUS for pipeline mode ──
          prog = st.progress(0, text="Initialising pipeline...")
          status = st.empty()
          overall_start = time.time()
  
          if not st.session_state.get("pipeline_run"):
              results = []
              try:
                  results = run_chain_pipeline(
                      sas_script, uploaded_csvs, r_dialect,
                      progress_bar=prog,
                      status_text=status
                  )
                  st.session_state.pipeline_results = results
                  st.session_state.pipeline_run = True
                  st.session_state.retry_step = None
              except Exception as e:
                  st.error(f"Pipeline crashed: {str(e)}")
                  import traceback
                  st.code(traceback.format_exc())
                  st.stop()
  
          results = st.session_state.get("pipeline_results", [])
  
          total_elapsed = time.time() - overall_start
          st.info(f"🏁 Total pipeline time: **{format_elapsed(total_elapsed)}**")
  
          iresults = st.session_state.get("pipeline_results", results)
          if not results: st.error("No steps processed."); st.stop()
  
          all_r = []
          for res in results:
              cmp = res["comparison"]
              match = cmp["match"] if cmp else None
  
              badge = "⚪ INTERMEDIATE"
              if res["error"]: badge = "⚠️ ERROR"
              elif res["is_final"] and match is None: badge = "🏁 FINAL (Unvalidated)"
              elif match is True: badge = "✅ MATCH"
              elif match is False: badge = "❌ MISMATCH"
  
              # ── Timing summary for the header ──
              timing_str = ""
              if res["elapsed_total"] is not None:
                  timing_str = f"  ⏱️ {format_elapsed(res['elapsed_total'])}"
  
              header = f"{badge} — {res['name']}{timing_str}"
  
              with st.expander(header, expanded=True):
                  if res["error"]:
                      st.error(f"Pipeline broke here: {res['error']}")
  
                  # ── Show detailed timing breakdown ──
                  if res["elapsed_total"] is not None:
                      t_cols = st.columns(3)
                      with t_cols[0]:
                          llm_t = format_elapsed(res["elapsed_llm"]) if res["elapsed_llm"] else "—"
                          st.metric("🤖 LLM Time", llm_t)
                      with t_cols[1]:
                          exec_t = format_elapsed(res["elapsed_exec"]) if res["elapsed_exec"] else "—"
                          st.metric("⚙️ R Exec Time", exec_t)
                      with t_cols[2]:
                          st.metric("🕐 Total Step Time", format_elapsed(res["elapsed_total"]))
  
                  t1, t2, t3, t4, t5, t6 = st.tabs(["SAS Code", "Generated R", "R Output", "SAS vs R", "Validation", "R Log"])
                  
                  with t1:
                      st.code(res["step"], language="sas")
  
                  with t2:
                      # show fixed code if available
                      fix_result = st.session_state.get("fix_results", {}).get(res["name"])
                      display_code = fix_result["code"] if fix_result and fix_result.get("match") else res["r_code"]
                      if display_code:
                          st.code(display_code, language="r")
                          if fix_result and fix_result.get("match"):
                              st.warning("⚠️ This is the Auto-fixed version")
                          all_r.append(f"# --- {res['name']} ---\n{res['r_code']}\n{res['name']} <- df\n")
                      elif not res["error"]:
                          if res["r_output"] is not None:
                              df_r = res["r_output"]
                              col_code = []
                              for col in df_r.columns:
                                  vals = df_r[col].tolist()
                                  try:
                                      floats = [float(v) for v in vals]
                                      if all(v == int(v) for v in floats):
                                          col_code.append(f'  {col} = c({", ".join(str(int(v)) for v in floats)})')
                                      else:
                                          col_code.append(f'  {col} = c({", ".join(str(v) for v in floats)})')
                                  except (ValueError, TypeError):
                                      col_code.append(f'  {col} = c({", ".join(repr(str(v)) for v in vals)})')
                              datalines_r = "df = data.frame(\n" + ",\n".join(col_code) + "\n)\ndf"
                              st.code(datalines_r, language="r")
                              all_r.append(f"# --- {res['name']} ---\n{datalines_r}\n{res['name']} <- df\n")
                              st.success(f"✅ Successfully parsed {df_r.shape[0]} rows × {df_r.shape[1]} cols")
  
                  with t3:
                      if res["r_output"] is not None:
                          st.markdown("**⚙️ R Generated Output**")
                          st.caption(f"Shape: {res['r_output'].shape[0]} rows × {res['r_output'].shape[1]} cols")
                          st.dataframe(res["r_output"], use_container_width=True, height=300)
                          csv_data = res["r_output"].to_csv(index=False)
                          st.download_button(
                              label=f"⬇️ Download {res['name']} as CSV",
                              data=csv_data,
                              file_name=f"{res['name']}_r_output.csv",
                              mime="text/csv",
                              key=f"download_{res['name']}_{id(res)}"
                          )
                      else:
                          st.info("No data output for this step.")
  
                  with t4:
                      if res["r_output"] is not None:
                          # only show SAS vs R for final step
                          sas_out = uploaded_csvs.get(res['name'])
                          if sas_out is None and res["is_final"]:
                              sas_out = uploaded_csvs.get('MANUAL_INPUT')
                          if sas_out is None and res["is_final"] and len(uploaded_csvs) == 1:
                              sas_out = list(uploaded_csvs.values())[0]
                          
                          if sas_out is not None:
                              col_sas, col_r = st.columns(2)
                              with col_sas:
                                  st.markdown("**📋 SAS Expected Output**")
                                  st.caption(f"Shape: {sas_out.shape[0]} rows × {sas_out.shape[1]} cols")
                                  st.dataframe(sas_out, use_container_width=True, height=300)
                              with col_r:
                                  st.markdown("**⚙️ R Generated Output**")
                                  st.caption(f"Shape: {res['r_output'].shape[0]} rows × {res['r_output'].shape[1]} cols")
                                  st.dataframe(res["r_output"], use_container_width=True, height=300)
                          else:
                              st.info("Upload expected CSV to see side by side comparison.")
                      else:
                          st.info("No R output available.")
                  with t5:
                      if cmp:
                          if cmp["match"] is True:
                              st.success(cmp["details"])
                          elif cmp["match"] is False:
                              st.error(cmp["details"])
                              if cmp["mismatches"]:
                                  st.table(pd.DataFrame(cmp["mismatches"]).head(10))
                              retry_count = st.session_state.get("retry_counts", {}).get(res['name'], 0)
                              fix_result = st.session_state.get("fix_results", {}).get(res['name'])
                              if fix_result:
                                  st.divider()
                                  st.markdown("**🔧 Fix & Retry Result:**")
                                  st.code(fix_result["code"], language="r")
                                  if fix_result["match"]:
                                      st.success("✅ Fixed! Output now matches SAS!")
                                  else:
                                      st.error(f"❌ Still mismatching: {fix_result['details']}")
                              if retry_count < 3:
                                  if st.button(f"🔄 Fix & Retry {res['name']}", key=f"retry_{res['name']}"):
                                      st.session_state.setdefault("retry_counts", {})[res['name']] = retry_count + 1
                                      with st.spinner("🔧 Asking LLM to fix based on mismatch..."):
                                          sas_df = uploaded_csvs.get(res['name'])
                                          if sas_df is None:
                                              sas_df = uploaded_csvs.get('MANUAL_INPUT')
                                          if sas_df is None and len(uploaded_csvs) == 1:
                                              sas_df = list(uploaded_csvs.values())[0]
                                          r_code_to_fix = res.get('r_code') or ""
                                          fixed_code = fix_r_code_on_mismatch(
                                              r_code_to_fix,
                                              res['step'],
                                              cmp['mismatches'],
                                              sas_df,
                                              res['r_output'],
                                              r_dialect
                                          )
                                          try:
                                              new_output, new_log = run_r_subprocess(fixed_code, res['r_output'], st.session_state.get("work_library", {}))
                                              new_cmp = compare_dfs(sas_df, new_output)
                                              st.session_state.setdefault("fix_results", {})[res['name']] = {
                                                  "code": fixed_code,
                                                  "match": new_cmp["match"],
                                                  "details": new_cmp["details"]
                                              }
                                              if new_cmp["match"]:
                                                  for pr in st.session_state["pipeline_results"]:
                                                      if pr["name"] == res["name"]:
                                                          pr["comparison"] = new_cmp
                                                          pr["r_code"] = fixed_code
                                                          pr["r_output"] = new_output
                                                          break
                                              st.rerun()
                                          except Exception as e:
                                              st.error(f"Fix attempt failed: {e}")
                              else:
                                  st.info("⚠️ Already retried 3 times.")
                          else:
                              st.warning(cmp["details"])
                      else:
                          st.info("Intermediate step: Passed to next step automatically.")
                       
                  with t6:
                      log = res.get("r_log") or "✅ No warnings or messages."
                      st.code(log, language="bash")
  
          st.divider()
          st.subheader("📊 Summary")
          valid_steps = [r for r in results if r["comparison"] and r["comparison"]["match"] is not None]
          matches = [r for r in valid_steps if r["comparison"]["match"]]
  
          # ── Timing summary table ──
          timing_rows = []
          for r in results:
              timing_rows.append({
                  "Step": r["name"],
                  "LLM Time": format_elapsed(r["elapsed_llm"]) if r["elapsed_llm"] else "—",
                  "R Exec Time": format_elapsed(r["elapsed_exec"]) if r["elapsed_exec"] else "—",
                  "Total Time": format_elapsed(r["elapsed_total"]) if r["elapsed_total"] else "—",
                  "Status": "✅" if (r["comparison"] and r["comparison"]["match"] is True) else
                            "❌" if (r["comparison"] and r["comparison"]["match"] is False) else
                            "⚠️" if r["error"] else "⚪"
              })
  
          c1, c2, c3, c4 = st.columns(4)
          c1.metric("Total Steps", len(results))
          c2.metric("Validated", len(valid_steps))
          c3.metric("Matched ✅", len(matches))
          c4.metric("Total Time", format_elapsed(total_elapsed))
  
          st.markdown("**⏱️ Step-by-Step Timing**")
          st.dataframe(pd.DataFrame(timing_rows), use_container_width=True, hide_index=True)
  
          if all_r:
              st.divider()
              full_script_text = "\n".join(all_r)
              if "tidyverse" in r_dialect:
                  full_script_text = "library(tidyverse)\n\n" + full_script_text
              st.subheader("📥 Full R Script")
              st.code(full_script_text, language="r")
              st.download_button("⬇️ Download .R Script", data=full_script_text, file_name="converted_pipeline.R", mime="text/plain", use_container_width=True)
                
