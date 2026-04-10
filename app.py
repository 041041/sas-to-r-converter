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

def call_llm_api(step, df_cols, env_names=None, dialect="Base R"):
    """Calls Gemini with a Groq fallback. Injects available table names for SQL Joins."""
    env_info = f"\nAvailable tables in R environment: {', '.join(env_names)}" if env_names else ""
    
    if not df_cols:
        input_context = "Convert this step. You have access to the tables listed below."
    else:
        input_context = f"A dataframe named 'df' with columns: {df_cols}"
        
    # Dynamically inject rules based on the user's chosen R dialect
    if dialect == "Modern R (dplyr)":
        rule_set = (
            f"1. Use modern R, specifically the tidyverse (dplyr, tidyr).\n"
            f"2. Use the pipe operator (%>%) for chaining operations.\n"
            f"3. Start the code block with `library(dplyr)`.\n"
            f"4. IF the SAS code uses DATALINES/CARDS, build the data.frame from the raw data.\n"
            f"5. IF the SAS code reads from an existing table (e.g., FROM SALES, SET WORK.SALES), start the pipeline exactly with `df <- SALES %>%`.\n"
        )
    else:
        rule_set = (
            f"1. Use ONLY pure Base R.\n"
            f"2. DO NOT use dplyr, tidyr, or pipes (%>%).\n"
            f"3. For aggregate(), ALWAYS use the formula interface (e.g., `aggregate(total_qty ~ product, data = df, FUN = sum)`). NEVER use `by = list(...)`.\n"
            f"4. ABSOLUTELY NO MATH inside aggregate() or cbind(). If SAS does sum(price*qty), do `df$new_col <- df$price * df$qty` BEFORE calling aggregate().\n"
            f"5. IF the SAS code uses DATALINES/CARDS, build the data.frame from the raw data.\n"
            f"6. IF the SAS code reads from an existing table (e.g., FROM SALES), start your code exactly with `df <- SALES`.\n"
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
    """Executes the generated R code in a controlled environment."""
    with tempfile.TemporaryDirectory() as d:
        inp_path = os.path.join(d, "input.csv")
        out_path = os.path.join(d, "output.csv")
        script_path = os.path.join(d, "script.R")
        
        input_df.to_csv(inp_path, index=False)
        
        # In case dplyr is used, ensure it is loaded in the script
        full_script = [
            'suppressWarnings(suppressMessages(library(dplyr)))',
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
        # Syntax error fixed below:
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
    """Processes SAS steps as a continuous chain."""
    steps = re.findall(r"((?:data|proc)\s+.*?;.*?(?:run|quit);)", sas_code, re.DOTALL | re.I)
    work_library = {}
    pipeline_results = []
    
    all_out_names = re.findall(r"(?:^data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", sas_code, re.I)
    final_ds_name = all_out_names[-1].split('.')[-1].upper().strip() if all_out_names else None

    for i, step in enumerate(steps):
        out_name_match = re.search(r"(?:^data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", step, re.I)
        target_name = out_name_match.group(1).split('.')[-1].upper().strip() if out_name_match else f"STEP_{i+1}"
        
        set_match = re.search(r"(?:set|from|join)\s+([\w.]+)", step, re.I)
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
                out_df = run_r_subprocess(r_code, active_df, work_library)
            
            work_library[target_name] = out_df
            res_entry["r_output"] = out_df
            
            # SMART COMPARISON LOGIC
            if target_name in uploaded_outputs:
                res_entry["comparison"] = compare_dfs(uploaded_outputs[target_name], out_df)
            elif target_name == final_ds_name and len(uploaded_outputs) == 1:
                only_csv_key = list(uploaded_outputs.keys())[0]
                res_entry["comparison"] = compare_dfs(uploaded_outputs[only_csv_key], out_df)
                res_entry["comparison"]["details"] = f"(Auto-mapped to '{only_csv_key}') " + res_entry["comparison"]["details"]
            elif target_name == final_ds_name:
                res_entry["comparison"] = {"match": None, "details": "Final output reached. Upload expected CSV to validate.", "mismatches": []}
                
        except Exception as e:
            res_entry["error"] = str(e)
            
        pipeline_results.append(res_entry)
        
    return pipeline_results

# --- STREAMLIT UI ---

st.title("🔄 Smart SAS to R Converter")
st.caption("Gemini 2.0 Flash + Groq fallback | Executes R via Rscript | Compares output vs SAS expected")
st.divider()

with st.sidebar:
    st.header("How to use")
    st.markdown("""
**Convert Only:**
1. Paste SAS code → Run
2. Download R script

---
**Convert + Validate:**
1. Paste SAS code
2. Upload expected CSVs
   - filename = dataset name
   - e.g. `LAB_RESULTS.csv`
   - Or paste CSV manually
   - *Note: App auto-maps a single CSV to the final step!*
3. Run → see ✅ MATCH / ❌ MISMATCH

---
**Supported SAS:**
- DATA step (SET, IF/ELSE)
- PROC SORT
- PROC TRANSPOSE
- PROC MEANS / FREQ
- PROC SQL (SELECT, JOIN, GROUP BY, HAVING)
""")
    st.caption("Built with Gemini + Groq + Rscript")

r_dialect = st.radio("R Dialect", ["Base R", "Modern R (dplyr)"], horizontal=True)
mode = st.radio("Mode", ["Convert Only", "Convert + Execute + Validate"], horizontal=True)
st.divider()

st.subheader("📋 SAS Code")
sas_script = st.text_area("sas", height=250, label_visibility="collapsed", placeholder="Paste your SAS code here...")

uploaded_csvs = {}

if mode == "Convert + Execute + Validate":
    st.divider()
    st.subheader("📊 Expected SAS Outputs")
    st.caption("Upload your final dataset (or intermediate ones). The app will automatically map a single uploaded CSV to the final step.")

    uploaded = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
    if uploaded:
        cols = st.columns(min(len(uploaded), 3))
        for i, f in enumerate(uploaded):
            name = os.path.splitext(f.name)[0].upper().strip()
            try:
                df = safe_read_csv(f)
                uploaded_csvs[name] = df
                with cols[i % 3]:
                    st.markdown(f"**{name}** ({df.shape[0]}r × {df.shape[1]}c)")
                    st.dataframe(df, use_container_width=True, height=140)
            except Exception as e:
                st.error(f"Failed to load {name}: {str(e)}")

    with st.expander("Or paste CSV text manually"):
        manual_name = st.text_input("Dataset name (e.g. FINAL_LABS)")
        manual_csv  = st.text_area("Paste CSV here", height=100)
        if manual_name and manual_csv:
            try:
                df = pd.read_csv(io.StringIO(manual_csv))
                uploaded_csvs[manual_name.upper().strip()] = df
                st.success(f"Loaded {manual_name.upper()} — {df.shape}")
                st.dataframe(df, height=140)
            except Exception as e:
                st.error(f"Parse error: {e}")

st.divider()
run_btn = st.button("⚡ Run", type="primary", use_container_width=True)

if run_btn:
    if not sas_script.strip():
        st.warning("Paste some SAS code first."); st.stop()
    st.divider()

    if mode == "Convert Only":
        st.subheader("Generated R Code")
        steps = re.findall(r"((?:data|proc)\s+.*?;.*?(?:run|quit);)", sas_script, re.DOTALL|re.IGNORECASE)
        if not steps: st.error("No valid SAS steps found."); st.stop()
        
        all_r = []
        known_tables = [] 
        
        for i, step in enumerate(steps, 1):
            m = re.search(r"(?:^data\s+|out\s*=\s*|create\s+table\s+)([\w.]+)", step, re.I)
            sname = m.group(1).split('.')[-1].upper().strip() if m else f"Step{i}"
            with st.expander(f"Step {i}: {sname}", expanded=True):
                t1, t2 = st.tabs(["SAS", "Generated R"])
                with t1: st.code(step.strip(), language="sas")
                with t2:
                    with st.spinner(f"Converting {sname}..."):
                        try:
                            # Pass an empty list for columns so it triggers the smart DATALINES logic
                            rc = call_llm_api(step, [], known_tables, r_dialect)
                            st.code(rc, language="r")
                            # Explicitly assign df to the dataset name in the script compilation
                            if "dplyr" in r_dialect:
                                all_r.append(f"# --- {sname} ---\nlibrary(dplyr)\n{rc}\n{sname} <- df\n")
                            else:
                                all_r.append(f"# --- {sname} ---\n{rc}\n{sname} <- df\n")
                            
                            if sname not in known_tables:
                                known_tables.append(sname)
                                
                            st.success(f"✅ {sname} converted")
                        except Exception as e: st.error(f"❌ {e}")
        
        if all_r:
            st.divider(); full = "\n".join(all_r)
            st.subheader("📥 Full R Script"); st.code(full, language="r")
            st.download_button("⬇️ Download .R", data=full, file_name="converted.R", mime="text/plain", use_container_width=True)

    else:
        st.subheader("Conversion + Execution + Validation")
        with st.spinner("Processing chain: LLM Conversion ➡️ R Execution ➡️ Data Flow..."):
            results = run_chain_pipeline(sas_script, uploaded_csvs, r_dialect)
        
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

            header = f"{badge} — {res['name']}"

            with st.expander(header, expanded=res["is_final"] or res["error"] is not None):
                if res["error"]:
                    st.error(f"Pipeline broke here: {res['error']}")
                
                t1, t2, t3, t4 = st.tabs(["SAS Code", "Generated R", "R Output", "Validation"])
                
                with t1:
                    st.code(res["step"], language="sas")
                with t2:
                    if res["r_code"]: 
                        st.code(res["r_code"], language="r")
                        # Explicitly assign df to the dataset name in the script compilation
                        if "dplyr" in r_dialect:
                            all_r.append(f"# --- {res['name']} ---\nlibrary(dplyr)\n{res['r_code']}\n{res['name']} <- df\n")
                        else:
                            all_r.append(f"# --- {res['name']} ---\n{res['r_code']}\n{res['name']} <- df\n")
                    elif not res["error"]: 
                        st.info("Datalines step — parsed directly without R.")
                with t3:
                    if res["r_output"] is not None: 
                        st.dataframe(res["r_output"], use_container_width=True)
                    else: 
                        st.info("No data output for this step.")
                with t4:
                    if cmp:
                        if cmp["match"] is True: 
                            st.success(cmp["details"])
                        elif cmp["match"] is False:
                            st.error(cmp["details"])
                            if cmp["mismatches"]: 
                                st.table(pd.DataFrame(cmp["mismatches"]).head(10))
                        else: 
                            st.warning(cmp["details"])
                    else: 
                        st.info("Intermediate step: Passed to next step automatically.")

        st.divider()
        st.subheader("📊 Summary")
        valid_steps = [r for r in results if r["comparison"] and r["comparison"]["match"] is not None]
        matches = [r for r in valid_steps if r["comparison"]["match"]]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Steps Processed", len(results))
        c2.metric("Validated Steps", len(valid_steps))
        c3.metric("Matched ✅", len(matches))

        if all_r:
            st.divider(); full = "\n".join(all_r)
            st.subheader("📥 Full R Script"); st.code(full, language="r")
            st.download_button("⬇️ Download .R Script", data=full, file_name="converted_pipeline.R", mime="text/plain", use_container_width=True)
