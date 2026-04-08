# SAS to R Converter - Streamlit App
# Features: LLM conversion + R execution + SAS vs R comparison

import os, re, subprocess, tempfile, io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SAS to R Converter", page_icon="🔄", layout="wide")

from google import genai
from groq import Groq

def get_secret(key):
    try: return st.secrets[key]
    except Exception: return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")

if not GEMINI_API_KEY or not GROQ_API_KEY:
    st.error("API keys missing! Go to Streamlit Cloud App Settings -> Secrets and add:")
    st.code('GEMINI_API_KEY = "your_key"\nGROQ_API_KEY = "your_key"', language="toml")
    st.stop()

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

def clean_r_code(text):
    if "```" in text:
        blocks = re.findall(r"```(?:r|python|R)?\n(.*?)\n```", text, re.DOTALL)
        if blocks: text = "\n".join(blocks)
    def fix_args(t):
        res, stack, i = "", 0, 0
        while i < len(t):
            c = t[i]
            if c == '(': stack += 1; res += c
            elif c == ')': stack -= 1; res += c
            elif stack > 0 and t[i:i+2] == '<-': res += '='; i += 1
            else: res += c
            i += 1
        return res
    text = fix_args(text)
    lines, out = text.split("\n"), []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(('#','```','library','print','display')): continue
        if any(x in line.lower() for x in ["explanation:","sas code:","run;","data.frame()"]): continue
        if "(" not in line and "=" not in line: line = re.sub(r'\s+=\s+',' <- ',line)
        out.append(line)
    cleaned = "\n".join(out)
    if "df" not in cleaned.split()[-1:]: cleaned += "\ndf"
    return cleaned

def enforce_df_usage(code, df_cols):
    if not code.strip(): return "df"
    out = []
    for line in code.split("\n"):
        if line.strip() == "df": out.append(line); continue
        temp = line
        for col in df_cols:
            temp = re.sub(rf"(?<!df\$)\b{col}\b", f"df${col}", temp)
        if "<-" in temp:
            parts = temp.split("<-",1); lhs, rhs = parts[0], parts[1]
            if "df$" not in lhs and lhs.strip() != "df":
                temp = f"df${lhs.strip()} <- {rhs.strip()}"
        out.append(temp)
    return "\n".join(out)

def call_llm_api(step, df_cols):
    is_proc = 'proc ' in step.lower()
    is_trans = 'transpose' in step.lower()
    prompt = f"TASK: Convert SAS to Base R. Input data is 'df'. Final line MUST be 'df'. NO EXPLANATIONS. SAS:\n{step}"
    if is_trans:
        prompt += "\nCRITICAL: For PREFIX=/SUFFIX=: store unique ID levels, reshape(), rename with paste0(). Use '=' inside reshape() args."
    try:
        raw = gemini_client.models.generate_content(model='gemini-2.0-flash', contents=prompt).text
    except Exception:
        res = groq_client.chat.completions.create(model='llama-3.3-70b-versatile',
              messages=[{'role':'user','content':prompt}], temperature=0)
        raw = res.choices[0].message.content
    cleaned = clean_r_code(raw)
    return cleaned if is_proc else enforce_df_usage(cleaned, df_cols)

def run_r_subprocess(r_code, input_df):
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d,"input.csv"); out = os.path.join(d,"output.csv"); scr = os.path.join(d,"s.R")
        input_df.to_csv(inp, index=False)
        with open(scr,"w") as f:
            f.write(f'df <- read.csv("{inp}", stringsAsFactors=FALSE)\n{r_code}\nwrite.csv(df,"{out}",row.names=FALSE)\n')
        res = subprocess.run(["Rscript",scr], capture_output=True, text=True, timeout=30)
        if res.returncode != 0: raise RuntimeError(res.stderr)
        return pd.read_csv(out)

def compare_dfs(sas_df, r_df, tol=1e-3):
    sas_df = sas_df.reset_index(drop=True)
    r_df   = r_df.reset_index(drop=True)
    if sas_df.shape != r_df.shape:
        return {"match":False,"details":f"Shape mismatch: SAS{sas_df.shape} vs R{r_df.shape}","mismatches":[]}
    sas_df.columns = sas_df.columns.str.upper()
    r_df.columns   = r_df.columns.str.upper()
    sc, rc = set(sas_df.columns), set(r_df.columns)
    if sc != rc:
        return {"match":False,"details":f"Column mismatch. Missing:{sc-rc} Extra:{rc-sc}","mismatches":[]}
    r_df = r_df[sas_df.columns]
    mismatches = []
    for col in sas_df.columns:
        for i in range(len(sas_df)):
            sv, rv = sas_df[col].iloc[i], r_df[col].iloc[i]
            try:
                if abs(float(sv)-float(rv)) > tol: mismatches.append({"col":col,"row":i,"sas":sv,"r":rv})
            except (ValueError,TypeError):
                if str(sv).strip().upper() != str(rv).strip().upper(): mismatches.append({"col":col,"row":i,"sas":sv,"r":rv})
    match = len(mismatches)==0
    return {"match":match,"details":"All values match!" if match else f"{len(mismatches)} value(s) differ","mismatches":mismatches}

def parse_datalines(step):
    """Parse SAS datalines block into a pandas DataFrame automatically."""
    try:
        # Extract column names from INPUT statement
        inp_match = re.search(r'input\s+(.*?);', step, re.IGNORECASE|re.DOTALL)
        if not inp_match: return None
        raw_cols = inp_match.group(1).split()
        # Remove SAS type indicators like $ (marks string cols)
        cols = [c.replace('$','').strip() for c in raw_cols if c.strip() != '$']
        str_cols = set()
        i = 0
        tokens = inp_match.group(1).split()
        for j, tok in enumerate(tokens):
            if tok == '$' and j > 0:
                str_cols.add(tokens[j-1].replace('$','').strip())
            elif tok.endswith('$'):
                str_cols.add(tok.replace('$','').strip())

        # Extract datalines block
        dl_match = re.search(r'datalines\s*;(.*?)\s*;', step, re.IGNORECASE|re.DOTALL)
        if not dl_match: return None
        raw_lines = [l.strip() for l in dl_match.group(1).strip().split('\n') if l.strip()]

        rows = []
        for line in raw_lines:
            vals = line.split()
            if len(vals) == len(cols):
                row = {}
                for col, val in zip(cols, vals):
                    if col in str_cols:
                        row[col] = val
                    else:
                        try: row[col] = float(val) if '.' in val else int(val)
                        except ValueError: row[col] = val
                rows.append(row)
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None

def run_pipeline_validate(sas_code, sas_outputs):
    steps = re.findall(r"((?:data|proc)\s+.*?;.*?run;)", sas_code, re.DOTALL|re.IGNORECASE)
    env, results = {}, []
    for step in steps:
        m = re.search(r"(?:^data\s+|out\s*=\s*)(\w+)", step, re.I)
        if not m: continue
        name = m.group(1).upper()
        res = {"name":name,"step":step,"r_code":None,"r_output":None,
               "sas_output":sas_outputs.get(name),"comparison":None,"error":None}
        if 'datalines' in step.lower():
            # Try uploaded CSV first, then auto-parse from SAS datalines
            if name in sas_outputs:
                seed_df = sas_outputs[name]
            else:
                seed_df = parse_datalines(step)
            if seed_df is not None:
                env[name] = seed_df
                res["r_output"] = seed_df
                if name in sas_outputs:
                    res["comparison"] = {"match":True,"details":"Datalines — using SAS output directly","mismatches":[]}
                else:
                    res["comparison"] = {"match":None,"details":"Datalines parsed from SAS code automatically","mismatches":[]}
            else:
                res["error"] = "Could not parse datalines — please upload a CSV for this dataset"
            results.append(res); continue
        inp = list(env.values())[-1] if env else None
        if inp is None:
            res["error"] = "No input DataFrame for this step — upload a CSV or fix datalines parsing"; results.append(res); continue
        try:
            r_code = call_llm_api(step, inp.columns.tolist())
            res["r_code"] = r_code
        except Exception as e:
            res["error"] = f"LLM error: {e}"; results.append(res); continue
        try:
            r_out = run_r_subprocess(r_code, inp)
            res["r_output"] = r_out; env[name] = r_out
        except Exception as e:
            res["error"] = f"R execution error: {e}"; results.append(res); continue
        if name in sas_outputs and sas_outputs[name] is not None:
            res["comparison"] = compare_dfs(sas_outputs[name], r_out)
        results.append(res)
    return results

# ── UI ────────────────────────────────────────────────────────

st.title("🔄 SAS to R Converter")
st.caption("Gemini 2.0 Flash + Groq fallback | Executes R via Rscript | Compares output vs SAS expected")
st.divider()

mode = st.radio("Mode", ["Convert Only", "Convert + Execute + Validate"], horizontal=True)
st.divider()

st.subheader("📋 SAS Code")
sas_input = st.text_area("sas", height=250, label_visibility="collapsed",
    placeholder="Paste your SAS code here...")

sas_outputs = {}

if mode == "Convert + Execute + Validate":
    st.divider()
    st.subheader("📊 Expected SAS Outputs")
    st.caption("Upload one CSV per dataset — filename must match the SAS dataset name (e.g. LAB_RESULTS.csv)")

    uploaded = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
    if uploaded:
        cols = st.columns(min(len(uploaded), 3))
        for i, f in enumerate(uploaded):
            name = os.path.splitext(f.name)[0].upper()
            df = pd.read_csv(f)
            sas_outputs[name] = df
            with cols[i % 3]:
                st.markdown(f"**{name}** ({df.shape[0]}r × {df.shape[1]}c)")
                st.dataframe(df, use_container_width=True, height=140)

    with st.expander("Or paste CSV text manually"):
        manual_name = st.text_input("Dataset name (e.g. FINAL_LABS)")
        manual_csv  = st.text_area("Paste CSV here", height=100)
        if manual_name and manual_csv:
            try:
                df = pd.read_csv(io.StringIO(manual_csv))
                sas_outputs[manual_name.upper()] = df
                st.success(f"Loaded {manual_name.upper()} — {df.shape}")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Parse error: {e}")

st.divider()
run_btn = st.button("⚡ Run", type="primary", use_container_width=True)

if run_btn:
    if not sas_input.strip():
        st.warning("Paste some SAS code first."); st.stop()
    st.divider()

    if mode == "Convert Only":
        st.subheader("Generated R Code")
        steps = re.findall(r"((?:data|proc)\s+.*?;.*?run;)", sas_input, re.DOTALL|re.IGNORECASE)
        if not steps: st.error("No valid SAS steps found."); st.stop()
        all_r = []
        for i, step in enumerate(steps, 1):
            m = re.search(r"(?:^data\s+|out\s*=\s*)(\w+)", step, re.I)
            sname = m.group(1) if m else f"Step{i}"
            with st.expander(f"Step {i}: {sname}", expanded=True):
                t1, t2 = st.tabs(["SAS", "Generated R"])
                with t1: st.code(step.strip(), language="sas")
                with t2:
                    with st.spinner(f"Converting {sname}..."):
                        try:
                            rc = call_llm_api(step, [])
                            st.code(rc, language="r"); all_r.append(f"# --- {sname} ---\n{rc}")
                            st.success(f"✅ {sname} converted")
                        except Exception as e: st.error(f"❌ {e}")
        if all_r:
            st.divider(); full = "\n\n".join(all_r)
            st.subheader("📥 Full R Script"); st.code(full, language="r")
            st.download_button("⬇️ Download .R", data=full, file_name="converted.R", mime="text/plain", use_container_width=True)

    else:
        st.subheader("Conversion + Execution + Validation")
        with st.spinner("Running pipeline: LLM → R execution → comparison..."):
            results = run_pipeline_validate(sas_input, sas_outputs)
        if not results: st.error("No steps processed."); st.stop()

        all_r = []
        for res in results:
            cmp   = res["comparison"]
            match = cmp["match"] if cmp else None
            badge = "✅ MATCH" if match is True else ("❌ MISMATCH" if match is False else "⚙️ NO VALIDATION")

            with st.expander(f"{badge}  —  {res['name']}", expanded=True):
                t1,t2,t3,t4,t5 = st.tabs(["SAS Code","Generated R","R Output","SAS Expected","Comparison"])
                with t1: st.code(res["step"].strip(), language="sas")
                with t2:
                    if res["r_code"]: st.code(res["r_code"], language="r"); all_r.append(f"# --- {res['name']} ---\n{res['r_code']}")
                    elif res["error"]: st.error(res["error"])
                    else: st.info("Datalines step — no R code generated")
                with t3:
                    if res["r_output"] is not None: st.dataframe(res["r_output"], use_container_width=True)
                    elif res["error"]: st.error(res["error"])
                    else: st.info("R not executed")
                with t4:
                    if res["sas_output"] is not None: st.dataframe(res["sas_output"], use_container_width=True)
                    else: st.info("No SAS expected output provided")
                with t5:
                    if cmp is None: st.info("No SAS expected output uploaded — cannot compare")
                    elif cmp["match"]: st.success(f"✅ MATCH — {cmp['details']}")
                    else:
                        st.error(f"❌ MISMATCH — {cmp['details']}")
                        if cmp["mismatches"]: st.dataframe(pd.DataFrame(cmp["mismatches"]), use_container_width=True)

        st.divider(); st.subheader("📊 Summary")
        total   = len([r for r in results if r["comparison"]])
        matched = len([r for r in results if r["comparison"] and r["comparison"]["match"]])
        c1,c2,c3 = st.columns(3)
        c1.metric("Steps Validated", total)
        c2.metric("Matched ✅", matched)
        c3.metric("Mismatched ❌", total - matched)

        if all_r:
            st.divider(); full = "\n\n".join(all_r)
            st.subheader("📥 Full R Script"); st.code(full, language="r")
            st.download_button("⬇️ Download .R", data=full, file_name="converted.R", mime="text/plain", use_container_width=True)

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
3. Run → see ✅ MATCH / ❌ MISMATCH

---
**Supported SAS:**
- DATA step (SET, IF/ELSE)
- PROC SORT
- PROC TRANSPOSE
- PROC MEANS / FREQ
""")
    st.caption("Built with Gemini + Groq + Rscript")
