"""
tlf_shell_builder.py
─────────────────────────────────────────────────────────────────────────────
TLF from Mock Shell  — agentic pipeline (LangGraph-style state machine)
Nodes: parse_shell → plan_code → generate_code → execute → validate → fix
Same patterns as graph_builder.py: file upload OR paste, AI box, diff review,
Gemini primary / Groq fallback, session-state prefixed ms_ to avoid collisions.
─────────────────────────────────────────────────────────────────────────────
"""

import os, re, io, subprocess, tempfile, traceback
from typing import TypedDict, Optional
import pandas as pd
import streamlit as st
from groq import Groq
from google import genai

# ─── Import execute_graph from graph_builder for Figure support ──────────────
try:
    from graph_builder import execute_graph as _execute_graph_fn
    _GRAPH_BUILDER_AVAILABLE = True
except ImportError:
    _GRAPH_BUILDER_AVAILABLE = False
    _execute_graph_fn = None

# ─── Clients (same helper as graph_builder) ──────────────────────────────────
def _get_secret(key):
    try:    return st.secrets[key]
    except Exception: return os.environ.get(key, "")

_gemini = genai.Client(api_key=_get_secret("GEMINI_API_KEY"))
_groq   = Groq(api_key=_get_secret("GROQ_API_KEY"))

MAX_RETRIES = 3

# ── R package auto-installer ──────────────────────────────────────────────────
_R_PACKAGES = [
    "gt", "dplyr", "tidyr", "ggplot2", "flextable",
    "tibble", "stringr", "scales", "officer",
]

# Critical packages that must be verified — others are nice-to-have
_R_PACKAGES_CRITICAL = ["dplyr", "tidyr", "ggplot2", "gt"]

def _ensure_r_packages() -> None:
    """
    Install any missing R packages into ~/R/library at session startup.
    Runs once per Streamlit session (guarded by st.session_state).
    Critical packages (dplyr, tidyr, ggplot2, gt) are verified after install.
    """
    if st.session_state.get("_r_packages_ready"):
        return

    user_lib = os.path.expanduser("~/R/library")
    os.makedirs(user_lib, exist_ok=True)

    pkg_list      = ", ".join(f'"{p}"' for p in _R_PACKAGES)
    critical_list = ", ".join(f'"{p}"' for p in _R_PACKAGES_CRITICAL)

    install_script = f"""
user_lib <- path.expand("~/R/library")
dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

# Install missing packages
pkgs    <- c({pkg_list})
missing <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)]
if (length(missing) > 0) {{
  install.packages(
    missing,
    lib   = user_lib,
    repos = "https://cloud.r-project.org",
    quiet = TRUE,
    Ncpus = 2
  )
}}

# Verify critical packages can be loaded
critical <- c({critical_list})
failed   <- critical[!sapply(critical, requireNamespace, quietly=TRUE)]
if (length(failed) > 0) {{
  cat("CRITICAL_LOAD_FAILURE:", paste(failed, collapse=","), "\\n")
}}
"""
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", "-e", install_script],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"[R package install warning] {result.stderr[:400]}")
        # Surface critical failures to Streamlit UI
        if "CRITICAL_LOAD_FAILURE:" in result.stdout:
            failed_pkgs = result.stdout.split("CRITICAL_LOAD_FAILURE:")[1].split("\n")[0].strip()
            st.warning(f"⚠️ R packages failed to load: {failed_pkgs}. Tables/figures may not render correctly.")
    except Exception as e:
        print(f"[R package install error] {e}")
    except Exception as e:
        print(f"[R package install error] {e}")

    st.session_state["_r_packages_ready"] = True


# ─── LangGraph-style State ────────────────────────────────────────────────────


def _sanitise_r_code(code: str) -> str:
    """
    Strip known-invalid gt arguments and fix column aliases that the LLM
    sometimes adds, causing R execution failures.
    Called from node_generate_code, node_fix, and the enhancement apply path.
    """
    # ── treatment column alias ─────────────────────────────────────────────
    _trt_subs = [
        (r'group_by\(treatment\)',              'group_by(`_trt_`)'),
        (r'group_by\(TREATMENT\)',              'group_by(`_trt_`)'),
        (r'names_from\s*=\s*treatment',       'names_from=`_trt_`'),
        (r'names_from\s*=\s*TREATMENT',       'names_from=`_trt_`'),
        (r'select\(treatment,',                 'select(`_trt_`,'),
        (r'by\s*=\s*"treatment"',               'by="_trt_"'),
        (r'df\$treatment',                    'df[["_trt_"]]'),
        (r'df\$TREATMENT',                    'df[["_trt_"]]'),
        (r'df\[\["treatment"\]\]',              'df[["_trt_"]]'),
        (r'df\[\["TREATMENT"\]\]',              'df[["_trt_"]]'),
        (r'unique\(df\$treatment\)',             'unique(df[["_trt_"]])'),
        (r'pivot_wider\(names_from=treatment,', 'pivot_wider(names_from=`_trt_`,'),
    ]
    for pat, repl in _trt_subs:
        code = re.sub(pat, repl, code)

    # ── invalid gt tab_options / tab_style arguments ───────────────────────
    _INVALID_GT_ARGS = [
        r',?\s*heading\.style\s*=\s*(?:list\([^)]*\)|"[^"]*")',
        r",?\s*heading\.style\s*=\s*'[^']*'",
        r',?\s*table\.font\.names\s*=\s*[^,)]+',
        r',?\s*table\.border\.style\s*=\s*"[^"]*"',
        r',?\s*row_group\.font\.weight\s*=\s*"[^"]*"',
        r',?\s*stub\.font\.weight\s*=\s*"[^"]*"',
        r',?\s*grand_summary_row\.[a-z_.]+\s*=\s*[^,)]+',
        r',?\s*align\s*=\s*"[^"]*"(?=\s*\))',
    ]
    for bad_pat in _INVALID_GT_ARGS:
        code = re.sub(bad_pat, '', code, flags=re.IGNORECASE)

    # Remove tab_style(cells_column_labels()) — causes tidyselect error in gt>=0.10
    code = re.sub(
        r'tab_style\s*\(.*?locations\s*=\s*cells_column_labels\(\).*?\)\s*%>%\s*',
        '', code, flags=re.DOTALL
    )

    # Collapse any newlines inside html("...") strings in tab_header()
    # These cause R to parse `title.` as a bare symbol and fail with
    # "object 'title.' not found".
    def _collapse_html_str(m):
        return m.group(0).replace("\n", " ").replace("\r", "")
    code = re.sub(r'html\("[^"]*"\)', _collapse_html_str, code)
    code = re.sub(r"html\('[^']*'\)", _collapse_html_str, code)

    # Strip trailing dots from unquoted R identifiers that look like `title.`
    # (artifact of a newline being interpolated just before a closing quote)
    code = re.sub(r'\btitle\.\b', 'title', code)
    code = re.sub(r'\bsubtitle\.\b', 'subtitle', code)

    return code

class ShellTLFState(TypedDict):
    shell_text:         str            # raw mock shell pasted/uploaded
    adam_csv:           Optional[str]  # CSV string of ADaM dataset
    parsed_spec:        dict           # extracted by parse_shell node
    generated_code:     str            # R code from generate_code node
    execution_output:   str            # stdout / table HTML
    execution_error:    str            # stderr if failed
    validation_result:  str            # "pass" | "fail: <reason>"
    retry_count:        int
    final_r_code:       str
    final_output:       str            # rendered HTML table or message
    detected_type:      str            # demog|disposition|ae_summary|ae_socpt|lab|vitals|efficacy|listing|figure|llm
    ai_instructions:    str            # extra user instructions for LLM enhancement
    llm_unavailable:    bool           # True when LLM rate-limited — skip fix retries
    _fig_requirements:  dict           # figure validation requirements extracted from shell


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Shell Parser
# ══════════════════════════════════════════════════════════════════════════════
class LLMRateLimitError(RuntimeError):
    """Raised when all LLMs are rate-limited (HTTP 429). Pipeline should skip fix node."""
    pass


def _is_rate_limit(err: Exception) -> bool:
    """Return True if the error is a 429 / quota / rate-limit response."""
    s = str(err).lower()
    return "429" in s or "rate_limit" in s or "quota" in s or "tokens per day" in s


def _gemini_available() -> bool:
    """True only if a non-empty GEMINI_API_KEY is configured."""
    return bool(_get_secret("GEMINI_API_KEY"))


def _groq_available() -> bool:
    """True only if a non-empty GROQ_API_KEY is configured."""
    return bool(_get_secret("GROQ_API_KEY"))


def _call_gemini(prompt: str) -> str:
    """Call Gemini directly. Raises on any error."""
    return _gemini.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    ).text


def _call_groq(prompt: str) -> str:
    """Call Groq directly. Raises on any error."""
    res = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content


def _call_llm(prompt: str) -> str:
    """
    Try Gemini first, then Groq.

    Switching logic:
      - If Gemini key is missing/empty  → skip Gemini, go straight to Groq
      - If Gemini returns auth error     → skip Gemini, go straight to Groq
      - If Gemini is rate-limited (429)  → try Groq
      - If Groq is rate-limited (429)
        AND Gemini also failed with 429  → raise LLMRateLimitError (graceful skip)
      - If Groq is rate-limited (429)
        AND Gemini had auth/other error  → raise LLMRateLimitError (still graceful)
      - Any other double failure         → raise RuntimeError

    LLMRateLimitError is caught by node_fix so the pipeline ends gracefully
    instead of crashing the LangGraph runner.
    """
    gemini_err = None
    groq_err   = None

    # ── Gemini ───────────────────────────────────────────────────────────
    if _gemini_available():
        try:
            return _call_gemini(prompt)
        except Exception as e:
            gemini_err = e
            # Auth errors (401/403) mean key is wrong — no point retrying Gemini
            # Rate-limit on Gemini (429) — fall through to Groq
            # Any other Gemini error     — fall through to Groq
    else:
        gemini_err = RuntimeError("GEMINI_API_KEY not configured")

    # ── Groq fallback ────────────────────────────────────────────────────
    if _groq_available():
        try:
            return _call_groq(prompt)
        except Exception as e:
            groq_err = e
    else:
        groq_err = RuntimeError("GROQ_API_KEY not configured")

    # ── Both failed ───────────────────────────────────────────────────────
    # If Groq is rate-limited, it means Groq quota is exhausted.
    # If Gemini is also rate-limited, both quotas are gone.
    # Either way, raise LLMRateLimitError so node_fix degrades gracefully.
    if _is_rate_limit(groq_err):
        raise LLMRateLimitError(
            f"Groq rate-limited (429). Gemini status: {gemini_err}. "
            f"Auto-fix skipped — pipeline returns best available output."
        )
    if _is_rate_limit(gemini_err):
        raise LLMRateLimitError(
            f"Gemini rate-limited (429). Groq also failed: {groq_err}. "
            f"Auto-fix skipped — pipeline returns best available output."
        )
    raise RuntimeError(
        f"Both LLMs failed.\n"
        f"  Gemini: {gemini_err}\n"
        f"  Groq:   {groq_err}"
    )


def node_parse_shell(state: ShellTLFState) -> ShellTLFState:
    """
    Extract full structured spec from mock shell.
    Handles: epochs/phases, nested rows, categories, treatment groups,
    population flags, dataset hints, statistics.
    """
    prompt = f"""You are an expert clinical SAS/R programmer. Parse this mock shell PRECISELY.

MOCK SHELL:
{state['shell_text']}

Return ONLY valid JSON. No markdown. No explanation. Use this EXACT schema:

{{
  "tlf_number":       "e.g. Table 14.1.1 or empty string",
  "title":            "full title text exactly as shown",
  "output_type":      "Table" or "Listing" or "Figure",
  "population":       "e.g. Safety Population",
  "pop_flag":         "ADaM column: SAFFL ITTFL FASFL PKFL RANDFL ENRLFL — infer from population text",
  "dataset_hint":     "primary ADaM dataset: ADSL ADAE ADVS ADLB ADDS ADEFF etc — infer from context",
  "footnotes":        ["exact footnote text 1", "exact footnote text 2"],
  "treatment_groups": ["Arm A", "Arm B"] — extract from column headers,
  "groupby_var":      "ADaM column for treatment: ARM TRT01P TRT01A TRTP ACTARM — infer from dataset and context",
  "has_sections":     true or false — true if table has EPOCH/PHASE/VISIT sections as row group headers,
  "sections":         [
    {{
      "label":      "Disposition Phase :EPOCH1" — exact section header text,
      "filter_var": "ADaM column for this section e.g. DSPHASE EPOCH VISIT AVISIT",
      "filter_val": "value to filter on e.g. EPOCH1 Baseline",
      "rows": [
        {{
          "label":      "Participants Started /Assigned to Treatment",
          "type":       "header" or "data" or "subrow",
          "indent":     0 or 1 or 2 — indentation level from shell,
          "adam_var":   "ADaM variable or flag e.g. RANDFL DSDECOD DSSCAT",
          "filter_val": "value if this row filters a variable e.g. COMPLETED ADVERSE EVENT",
          "stat_type":  "n" or "n_pct" or "pct" — statistic shown
        }}
      ]
    }}
  ],
  "parameters":       [] — for non-sectioned tables, list parameters here instead,
  "statistics_global": ["n_pct"] — stat types used across the whole table
}}

CRITICAL PARSING RULES:
1. sections: If the shell has repeated blocks separated by phase/epoch/visit headers
   (e.g. "Disposition Phase :EPOCH1", "Disposition Phase :EPOCH2"), extract EACH as a section.
2. rows.indent: Count leading spaces/tabs in shell to determine indent level (0=top, 1=sub, 2=sub-sub).
3. rows.adam_var: Map row labels to ADaM variables:
   - "Participants Started/Assigned" → RANDFL or ENRLFL
   - "Discontinued" → DSDECOD (where DSDECOD != COMPLETED)
   - "Adverse Event" → DSDECOD = ADVERSE EVENT
   - "Death" → DSDECOD = DEATH
   - "Protocol Violation" → DSDECOD = PROTOCOL VIOLATION
   - "Completed" → DSDECOD = COMPLETED
   - "Other" → DSDECOD = OTHER
4. groupby_var: For ADDS dataset → ARM. For ADSL → TRT01P. For ADAE → TRT01A.
5. pop_flag: "Safety" → SAFFL. "Randomized" → RANDFL. "Enrolled" → ENRLFL. "ITT" → ITTFL.
6. If no sections, put row structure in parameters[] instead.
7. has_sections = true whenever you see repeated block structure with phase/epoch/visit headers."""

    raw = _call_llm(prompt)
    raw = re.sub(r'```json|```', '', raw).strip()
    import json
    try:
        spec = json.loads(raw)
    except Exception:
        spec = {
            "tlf_number":"","title":"Table","footnotes":[],
            "population":"","pop_flag":"SAFFL","output_type":"Table",
            "dataset_hint":"","columns":[],"row_stubs":[],
            "statistics":["n","pct"],"groupby":"",
            "has_sections":False,"sections":[],"parameters":[],
            "treatment_groups":[],"groupby_var":"",
            "statistics_global":["n_pct"]
        }

    # Backfill legacy keys for backward compat
    spec.setdefault("has_sections", False)
    spec.setdefault("sections", [])
    spec.setdefault("parameters", [])
    spec.setdefault("treatment_groups", [])
    spec.setdefault("groupby_var", spec.get("groupby","") or "")
    spec.setdefault("statistics_global", ["n_pct"])
    spec.setdefault("pop_flag", spec.get("pop_flag") or "SAFFL")
    spec.setdefault("dataset_hint", "")

    # ── Auto-detect has_sections if LLM missed it ─────────────────────────
    # Look for epoch/phase/period/visit section headers in the raw shell text
    shell_lower = state['shell_text'].lower()
    _section_kws = ["epoch1","epoch2","epoch3","phase 1","phase 2","period 1","period 2",
                    "disposition phase","visit 1","baseline visit","screening"]
    if not spec.get("has_sections") and any(kw in shell_lower for kw in _section_kws):
        spec["has_sections"] = True
        # If sections list is empty, build minimal sections from shell text
        if not spec.get("sections"):
            import re as _re2
            # Extract lines that look like section headers
            epoch_lines = [
                l.strip() for l in state['shell_text'].splitlines()
                if any(kw in l.lower() for kw in ["epoch","phase :","period :","disposition phase"])
                and len(l.strip()) > 3
            ]
            if epoch_lines:
                spec["sections"] = []
                for el in epoch_lines[:4]:
                    m = _re2.search(r'(EPOCH\d+|Phase\s*\d+|Period\s*\d+)', el, _re2.IGNORECASE)
                    fval = m.group(1).upper() if m else el.split()[-1].upper()
                    spec["sections"].append({
                        "label":      el.strip(),
                        "filter_var": "DSPHASE",
                        "filter_val": fval,
                        "rows": [
                            {"label":"Participants Started /Assigned to Treatment","type":"data","indent":0,"adam_var":"RANDFL","filter_val":"","stat_type":"n"},
                            {"label":"Discontinued","type":"data","indent":0,"adam_var":"DSDECOD","filter_val":"DISCONTINUED","stat_type":"n_pct"},
                            {"label":"Adverse Event","type":"data","indent":1,"adam_var":"DSDECOD","filter_val":"ADVERSE EVENT","stat_type":"n_pct"},
                            {"label":"Death","type":"data","indent":1,"adam_var":"DSDECOD","filter_val":"DEATH","stat_type":"n_pct"},
                            {"label":"Protocol Violation","type":"data","indent":1,"adam_var":"DSDECOD","filter_val":"PROTOCOL VIOLATION","stat_type":"n_pct"},
                            {"label":"Other","type":"data","indent":1,"adam_var":"DSDECOD","filter_val":"OTHER","stat_type":"n_pct"},
                            {"label":"Completed","type":"data","indent":0,"adam_var":"DSDECOD","filter_val":"COMPLETED","stat_type":"n_pct"},
                        ]
                    })

    # Legacy row_stubs from sections
    if spec.get("has_sections") and spec.get("sections"):
        spec["row_stubs"] = [
            row.get("label","")
            for sec in spec["sections"]
            for row in sec.get("rows",[])
        ]
    elif not spec.get("row_stubs"):
        spec["row_stubs"] = [p.get("label","") for p in spec.get("parameters",[])]

    spec["groupby"] = spec.get("groupby_var","") or spec.get("groupby","") or "ARM"
    spec["columns"] = spec.get("treatment_groups",[])
    spec["statistics"] = spec.get("statistics_global",["n_pct"])

    state["parsed_spec"] = spec
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Code Generator
# ══════════════════════════════════════════════════════════════════════════════
def _build_demog_r_code(spec: dict, has_adam: bool) -> str:
    """
    Fully spec-driven demographic table.
    Fixes:
    1. groupby from spec — not hardcoded TRT01P
    2. parameters from spec — not fixed to AGE/SEX/RACE/BMIBL
    3. R code uses actual groupby var — no literal TRT01P
    4. dummy data generated from spec parameters — not hardcoded 4 vars
    """
    title     = (spec.get("title") or '"Summary of Demographic and Baseline Characteristics"').strip().replace("\n","").replace("\r","").replace('"', "&quot;")
    pop_flag   = spec.get("pop_flag", "SAFFL") or "SAFFL"
    footnotes  = spec.get("footnotes", [])
    # Resolve groupby to the actual column that will exist in df at R runtime.
    # node_execute prefix GUARANTEES df$TRT01P exists when adam_csv is present
    # (it either reads it directly or aliases from TRT01A/ARM/etc.).
    # When has_adam=False the dummy data uses whatever groupby the spec gives us,
    # falling back to "_trt_" for generic/empty values so the dummy path still works.
    _GENERIC_GROUPBY = {"treatment","trt","group","arm","therapy","drug","intervention",""}
    _raw_groupby = (spec.get("groupby_var") or spec.get("groupby") or "").strip()
    if has_adam:
        # With real data, always use TRT01P — node_execute guarantees it exists.
        groupby = "TRT01P"
    elif _raw_groupby.lower() in _GENERIC_GROUPBY:
        groupby = "_trt_"   # dummy data path — materialised in the dummy df block
    else:
        groupby = _raw_groupby
    parameters = spec.get("parameters", [])

    pop_label_map = {
        "SAFFL":"Safety Population","ITTFL":"ITT Population",
        "FASFL":"Full Analysis Set","PKFL":"PK Population","PPROTFL":"Per-Protocol Population"
    }
    pop_label = pop_label_map.get(pop_flag, spec.get("population","Analysis Population"))

    fn_lines = ""
    for fn in footnotes[:5]:
        fn_lines += f'  tab_source_note(source_note="{fn}") %>%\n'

    # ── Fallback parameters if parser returned nothing ────────────────────
    if not parameters:
        parameters = [
            {"label":"Age (years)",  "type":"continuous",  "adam_var":"AGE",
             "statistics":["n","mean_sd","median","min_max"],"categories":[]},
            {"label":"Sex, n (%)",   "type":"categorical", "adam_var":"SEX",
             "statistics":["n_pct"],"categories":["Male","Female"]},
            {"label":"Race, n (%)",  "type":"categorical", "adam_var":"RACE",
             "statistics":["n_pct"],"categories":[]},
            {"label":"BMI (kg/m²)", "type":"continuous",  "adam_var":"BMIBL",
             "statistics":["n","mean_sd","median","min_max"],"categories":[]},
        ]

    # ── Variable fallback map (try multiple column names per concept) ─────
    var_fallbacks = {
        "AGE":      ["AGE","AGEGR1","AGEGRP","AGE_GRP"],
        "SEX":      ["SEX","GENDER","SEXCD"],
        "RACE":     ["RACE","ETHNIC","RACEGR1","RACECD"],
        "BMIBL":    ["BMIBL","BMI","BMICAT","BMIGRP"],
        "WEIGHTBL": ["WEIGHTBL","WEIGHT","WGTBL"],
        "HEIGHTBL": ["HEIGHTBL","HEIGHT","HGTBL"],
        "DIABFL":   ["DIABFL","DIAB"],
        "SMOKEFL":  ["SMOKEFL","SMOKE","SMOKFL"],
    }

    # ── Build dummy data from spec parameters — not hardcoded ────────────
    if not has_adam:
        dummy_cols = [
            f'  USUBJID  = paste0("S-", 1:(n_per*2))',
            f'  {groupby} = rep(c("Placebo","Drug A"), each=n_per)',
        ]
        for p in parameters:
            av  = p.get("adam_var","")
            pt  = p.get("type","continuous")
            cats= p.get("categories",[])
            if not av:
                continue
            if pt == "continuous":
                dummy_cols.append(f'  {av} = round(c(rnorm(n_per,50,10),rnorm(n_per,53,10)),1)')
            else:
                cat_list = cats if cats else ["Category A","Category B"]
                cats_r   = "c(" + ",".join(f'"{c}"' for c in cat_list) + ")"
                dummy_cols.append(f'  {av} = sample({cats_r}, n_per*2, replace=TRUE)')

        # Add pop flags
        for flg in ["SAFFL","ITTFL","FASFL"]:
            dummy_cols.append(f'  {flg} = "Y"')

        dummy_data = f"""
set.seed(42)
n_per <- 10
df <- data.frame(
{chr(44)+chr(10)}.join(dummy_cols),
  stringsAsFactors=FALSE
)
"""
    else:
        dummy_data = ""

    pop_filter = f"""if ("{pop_flag}" %in% names(df)) df <- df %>% filter({pop_flag}=="Y")"""

    ind_r = 'strrep(intToUtf8(160), 6)'

    # ── Build R blocks for each parameter from spec ───────────────────────
    def make_cont_block(var_col, param_name, stats, safe_name):
        fallbacks    = var_fallbacks.get(var_col, [var_col])
        fallbacks_r  = "c(" + ",".join(f'"{v}"' for v in fallbacks) + ")"
        stat_rows    = []
        total_rows   = []
        if "n" in stats or "mean_sd" in stats:
            stat_rows.append(f'df %>% group_by({groupby}) %>% summarise(val=as.character(n()), .groups="drop") %>% mutate(Statistic=paste0(.ind,"n"))')
            total_rows.append(f'data.frame(val=as.character(nrow(df)), Statistic=paste0(.ind,"n"))')
        if "mean_sd" in stats:
            stat_rows.append(f'df %>% group_by({groupby}) %>% summarise(val=sprintf("%.1f (%.1f)",mean(.cv.,na.rm=T),sd(.cv.,na.rm=T)),.groups="drop") %>% mutate(Statistic=paste0(.ind,"Mean (SD)"))')
            total_rows.append(f'data.frame(val=sprintf("%.1f (%.1f)",mean(df$.cv.,na.rm=T),sd(df$.cv.,na.rm=T)),Statistic=paste0(.ind,"Mean (SD)"))')
        if "median" in stats:
            stat_rows.append(f'df %>% group_by({groupby}) %>% summarise(val=sprintf("%.1f",median(.cv.,na.rm=T)),.groups="drop") %>% mutate(Statistic=paste0(.ind,"Median"))')
            total_rows.append(f'data.frame(val=sprintf("%.1f",median(df$.cv.,na.rm=T)),Statistic=paste0(.ind,"Median"))')
        if "min_max" in stats:
            stat_rows.append(f'df %>% group_by({groupby}) %>% summarise(val=sprintf("%g, %g",min(.cv.,na.rm=T),max(.cv.,na.rm=T)),.groups="drop") %>% mutate(Statistic=paste0(.ind,"Min, Max"))')
            total_rows.append(f'data.frame(val=sprintf("%g, %g",min(df$.cv.,na.rm=T),max(df$.cv.,na.rm=T)),Statistic=paste0(.ind,"Min, Max"))')

        bind_trt   = "  bind_rows(\n    " + ",\n    ".join(stat_rows)   + "\n  )"
        bind_total = "  bind_rows(\n    " + ",\n    ".join(total_rows) + "\n  )"

        return f"""
# --- {param_name} ---
.col_try <- {fallbacks_r}
.col_nm  <- .col_try[.col_try %in% names(df)][1]
if (!is.na(.col_nm)) {{
  .ind <- {ind_r}
  df$.cv. <- as.numeric(df[[.col_nm]])
  {safe_name}_by_trt <-
{bind_trt} %>%
    pivot_wider(id_cols=Statistic, names_from={groupby}, values_from=val)
  {safe_name}_total <-
{bind_total}
  {safe_name}_by_trt$Total     <- {safe_name}_total$val
  {safe_name}_by_trt$Parameter <- "{param_name}"
  df$.cv. <- NULL
}}
"""

    def make_cat_block(var_col, param_name, cats, safe_name):
        fallbacks   = var_fallbacks.get(var_col, [var_col])
        fallbacks_r = "c(" + ",".join(f'"{v}"' for v in fallbacks) + ")"
        if cats:
            cats_code = "expected_cats <- c(" + ",".join(f'"{c}"' for c in cats) + ")"
        else:
            cats_code = "expected_cats <- sort(unique(df$.cv.))"

        return f"""
# --- {param_name} ---
.col_try <- {fallbacks_r}
.col_nm  <- .col_try[.col_try %in% names(df)][1]
if (!is.na(.col_nm)) {{
  .ind  <- {ind_r}
  df$.cv. <- df[[.col_nm]]
  {cats_code}
  .n_denom       <- df %>% group_by({groupby}) %>% summarise(N=n_distinct(USUBJID),.groups="drop")
  .n_denom_total <- n_distinct(df$USUBJID)
  {safe_name}_rows <- lapply(expected_cats, function(cat) {{
    tv <- df %>% group_by({groupby}) %>%
      summarise(nc=sum(.cv.==cat,na.rm=T),.groups="drop") %>%
      left_join(.n_denom, by="{groupby}") %>%
      mutate(val=sprintf("%d (%.1f%%)",nc,100*nc/pmax(N,1))) %>%
      select({groupby},val) %>%
      pivot_wider(names_from={groupby},values_from=val,values_fill="0 (0.0%)")
    nc_tot        <- sum(df$.cv.==cat,na.rm=T)
    tv$Total      <- sprintf("%d (%.1f%%)",nc_tot,100*nc_tot/max(.n_denom_total,1))
    tv$Statistic  <- paste0(.ind,cat)
    tv$Parameter  <- "{param_name}"
    tv
  }})
  {safe_name}_by_trt <- bind_rows({safe_name}_rows)
  df$.cv. <- NULL
}}
"""

    # ── Build blocks in spec order ────────────────────────────────────────
    param_blocks = []
    param_names  = []
    bind_vars    = []

    for i, param in enumerate(parameters):
        label    = param.get("label","")
        ptype    = param.get("type","continuous")
        adam_var = param.get("adam_var","")
        stats    = param.get("statistics",["n","mean_sd","median","min_max"])
        cats     = param.get("categories",[])
        if not label or not adam_var:
            continue

        safe_name = re.sub(r'[^A-Za-z0-9]','_', f"p{i}_{adam_var}")
        param_names.append(label)
        bind_vars.append(f'if (exists("{safe_name}_by_trt")) {safe_name}_by_trt else NULL')

        if ptype == "continuous":
            param_blocks.append(make_cont_block(adam_var, label, stats, safe_name))
        else:
            param_blocks.append(make_cat_block(adam_var, label, cats, safe_name))

    # ── If all parameters were skipped (missing adam_var), fall back to defaults ──
    if not bind_vars:
        parameters = [
            {"label":"Age (years)",  "type":"continuous",  "adam_var":"AGE",
             "statistics":["n","mean_sd","median","min_max"],"categories":[]},
            {"label":"Sex, n (%)",   "type":"categorical", "adam_var":"SEX",
             "statistics":["n_pct"],"categories":["Male","Female"]},
            {"label":"Race, n (%)",  "type":"categorical", "adam_var":"RACE",
             "statistics":["n_pct"],"categories":[]},
            {"label":"BMI (kg/m²)", "type":"continuous",  "adam_var":"BMIBL",
             "statistics":["n","mean_sd","median","min_max"],"categories":[]},
        ]
        param_blocks, param_names, bind_vars = [], [], []
        for i, param in enumerate(parameters):
            label    = param.get("label","")
            ptype    = param.get("type","continuous")
            adam_var = param.get("adam_var","")
            stats    = param.get("statistics",["n","mean_sd","median","min_max"])
            cats     = param.get("categories",[])
            safe_name = re.sub(r'[^A-Za-z0-9]','_', f"p{i}_{adam_var}")
            param_names.append(label)
            bind_vars.append(f'if (exists("{safe_name}_by_trt")) {safe_name}_by_trt else NULL')
            if ptype == "continuous":
                param_blocks.append(make_cont_block(adam_var, label, stats, safe_name))
            else:
                param_blocks.append(make_cat_block(adam_var, label, cats, safe_name))

    bind_call = "tbl_data <- bind_rows(\n  " + ",\n  ".join(bind_vars) + "\n)"
    param_order_r = "c(" + ",".join(f'"{p}"' for p in param_names) + ")"
    fn_list = ", ".join(f'"{fn}"' for fn in footnotes[:5]) if footnotes else ""

    # Build R footnote vector safely
    if footnotes:
        fn_r_vec = "c(" + ", ".join(f'"{fn.replace(chr(34), chr(39))}"' for fn in footnotes[:5]) + ")"
    else:
        fn_r_vec = "character(0)"

    code = f"""suppressPackageStartupMessages({{
  library(dplyr)
  library(tidyr)
  library(gt)
}})
{dummy_data}
{pop_filter}

# Ensure TRT01P exists (node_execute prefix already handles this, belt-and-suspenders)
if (!"TRT01P" %in% names(df)) {{
  .tc <- Filter(function(x) x %in% names(df), c("TRT01A","TRTP","ARM","ACTARM"))
  df[["TRT01P"]] <- if (length(.tc)) df[[.tc[1]]] else "Total"
}}

{"".join(param_blocks)}

{bind_call}

# Guard: only select columns that actually exist
.sel_cols <- intersect(c("Parameter","Statistic"), names(tbl_data))
if (length(.sel_cols) == 2) {{
  tbl_data <- tbl_data %>% select(Parameter, Statistic, everything())
}} else {{
  stop(paste("tbl_data missing required columns. Got:", paste(names(tbl_data), collapse=", ")))
}}
tbl_data$Statistic <- gsub("^[ \\t\\r\\n]+|[ \\t\\r\\n]+$","",tbl_data$Statistic)

# Dynamic N headers — html() gives proper <br> line break
trts      <- sort(unique(df${groupby}))
n_per_trt <- sapply(trts, function(t) n_distinct(df$USUBJID[df${groupby}==t]))
n_total   <- n_distinct(df$USUBJID)
col_label_names <- c(trts, "Total")
col_label_vals  <- c(
  lapply(seq_along(trts), function(i) html(paste0("<b>",trts[i],"</b><br>(N=",n_per_trt[i],")"))),
  list(html(paste0("<b>Total</b><br>(N=",n_total,")")))
)
col_labels <- setNames(col_label_vals, col_label_names)

# Enforce shell parameter order (guard: skip if levels vector is empty)
.param_levels <- {param_order_r}
if (length(.param_levels) > 0) {{
  tbl_data$Parameter <- factor(tbl_data$Parameter, levels=.param_levels)
  tbl_data <- tbl_data[order(tbl_data$Parameter),]
  tbl_data$Parameter <- as.character(tbl_data$Parameter)
}}

# footnotes vector
fn_text <- {fn_r_vec}

tbl <- gt(tbl_data, groupname_col="Parameter") %>%
  tab_header(
    title = html("<b>{title}</b>"),
    subtitle = html('<div style="text-align:left;font-size:12px;font-weight:normal;">{pop_label}</div>')
  ) %>%
  cols_label(.list=col_labels) %>%
  cols_label(Statistic="") %>%
  cols_hide("Parameter") %>%
  tab_style(style=cell_text(weight="bold"), locations=cells_row_groups()) %>%
  tab_style(style=cell_text(indent=px(20)), locations=cells_body(columns="Statistic")) %>%
  cols_align(align="left", columns="Statistic") %>%
  tab_options(
    table.width=pct(100),
    row_group.background.color="#f5f5f5",
    heading.align="left",
    column_labels.font.weight="bold"
  )

# Add lettered footnotes a. b. c.  (plain letter + period, no superscript)
if (length(fn_text) > 0) {{
  for (i in seq_along(fn_text)) {{
    tbl <- tbl %>% tab_source_note(
      source_note = html(paste0(letters[i], ". ", fn_text[i]))
    )
  }}
  tbl <- tbl %>% tab_style(
    style = cell_text(size = px(11), font = "Arial, sans-serif"),
    locations = cells_source_notes()
  )
}}

cat(as_raw_html(tbl))
"""
    return code


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: AE Summary (incidence by treatment)
# ══════════════════════════════════════════════════════════════════════════════
def _build_ae_summary_r_code(spec: dict, has_adam: bool) -> str:
    title     = (spec.get("title") or '"Summary of Adverse Events"').strip().replace("\n","").replace("\r","").replace('"', "&quot;")
    pop_flag  = spec.get("pop_flag", "SAFFL") or "SAFFL"
    footnotes = spec.get("footnotes", [])

    fn_lines = ""
    for fn in footnotes[:5]:
        fn_lines += f'  tab_footnote(footnote="{fn}", locations=cells_title()) %>%\n'

    dummy = "" if has_adam else """
set.seed(42)
# Build one record per subject-AE combination explicitly
df <- data.frame(
  USUBJID  = c(paste0("P",1:10), paste0("P",1:10), paste0("P",1:8),
               paste0("D",1:10), paste0("D",1:10), paste0("D",1:8)),
  TRT01P   = c(rep("Placebo",28), rep("Drug A",28)),
  TRTEMFL  = "Y",
  AEBODSYS = sample(c("Gastrointestinal disorders","Nervous system disorders","Skin disorders"), 56, replace=TRUE),
  AEDECOD  = sample(c("Nausea","Headache","Rash","Vomiting","Dizziness"), 56, replace=TRUE),
  AESER    = sample(c("Y","N"), 56, replace=TRUE, prob=c(0.2,0.8)),
  SAFFL    = "Y",
  stringsAsFactors=FALSE
)
"""
    pop_filter = f"""if ("{pop_flag}" %in% names(df)) df <- df[df${pop_flag}=="Y", ]"""

    return f"""{dummy}
{pop_filter}

# Denominators — named vector so [[t]] lookup works with spaces in names
trts  <- sort(unique(df$TRT01P))
n_trt <- setNames(
  sapply(trts, function(t) length(unique(df$USUBJID[df$TRT01P==t]))),
  trts
)
n_all <- length(unique(df$USUBJID))

# TEAE subset — guard against NA and whitespace in TRTEMFL
if ("TRTEMFL" %in% names(df)) {{
  ae <- df[!is.na(df$TRTEMFL) & trimws(df$TRTEMFL) == "Y", ]
}} else {{
  ae <- df
}}

# Core function: count unique subjects per treatment
make_row <- function(subdata, label) {{
  trt_vals <- sapply(trts, function(t) {{
    n_subj <- length(unique(subdata$USUBJID[subdata$TRT01P == t]))
    denom  <- n_trt[[t]]
    if (is.null(denom) || is.na(denom) || denom == 0) return("0 (0.0%)")
    sprintf("%d (%.1f%%)", n_subj, 100 * n_subj / denom)
  }})
  tot_n      <- length(unique(subdata$USUBJID))
  row        <- as.data.frame(t(trt_vals), stringsAsFactors=FALSE)
  names(row) <- trts
  row$Total    <- sprintf("%d (%.1f%%)", tot_n, 100 * tot_n / max(n_all, 1))
  row$Category <- label
  row
}}

# Build each summary row separately then rbind
r1 <- make_row(ae, "Any TEAE")
r2 <- make_row(ae[!is.na(ae$AESER) & ae$AESER == "Y", ], "Any Serious TEAE")
r3 <- make_row(ae[grepl("Gastro",  ae$AEBODSYS, ignore.case=TRUE), ], "Gastrointestinal disorders")
r4 <- make_row(ae[grepl("Nervous", ae$AEBODSYS, ignore.case=TRUE), ], "Nervous system disorders")
r5 <- make_row(ae[grepl("Skin",    ae$AEBODSYS, ignore.case=TRUE), ], "Skin disorders")
rows <- rbind(r1, r2, r3, r4, r5)

# Reorder: Category | trts alphabetically | Total
col_order <- c("Category", sort(trts), "Total")
col_order <- col_order[col_order %in% names(rows)]
rows      <- rows[, col_order, drop=FALSE]

tbl <- gt(rows) %>%
  tab_header(title="{title}") %>%
  {fn_lines}  cols_label(Category="Adverse Event Category") %>%
  tab_style(
    style=cell_text(weight="bold"),
    locations=cells_body(columns="Category", rows=1)
  ) %>%
  tab_options(table.width=pct(100))

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: AE by SOC and PT (nested)
# ══════════════════════════════════════════════════════════════════════════════
def _build_ae_socpt_r_code(spec: dict, has_adam: bool) -> str:
    title     = (spec.get("title") or '"Adverse Events by System Organ Class and Preferred Term"').strip().replace("\n","").replace("\r","").replace('"', "&quot;")
    pop_flag  = spec.get("pop_flag", "SAFFL") or "SAFFL"
    footnotes = spec.get("footnotes", [])

    fn_lines = ""
    for fn in footnotes[:5]:
        fn_lines += f'  tab_footnote(footnote="{fn}", locations=cells_title()) %>%\n'

    dummy = "" if has_adam else """
set.seed(42)
subj_ids <- paste0("S", 1:15)
df <- data.frame(
  USUBJID  = sample(subj_ids, 40, replace=TRUE),
  TRT01P   = sample(c("Placebo","Drug A"), 40, replace=TRUE),
  TRTEMFL  = "Y",
  AEBODSYS = sample(c("Gastrointestinal disorders","Nervous system disorders","Skin disorders"), 40, replace=TRUE),
  AEDECOD  = sample(c("Nausea","Vomiting","Headache","Dizziness","Rash","Pruritus"), 40, replace=TRUE),
  SAFFL    = "Y",
  stringsAsFactors=FALSE
)
"""
    pop_filter = f"""if ("{pop_flag}" %in% names(df)) df <- df[df${pop_flag}=="Y", ]"""

    return f"""{dummy}
{pop_filter}

ae    <- df[!is.na(df$TRTEMFL) & df$TRTEMFL=="Y", ]
trts  <- sort(unique(df$TRT01P))
n_trt <- sapply(trts, function(t) length(unique(df$USUBJID[df$TRT01P==t])))
n_all <- length(unique(df$USUBJID))

fmt_n <- function(n, denom) sprintf("%d (%.1f%%)", n, 100*n/max(denom,1))

# Count unique subjects per treatment for a subset
count_row <- function(subdata, label, indent=FALSE) {{
  trt_vals <- sapply(trts, function(t) {{
    fmt_n(length(unique(subdata$USUBJID[subdata$TRT01P==t])), n_trt[t])
  }})
  tot <- length(unique(subdata$USUBJID))
  row <- as.data.frame(t(trt_vals), stringsAsFactors=FALSE)
  row$Total <- fmt_n(tot, n_all)
  row$Term  <- if (indent) paste0("    ", label) else label
  row$Level <- if (indent) "PT" else "SOC"
  row
}}

# Build SOC then PT rows interleaved
all_soc  <- sort(unique(ae$AEBODSYS))
tbl_rows <- do.call(rbind, lapply(all_soc, function(soc) {{
  soc_data <- ae[ae$AEBODSYS==soc, ]
  soc_row  <- count_row(soc_data, soc, indent=FALSE)
  pts      <- sort(unique(soc_data$AEDECOD))
  pt_rows  <- do.call(rbind, lapply(pts, function(pt) {{
    count_row(soc_data[soc_data$AEDECOD==pt, ], pt, indent=TRUE)
  }}))
  rbind(soc_row, pt_rows)
}}))

# Column order
col_order  <- c("Term", sort(trts), "Total", "Level")
col_order  <- col_order[col_order %in% names(tbl_rows)]
tbl_rows   <- tbl_rows[, col_order]

tbl <- gt(tbl_rows) %>%
  tab_header(title="{title}") %>%
  {fn_lines}  cols_label(Term="System Organ Class / Preferred Term") %>%
  cols_hide("Level") %>%
  tab_style(
    style = cell_text(weight="bold"),
    locations = cells_body(columns="Term", rows=Level=="SOC")
  ) %>%
  tab_options(table.width=pct(100))

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: Lab Values / Vital Signs Summary
# ══════════════════════════════════════════════════════════════════════════════
def _build_lab_r_code(spec: dict, has_adam: bool) -> str:
    title     = (spec.get("title") or '"Summary of Laboratory Values"').strip().replace("\n","").replace("\r","").replace('"', "&quot;")
    pop_flag  = spec.get("pop_flag", "SAFFL") or "SAFFL"
    footnotes = spec.get("footnotes", [])
    is_vitals = any(k in spec.get("title","").lower() for k in ["vital","weight","height","pulse","blood pressure"])

    fn_lines = ""
    for fn in footnotes[:5]:
        fn_lines += f'  tab_footnote(footnote="{fn}", locations=cells_title()) %>%\n'

    dummy = "" if has_adam else f"""
set.seed(42)
n <- 40
params <- if ({str(is_vitals).upper()} == TRUE) c("Systolic BP","Diastolic BP","Pulse","Weight") else c("ALT","AST","Creatinine","Hemoglobin")
visits  <- c("Baseline","Week 4","Week 8","Week 12")
df <- expand.grid(
  USUBJID  = paste0("S", 1:5),
  TRT01P   = c("Placebo","Drug A"),
  PARAM    = params,
  AVISIT   = visits,
  stringsAsFactors=FALSE
)
df$AVAL  <- round(rnorm(nrow(df), 50, 10), 1)
df$BASE  <- round(rnorm(nrow(df), 50, 8),  1)
df$CHG   <- round(df$AVAL - df$BASE, 1)
df$SAFFL <- "Y"
df$AVISITN <- match(df$AVISIT, visits)
"""
    pop_filter = f"""if ("{pop_flag}" %in% names(df)) df <- df %>% filter({pop_flag} == "Y")"""

    return f"""{dummy}
{pop_filter}

# Detect column names flexibly
param_col <- if ("PARAM" %in% names(df)) "PARAM" else if ("PARAMCD" %in% names(df)) "PARAMCD" else names(df)[1]
visit_col <- if ("AVISIT" %in% names(df)) "AVISIT" else if ("VISIT" %in% names(df)) "VISIT" else names(df)[2]
val_col   <- if ("AVAL"   %in% names(df)) "AVAL"   else if ("VALUE"  %in% names(df)) "VALUE"  else names(df)[3]

params_list <- sort(unique(df[[param_col]]))
visits_list <- unique(df[[visit_col]])
if ("AVISITN" %in% names(df)) visits_list <- visits_list[order(match(visits_list, df[[visit_col]][order(df$AVISITN)]))]

make_cont_row <- function(data, vc, stat_label, stat_fn) {{
  vals <- data[[vc]]
  data %>% group_by(TRT01P) %>%
    summarise(val=stat_fn(vals[match(seq_along(vals), which(data$TRT01P==TRT01P[1]))]), .groups="drop") %>%
    pivot_wider(names_from=TRT01P, values_from=val) %>%
    mutate(Statistic=stat_label)
}}

# Simpler: compute stats directly without dynamic column issues
tbl_list <- lapply(params_list, function(p) {{
  lapply(visits_list, function(v) {{
    sub <- df[df[[param_col]]==p & df[[visit_col]]==v, ]
    if (nrow(sub)==0) return(NULL)
    avals <- sub[[val_col]]
    trts  <- sort(unique(sub$TRT01P))
    
    make_stat <- function(stat_label, fn) {{
      vals <- sapply(trts, function(t) fn(avals[sub$TRT01P==t]))
      row  <- as.data.frame(t(vals))
      names(row) <- trts
      row$Statistic <- stat_label
      row
    }}
    
    rows <- bind_rows(
      make_stat(paste0(strrep(intToUtf8(160),6),"n"),         function(x) as.character(sum(!is.na(x)))),
      make_stat(paste0(strrep(intToUtf8(160),6),"Mean (SD)"), function(x) sprintf("%.1f (%.1f)", mean(x,na.rm=TRUE), sd(x,na.rm=TRUE))),
      make_stat(paste0(strrep(intToUtf8(160),6),"Median"),    function(x) sprintf("%.1f", median(x,na.rm=TRUE))),
      make_stat(paste0(strrep(intToUtf8(160),6),"Min, Max"),  function(x) sprintf("%g, %g", min(x,na.rm=TRUE), max(x,na.rm=TRUE)))
    )
    rows$Total     <- c(
      as.character(sum(!is.na(avals))),
      sprintf("%.1f (%.1f)", mean(avals,na.rm=TRUE), sd(avals,na.rm=TRUE)),
      sprintf("%.1f", median(avals,na.rm=TRUE)),
      sprintf("%g, %g", min(avals,na.rm=TRUE), max(avals,na.rm=TRUE))
    )
    rows$Parameter <- p
    rows$Visit     <- v
    rows
  }})
}})

tbl_data <- bind_rows(unlist(tbl_list, recursive=FALSE)) %>%
  select(Parameter, Visit, Statistic, everything())

# Trim only ASCII whitespace — preserve nbsp indent (\u00a0) used for sub-row indent
tbl_data$Statistic <- gsub("^[ \t\r\n]+|[ \t\r\n]+$", "", tbl_data$Statistic)

tbl <- gt(tbl_data, groupname_col="Parameter") %>%
  tab_header(title="{title}") %>%
  {fn_lines}  cols_label(Visit="Visit", Statistic="Statistic") %>%
  tab_style(style=cell_text(weight="bold"), locations=cells_row_groups()) %>%
  cols_align(align="left", columns="Statistic") %>%
  tab_options(table.width=pct(100), row_group.background.color="#f5f5f5")

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: Efficacy / Primary Endpoint
# ══════════════════════════════════════════════════════════════════════════════
def _build_efficacy_r_code(spec: dict, has_adam: bool) -> str:
    title     = (spec.get("title") or '"Summary of Efficacy"').strip().replace("\n","").replace("\r","").replace('"', "&quot;")
    pop_flag  = spec.get("pop_flag", "ITTFL") or "ITTFL"
    footnotes = spec.get("footnotes", [])

    fn_lines = ""
    for fn in footnotes[:5]:
        fn_lines += f'  tab_footnote(footnote="{fn}", locations=cells_title()) %>%\n'

    dummy = "" if has_adam else """
set.seed(42)
n_per <- 15
df <- data.frame(
  USUBJID = paste0("S", 1:(n_per*2)),
  TRT01P  = rep(c("Placebo","Drug A"), each=n_per),
  AVAL    = c(rnorm(n_per, 2.1, 0.8), rnorm(n_per, 3.4, 0.9)),
  BASE    = rnorm(n_per*2, 2.0, 0.5),
  ITTFL   = "Y",
  AVALCAT1= sample(c("Responder","Non-Responder"), n_per*2, replace=TRUE, prob=c(0.4,0.6)),
  stringsAsFactors=FALSE
)
df$CHG  <- df$AVAL - df$BASE
df$PCHG <- 100 * df$CHG / df$BASE
"""
    pop_filter = f"""if ("{pop_flag}" %in% names(df)) df <- df %>% filter({pop_flag} == "Y")"""

    return f"""{dummy}
{pop_filter}

n_trt   <- df %>% select(USUBJID, TRT01P) %>% distinct() %>% count(TRT01P, name="N_total")
n_all   <- n_distinct(df$USUBJID)

# Continuous endpoint summary
val_col <- if ("AVAL" %in% names(df)) "AVAL" else names(df)[3]
chg_col <- if ("CHG"  %in% names(df)) "CHG"  else NULL

make_row <- function(data, col, label) {{
  ind <- strrep(intToUtf8(160), 6)
  r <- data %>% group_by(TRT01P) %>%
    summarise(
      n_val   = as.character(sum(!is.na(.data[[col]]))),
      mean_sd = sprintf("%.2f (%.2f)", mean(.data[[col]],na.rm=TRUE), sd(.data[[col]],na.rm=TRUE)),
      med_val = sprintf("%.2f", median(.data[[col]],na.rm=TRUE)),
      rng_val = sprintf("%.2f, %.2f", min(.data[[col]],na.rm=TRUE), max(.data[[col]],na.rm=TRUE)),
      .groups = "drop"
    )
  bind_rows(
    r %>% select(TRT01P, val=n_val)   %>% pivot_wider(names_from=TRT01P, values_from=val) %>% mutate(Statistic=paste0(ind,"n"), Parameter=label),
    r %>% select(TRT01P, val=mean_sd) %>% pivot_wider(names_from=TRT01P, values_from=val) %>% mutate(Statistic=paste0(ind,"Mean (SD)"), Parameter=label),
    r %>% select(TRT01P, val=med_val) %>% pivot_wider(names_from=TRT01P, values_from=val) %>% mutate(Statistic=paste0(ind,"Median"), Parameter=label),
    r %>% select(TRT01P, val=rng_val) %>% pivot_wider(names_from=TRT01P, values_from=val) %>% mutate(Statistic=paste0(ind,"Min, Max"), Parameter=label)
  )
}}

rows <- make_row(df, val_col, "Primary Endpoint")
if (!is.null(chg_col) && chg_col %in% names(df)) {{
  rows <- bind_rows(rows, make_row(df, chg_col, "Change from Baseline"))
}}

# Responder analysis if categorical column exists
if ("AVALCAT1" %in% names(df)) {{
  ind <- strrep(intToUtf8(160), 6)
  resp <- df %>% group_by(TRT01P) %>%
    summarise(val=sprintf("%d (%.1f%%)",
      sum(AVALCAT1=="Responder",na.rm=TRUE),
      100*mean(AVALCAT1=="Responder",na.rm=TRUE)), .groups="drop") %>%
    pivot_wider(names_from=TRT01P, values_from=val) %>%
    mutate(Statistic=paste0(ind,"Responders, n (%)"), Parameter="Response")
  rows <- bind_rows(rows, resp)
}}

# Total column — strip only ASCII whitespace (not nbsp) before matching
tot_aval <- df[[val_col]]
rows$Total <- NA_character_
for (i in seq_len(nrow(rows))) {{
  stat <- gsub("^[ \t\r\n]+|[ \t\r\n]+$", "", rows$Statistic[i])
  if (stat=="n")              rows$Total[i] <- as.character(sum(!is.na(tot_aval)))
  else if (stat=="Mean (SD)") rows$Total[i] <- sprintf("%.2f (%.2f)", mean(tot_aval,na.rm=TRUE), sd(tot_aval,na.rm=TRUE))
  else if (stat=="Median")    rows$Total[i] <- sprintf("%.2f", median(tot_aval,na.rm=TRUE))
  else if (stat=="Min, Max")  rows$Total[i] <- sprintf("%.2f, %.2f", min(tot_aval,na.rm=TRUE), max(tot_aval,na.rm=TRUE))
  else if (grepl("Responder", stat)) rows$Total[i] <- sprintf("%d (%.1f%%)", sum(df$AVALCAT1=="Responder",na.rm=TRUE), 100*mean(df$AVALCAT1=="Responder",na.rm=TRUE))
}}

# Trim only ASCII whitespace — preserve nbsp indent
rows$Statistic <- gsub("^[ \t\r\n]+|[ \t\r\n]+$", "", rows$Statistic)

tbl <- gt(rows, groupname_col="Parameter") %>%
  tab_header(title="{title}") %>%
  {fn_lines}  cols_label(Statistic="Statistic") %>%
  cols_hide("Parameter") %>%
  tab_style(style=cell_text(weight="bold"), locations=cells_row_groups()) %>%
  cols_align(align="left", columns="Statistic") %>%
  tab_options(table.width=pct(100), row_group.background.color="#f5f5f5")

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: Generic Clinical Listing
# ══════════════════════════════════════════════════════════════════════════════
def _build_listing_r_code(spec: dict, has_adam: bool) -> str:
    title     = (spec.get("title") or '"Clinical Listing"').strip().replace("\n","").replace("\r","").replace('"', "&quot;")
    pop_flag  = spec.get("pop_flag", "SAFFL") or "SAFFL"
    footnotes = spec.get("footnotes", [])
    columns   = spec.get("columns", [])

    fn_lines = ""
    for fn in footnotes[:5]:
        fn_lines += f'  tab_footnote(footnote="{fn}", locations=cells_title()) %>%\n'

    col_select = ""
    if columns:
        cols_r = ", ".join(f'"{c}"' for c in columns[:10])
        col_select = f"""
# Select requested columns if they exist
req_cols <- c({cols_r})
avail    <- req_cols[req_cols %in% names(df)]
if (length(avail) > 0) df <- df %>% select(all_of(avail))
"""

    dummy = "" if has_adam else f"""
set.seed(42)
df <- data.frame(
  USUBJID  = paste0("SUBJ-", 1:10),
  TRT01P   = rep(c("Placebo","Drug A"), 5),
  AGE      = sample(30:70, 10),
  SEX      = sample(c("M","F"), 10, replace=TRUE),
  AEDECOD  = sample(c("Headache","Nausea","Rash","Fatigue"), 10, replace=TRUE),
  AESTDTC  = format(Sys.Date() - sample(1:60, 10), "%Y-%m-%d"),
  AEENDTC  = format(Sys.Date() - sample(0:30, 10), "%Y-%m-%d"),
  AESEV    = sample(c("MILD","MODERATE","SEVERE"), 10, replace=TRUE),
  SAFFL    = "Y",
  stringsAsFactors=FALSE
)
"""
    pop_filter = f"""if ("{pop_flag}" %in% names(df)) df <- df %>% filter({pop_flag} == "Y")"""

    return f"""{dummy}
{pop_filter}
{col_select}

# Sort by USUBJID if present
if ("USUBJID" %in% names(df)) df <- df %>% arrange(USUBJID)

tbl <- gt(df) %>%
  tab_header(title="{title}") %>%
  {fn_lines}  tab_options(table.width=pct(100))

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE ROUTER — detect table type and dispatch
# ══════════════════════════════════════════════════════════════════════════════

def _build_disposition_r_code(spec: dict, has_adam: bool) -> str:
    """
    Disposition table — hierarchical by EPOCH/phase with n (%) per treatment arm.
    Uses structured sections from parser when available, falls back to row_stubs parsing.
    """
    import re as _re

    title     = (spec.get("title") or "Disposition of Participants").strip().replace("\n","").replace("\r","").replace('"',"&quot;")
    pop_flag  = (spec.get("pop_flag") or "RANDFL").strip()
    footnotes = spec.get("footnotes", [])
    row_stubs = spec.get("row_stubs", [])

    # ── Use structured sections from new parser if available ─────────────
    if spec.get("has_sections") and spec.get("sections"):
        return _build_sectioned_r_code(spec, has_adam)

    DISCON_KW = ["adverse event","death","protocol violation","other",
                 "lack of efficacy","withdrawal","sponsor decision",
                 "lost to follow","non-compliance"]

    # Parse phase sections from row_stubs (legacy path)
    epochs = []
    cur_label, cur_val, cur_reasons = None, None, []
    for stub in row_stubs:
        s  = stub.strip()
        sl = s.lower()
        if "epoch" in sl or ("phase" in sl and ":" in s) or "period" in sl:
            if cur_label:
                epochs.append({"label": cur_label, "val": cur_val, "reasons": cur_reasons})
            m = _re.search(r'[:\-]\s*(\S+)\s*$', s)
            cur_val     = m.group(1).strip() if m else s.split()[-1]
            cur_label   = s
            cur_reasons = []
        elif cur_label and any(k in sl for k in DISCON_KW) and "started" not in sl and "assigned" not in sl:
            cur_reasons.append(s)
    if cur_label:
        epochs.append({"label": cur_label, "val": cur_val, "reasons": cur_reasons})

    if not epochs:
        epochs = [{"label": "Overall Disposition", "val": "", "reasons":
                   [s.strip() for s in row_stubs
                    if any(k in s.lower() for k in DISCON_KW)
                    and "completed" not in s.lower()]}]

    all_reasons = []
    for e in epochs:
        for r in e["reasons"]:
            if r not in all_reasons:
                all_reasons.append(r)
    if not all_reasons:
        all_reasons = ["Adverse Event","Death","Protocol Violation","Other"]

    # Footnote R code
    fn_r = ""
    if footnotes:
        fn_items = ", ".join(f'"{f}"' for f in footnotes[:5])
        fn_r = f"""
fn_text <- c({fn_items})
if (length(fn_text) > 0) {{
  for (i in seq_along(fn_text)) {{
    ltr <- letters[i]
    tbl <- tbl %>% tab_source_note(
      source_note = html(paste0(
        '<span style="font-family:Arial,Helvetica,sans-serif;font-size:11px;">',
        '<span style="font-size:8px;vertical-align:super;line-height:0;">',
        ltr, '</span>. ', fn_text[i], '</span>'
      ))
    )
  }}
}}"""

    pop_filter_r = f'if ("{pop_flag}" %in% names(df)) df <- df %>% filter({pop_flag} == "Y")' if pop_flag else ""

    # Dummy data
    ep_vals_r  = ", ".join(f'"{e["val"]}"' for e in epochs if e["val"]) or '"EPOCH1","EPOCH2"'
    reasons_r  = ", ".join(f'"{r}"' for r in all_reasons[:6])
    dummy_data = "" if has_adam else f"""
set.seed(42)
n_subj <- 24
df <- data.frame(
  USUBJID = paste0("SUBJ", sprintf("%03d", 1:n_subj)),
  ARM     = rep(c("Arm A","Arm B"), each = n_subj/2),
  DSCAT   = "DISPOSITION EVENT",
  DSDECOD = sample(c({reasons_r},"COMPLETED"), n_subj, replace=TRUE),
  DSPHASE = sample(c({ep_vals_r}), n_subj, replace=TRUE),
  RANDFL  = "Y", ENRLFL = "Y", SAFFL = "Y",
  stringsAsFactors = FALSE
)
"""

    # Per-epoch R blocks
    ep_blocks = []
    for idx, ep in enumerate(epochs):
        lbl     = ep["label"]
        val     = ep["val"]
        reasons = [r for r in (ep["reasons"] if ep["reasons"] else all_reasons)
                   if "completed" not in r.lower()]
        reas_r  = ", ".join(f'"{r}"' for r in reasons)
        sid     = f"ep{idx}"
        ep_filt = f'df_ep <- df %>% filter(DSPHASE == "{val}")' if val else "df_ep <- df"

        blk = f"""
# ── {lbl} ──
{ep_filt}
.denom_trt  <- df_ep %>% group_by(`_trt_`) %>% summarise(N=n_distinct(USUBJID),.groups="drop")
.denom_tot  <- n_distinct(df_ep$USUBJID)
{sid}_rows  <- list()

# Started
.sf <- if ("RANDFL" %in% names(df_ep)) "RANDFL" else if ("ENRLFL" %in% names(df_ep)) "ENRLFL" else NA
.ns_trt <- df_ep %>%
  filter(if(!is.na(.sf)) .data[[.sf]]=="Y" else TRUE) %>%
  group_by(`_trt_`) %>% summarise(n=n_distinct(USUBJID),.groups="drop")
.ns_tot <- df_ep %>%
  filter(if(!is.na(.sf)) .data[[.sf]]=="Y" else TRUE) %>%
  summarise(n=n_distinct(USUBJID)) %>% pull(n)
.r <- .denom_trt %>% left_join(.ns_trt,by="_trt_") %>%
  mutate(n=coalesce(n,0L),val=as.character(n)) %>%
  select(`_trt_`,val) %>%
  pivot_wider(names_from=`_trt_`,values_from=val,values_fill="0")
.r$Total <- as.character(.ns_tot)
.r$Statistic <- paste0(.iB,"Participants Started /Assigned to Treatment")
.r$Parameter <- "{lbl}"
{sid}_rows[["started"]] <- .r

# Discontinued (header)
.disc <- df_ep %>% filter(toupper(DSDECOD)!="COMPLETED")
.nd_trt <- .disc %>% group_by(`_trt_`) %>% summarise(n=n_distinct(USUBJID),.groups="drop")
.nd_tot <- n_distinct(.disc$USUBJID)
.r <- .denom_trt %>% left_join(.nd_trt,by="_trt_") %>%
  mutate(n=coalesce(n,0L),
         val=sprintf("%d (%.1f%%)",n,100*n/pmax(N,1))) %>%
  select(`_trt_`,val) %>%
  pivot_wider(names_from=`_trt_`,values_from=val,values_fill="0 (0.0%)")
.r$Total <- sprintf("%d (%.1f%%)",.nd_tot,100*.nd_tot/max(.denom_tot,1))
.r$Statistic <- paste0(.iB,"Discontinued")
.r$Parameter <- "{lbl}"
{sid}_rows[["discon"]] <- .r

# Discontinuation reasons
for (.rsn in c({reas_r})) {{
  .rn_trt <- df_ep %>% filter(toupper(DSDECOD)==toupper(.rsn)) %>%
    group_by(`_trt_`) %>% summarise(n=n_distinct(USUBJID),.groups="drop")
  .rn_tot <- df_ep %>% filter(toupper(DSDECOD)==toupper(.rsn)) %>%
    summarise(n=n_distinct(USUBJID)) %>% pull(n)
  .r <- .denom_trt %>% left_join(.rn_trt,by="_trt_") %>%
    mutate(n=coalesce(n,0L),
           val=sprintf("%d (%.1f%%)",n,100*n/pmax(N,1))) %>%
    select(`_trt_`,val) %>%
    pivot_wider(names_from=`_trt_`,values_from=val,values_fill="0 (0.0%)")
  .r$Total <- sprintf("%d (%.1f%%)",.rn_tot,100*.rn_tot/max(.denom_tot,1))
  .r$Statistic <- paste0(.iR,.rsn)
  .r$Parameter <- "{lbl}"
  {sid}_rows[[paste0("r_",.rsn)]] <- .r
}}

# Completed
.comp <- df_ep %>% filter(toupper(DSDECOD)=="COMPLETED")
.nc_trt <- .comp %>% group_by(`_trt_`) %>% summarise(n=n_distinct(USUBJID),.groups="drop")
.nc_tot <- n_distinct(.comp$USUBJID)
.r <- .denom_trt %>% left_join(.nc_trt,by="_trt_") %>%
  mutate(n=coalesce(n,0L),
         val=sprintf("%d (%.1f%%)",n,100*n/pmax(N,1))) %>%
  select(`_trt_`,val) %>%
  pivot_wider(names_from=`_trt_`,values_from=val,values_fill="0 (0.0%)")
.r$Total <- sprintf("%d (%.1f%%)",.nc_tot,100*.nc_tot/max(.denom_tot,1))
.r$Statistic <- paste0(.iB,"Completed")
.r$Parameter <- "{lbl}"
{sid}_rows[["completed"]] <- .r

{sid}_tbl <- bind_rows({sid}_rows)
"""
        ep_blocks.append(blk)

    bind_parts = "\n".join(
        f'  if (exists("ep{i}_tbl")) ep{i}_tbl else NULL,' for i in range(len(epochs))
    )
    param_order_r = ", ".join(f'"{e["label"]}"' for e in epochs)

    code = f"""suppressPackageStartupMessages({{
  library(dplyr)
  library(tidyr)
  library(gt)
}})
{dummy_data}
{pop_filter_r}

# Auto-detect treatment column
.trt_cands <- c("ARM","TRT01P","TRT01A","TRTP","ACTARM")
.trt_col   <- Filter(function(x) x %in% names(df), .trt_cands)
.trt_col   <- if (length(.trt_col)) .trt_col[1] else names(df)[2]
df[["_trt_"]] <- df[[.trt_col]]

# Filter to disposition events
if ("DSCAT" %in% names(df)) df <- df %>% filter(DSCAT == "DISPOSITION EVENT")

# Indent helpers
.iB <- strrep(intToUtf8(160), 2)   # 2 nbsp — bold rows
.iR <- strrep(intToUtf8(160), 6)   # 6 nbsp — reason sub-rows

{"".join(ep_blocks)}

# Combine epochs
tbl_data <- bind_rows(
{bind_parts}
  NULL
)
tbl_data <- tbl_data %>% select(Parameter, Statistic, everything())
tbl_data$Statistic <- gsub("^[ \\t\\r\\n]+|[ \\t\\r\\n]+$","",tbl_data$Statistic)

param_order <- c({param_order_r})
tbl_data$Parameter <- factor(tbl_data$Parameter, levels=param_order)
tbl_data <- tbl_data[order(tbl_data$Parameter),]
tbl_data$Parameter <- as.character(tbl_data$Parameter)

# N-counts in column headers
.trts <- sort(unique(df[["_trt_"]]))
.n_per <- sapply(.trts, function(t) n_distinct(df$USUBJID[df[["_trt_"]]==t]))
.n_tot <- n_distinct(df$USUBJID)
.col_labs <- setNames(
  c(lapply(seq_along(.trts), function(i) html(paste0("<b>",.trts[i],"</b><br>(N=",.n_per[i],")"))),
    list(html(paste0("<b>Total</b><br>(N=",.n_tot,")")))),
  c(.trts,"Total")
)

tbl <- gt(tbl_data, groupname_col="Parameter") %>%
  tab_header(
    title    = html("<b>{title}</b>"),
    subtitle = html('<div style="text-align:left;font-size:12px;">Number (%) of Participants</div>')
  ) %>%
  cols_label(.list=.col_labs) %>%
  cols_label(Statistic="") %>%
  cols_hide("Parameter") %>%
  tab_style(style=cell_text(weight="bold"), locations=cells_row_groups()) %>%
  cols_align(align="left", columns="Statistic") %>%
  tab_options(
    table.width=pct(100),
    row_group.background.color="#f0f0f0",
    heading.align="left",
    column_labels.font.weight="bold",
    source_notes.font.size=px(11),
    source_notes.padding=px(4)
  )
{fn_r}
cat(as_raw_html(tbl))
"""
    return code



def _build_sectioned_r_code(spec: dict, has_adam: bool) -> str:
    """
    Generates R code for tables with EPOCH/PHASE/VISIT sections.
    Each section filters on a variable+value and shows the same row structure.
    Handles: Disposition, Subject Accountability, etc.
    """
    title     = (spec.get("title") or "Table").strip().replace('"',"'")
    pop_flag  = spec.get("pop_flag","SAFFL") or "SAFFL"
    footnotes = spec.get("footnotes",[])
    # node_execute guarantees TRT01P exists when adam_csv is present
    groupby   = "TRT01P" if has_adam else (spec.get("groupby_var") or spec.get("groupby") or "ARM")
    sections  = spec.get("sections",[])
    dataset   = spec.get("dataset_hint","ADDS").upper()

    pop_label_map = {
        "SAFFL":"Safety Population","ITTFL":"ITT Population",
        "RANDFL":"Randomized Population","ENRLFL":"Enrolled Population",
        "FASFL":"Full Analysis Set",
    }
    pop_label = pop_label_map.get(pop_flag, spec.get("population","Analysis Population"))

    # Build footnotes
    fn_r_vec = "character(0)"
    if footnotes:
        fn_r_vec = "c(" + ", ".join(f'"{fn.replace(chr(34),chr(39))}"' for fn in footnotes[:5]) + ")"

    # Dummy data from sections
    dummy = "" if has_adam else f"""
set.seed(42)
n_per <- 12
arms  <- c("Arm A","Arm B")
phases <- c("EPOCH1","EPOCH2")
reasons <- c("ADVERSE EVENT","DEATH","PROTOCOL VIOLATION","OTHER","COMPLETED")
df <- do.call(rbind, lapply(arms, function(arm) {{
  do.call(rbind, lapply(phases, function(ph) {{
    data.frame(
      USUBJID  = paste0(arm,"_",ph,"_",1:n_per),
      ARM      = arm,
      DSPHASE  = ph,
      EPOCH    = ph,
      DSDECOD  = sample(reasons, n_per, replace=TRUE, prob=c(0.2,0.2,0.2,0.2,0.2)),
      DSCAT    = "DISPOSITION EVENT",
      RANDFL   = "Y",
      ENRLFL   = "Y",
      SAFFL    = "Y",
      stringsAsFactors=FALSE
    )
  }}))
}}))
"""
    pop_filter = f'if ("{pop_flag}" %in% names(df)) df <- df[df${pop_flag}=="Y",]'

    # Build section R blocks
    section_blocks = []
    for sec in sections:
        sec_label  = sec.get("label","Section").replace('"',"'")
        filter_var = sec.get("filter_var","DSPHASE") or "DSPHASE"
        filter_val = sec.get("filter_val","") or ""
        rows       = sec.get("rows",[])

        row_lines = []
        for row in rows:
            rlabel    = row.get("label","").replace('"',"'")
            rtype     = row.get("type","data")
            indent    = row.get("indent",0)
            adam_var  = row.get("adam_var","") or ""
            fval      = row.get("filter_val","") or ""
            stat_type = row.get("stat_type","n_pct")

            if rtype == "header":
                row_lines.append(f"""
  # --- header row: {rlabel} ---
  h_row <- data.frame(Category="{rlabel}", stringsAsFactors=FALSE)
  for (tr in trts) h_row[[tr]] <- ""
  h_row$Total <- ""
  h_row$is_bold <- TRUE
  h_row$indent  <- 0
  all_rows <- rbind(all_rows, h_row)""")
            else:
                # Build filter condition for the row
                fval_upper = fval.upper()
                if adam_var in ["RANDFL","ENRLFL","SAFFL","RANDFL"]:
                    # Flag variable — count subjects with flag=Y
                    row_filter = f'sec_df[!is.na(sec_df${adam_var}) & sec_df${adam_var}=="Y", ]'
                elif fval_upper == "DISCONTINUED":
                    # Discontinued = all who did NOT complete
                    row_filter = f'sec_df[!is.na(sec_df${adam_var}) & toupper(sec_df${adam_var}) != "COMPLETED", ]'
                elif fval:
                    row_filter = f'sec_df[!is.na(sec_df${adam_var}) & toupper(sec_df${adam_var})=="{fval_upper}", ]'
                else:
                    row_filter = 'sec_df'

                if stat_type == "n":
                    fmt_trt   = 'as.character(length(unique(rd$USUBJID)))'
                    fmt_total = f'as.character(length(unique(({row_filter})$USUBJID)))'
                    tot_pct   = ""
                else:
                    fmt_trt   = 'sprintf("%d (%.1f%%)", length(unique(rd$USUBJID)), 100*length(unique(rd$USUBJID))/max(n_trt[[tr]],1))'
                    fmt_total = f'sprintf("%d (%.1f%%)", length(unique(rd_all$USUBJID)), 100*length(unique(rd_all$USUBJID))/max(n_all,1))'
                    tot_pct   = f'\n  rd_all <- {row_filter}'

                row_lines.append(f"""
  # --- {rlabel} ---
  d_row <- data.frame(Category="{rlabel}", stringsAsFactors=FALSE)
  for (tr in trts) {{
    rd <- {row_filter}
    rd <- rd[rd${groupby}==tr, ]
    d_row[[tr]] <- {fmt_trt}
  }}{tot_pct}
  d_row$Total   <- {fmt_total}
  d_row$is_bold <- FALSE
  d_row$indent  <- {indent}L
  all_rows <- rbind(all_rows, d_row)""")

        filter_cond = f'sec_df <- df[df${filter_var}=="{filter_val}",]' if filter_val else 'sec_df <- df'

        section_blocks.append(f"""
# ══ Section: {sec_label} ══
{filter_cond}
n_trt <- setNames(
  sapply(trts, function(t) length(unique(sec_df$USUBJID[sec_df${groupby}==t]))),
  trts
)
n_all <- length(unique(sec_df$USUBJID))
sec_header <- data.frame(Category="{sec_label}", stringsAsFactors=FALSE)
for (tr in trts) sec_header[[tr]] <- ""
sec_header$Total   <- ""
sec_header$is_bold <- TRUE
sec_header$indent  <- -1
all_rows <- rbind(all_rows, sec_header)
{"".join(row_lines)}
""")

    return f"""{dummy}
{pop_filter}

# Treatment groups and denominators
trts  <- sort(unique(df${groupby}))
n_per_trt <- setNames(
  sapply(trts, function(t) length(unique(df$USUBJID[df${groupby}==t]))),
  trts
)
n_all_total <- length(unique(df$USUBJID))

# Column headers with N
col_label_names <- c(trts,"Total")
col_label_vals  <- c(
  lapply(seq_along(trts), function(i)
    html(paste0("<b>",trts[i],"</b><br>(N=",n_per_trt[[trts[i]]],")"))),
  list(html(paste0("<b>Total</b><br>(N=",n_all_total,")")))
)
col_labels <- setNames(col_label_vals, col_label_names)

# Build all rows
all_rows <- data.frame(Category=character(), stringsAsFactors=FALSE)
for (tr in trts) all_rows[[tr]] <- character()
all_rows$Total   <- character()
all_rows$is_bold <- logical()
all_rows$indent  <- integer()

{"".join(section_blocks)}

# Build gt table
tbl <- gt(all_rows) %>%
  tab_header(
    title    = html("<b>{title}</b>"),
    subtitle = html('<div style="text-align:left;font-size:12px;">{pop_label}</div>')
  ) %>%
  cols_label(.list=col_labels) %>%
  cols_label(Category="") %>%
  cols_hide(c("is_bold","indent")) %>%
  tab_style(
    style     = cell_text(weight="bold"),
    locations = cells_body(columns="Category", rows=is_bold==TRUE)
  ) %>%
  tab_style(
    style     = cell_text(indent=px(20)),
    locations = cells_body(columns="Category", rows=indent==1L)
  ) %>%
  tab_style(
    style     = cell_text(indent=px(40)),
    locations = cells_body(columns="Category", rows=indent==2L)
  ) %>%
  tab_style(
    style     = cell_text(color="#666666"),
    locations = cells_body(columns="Category", rows=indent>=1L)
  ) %>%
  cols_align(align="left",  columns="Category") %>%
  cols_align(align="right", columns=c(trts,"Total")) %>%
  tab_options(
    table.width=pct(100),
    heading.align="left",
    column_labels.font.weight="bold",
    source_notes.font.size=px(11),
    source_notes.padding=px(4)
  )

# Footnotes
fn_text <- {fn_r_vec}
if (length(fn_text)>0) {{
  for (i in seq_along(fn_text)) {{
    tbl <- tbl %>% tab_source_note(
      source_note=html(paste0('<sup>',letters[i],'</sup> ',fn_text[i]))
    )
  }}
}}

cat(as_raw_html(tbl))
"""


def _detect_table_type(spec: dict) -> str:
    """
    Returns one of: demog | ae_summary | ae_socpt | lab | vitals |
                    efficacy | listing | figure | disposition | llm
    """
    output_type = spec.get("output_type", "Table")
    if output_type == "Figure":
        return "figure"

    # Sectioned tables (epoch/phase/visit blocks) — detected from parser
    if spec.get("has_sections") and spec.get("sections"):
        return "disposition"

    title     = (spec.get("title") or "").lower().strip()
    row_stubs = [r.lower() for r in spec.get("row_stubs", [])]
    dataset   = (spec.get("dataset_hint") or "").upper()
    all_text  = title + " " + " ".join(row_stubs)

    # Title-based detection — title is primary signal
    if any(k in title for k in ["discontinu","disposition","accountability"]):
        return "disposition"
    if any(k in title for k in ["demographic","baseline characteristic"]):
        return "demog"
    if any(k in title for k in ["system organ class","preferred term","by soc","by body system"]):
        return "ae_socpt"
    if any(k in title for k in ["adverse event","teae","treatment-emergent"]):
        return "ae_summary"
    if any(k in title for k in ["vital sign","blood pressure","pulse"]):
        return "vitals"
    if any(k in title for k in ["laborator","lab value","chemistry","haematol","hematol"]):
        return "lab"
    if any(k in title for k in ["efficacy","primary endpoint","response rate","change from baseline"]):
        return "efficacy"
    if output_type == "Listing" or any(k in title for k in ["listing","patient data","subject data"]):
        return "listing"

    # Dataset-based detection
    if dataset == "ADDS":
        return "disposition"
    if dataset == "ADSL":
        return "demog"
    if dataset == "ADAE":
        return "ae_summary"
    if dataset == "ADLB":
        return "lab"
    if dataset == "ADVS":
        return "vitals"

    # Row stubs — only if title gave no signal
    stub_text = " ".join(row_stubs)
    if any(k in stub_text for k in ["epoch","dsphase","disposition phase","participants started"]):
        return "disposition"
    if any(k in stub_text for k in ["mean (sd)","median","min, max"]) and \
       any(k in stub_text for k in ["age","bmi","weight"]):
        return "demog"

    return "llm"


def node_generate_code(state: ShellTLFState) -> ShellTLFState:
    """
    Hybrid approach — routes to Python template or LLM based on detected table type.
    Templates: demog | ae_summary | ae_socpt | lab | vitals | efficacy | listing
    LLM fallback: everything else + Figures
    """
    spec        = state["parsed_spec"]
    output_type = spec.get("output_type", "Table")
    has_adam    = bool(state.get("adam_csv"))

    adam_hint = ""
    if has_adam:
        try:
            df_preview = pd.read_csv(io.StringIO(state["adam_csv"])).head(5).to_string()
            adam_hint  = f"\nADaM dataset preview (first 5 rows):\n{df_preview}\n"
        except Exception:
            adam_hint = ""

    # Figures → dedicated figure prompt using execute_graph pattern
    if output_type == "Figure":
        table_type = "figure"
    else:
        table_type = _detect_table_type(spec)

    # Dispatch to template
    if table_type == "figure":
        title      = spec.get("title", "Clinical Figure")
        groupby    = spec.get("groupby_var") or spec.get("groupby") or "TRT01P"
        footnotes  = spec.get("footnotes", [])
        fn_caption = footnotes[0] if footnotes else ""
        shell_text = state.get("shell_text", "")

        # ── Detect figure type from title ─────────────────────────────────
        title_lower = title.lower()
        shell_lower = shell_text.lower()
        combined    = title_lower + " " + shell_lower

        if any(k in combined for k in ["kaplan","km","survival","time to event"]):
            fig_type = "km"
        elif any(k in combined for k in ["forest","odds ratio","hazard ratio","risk ratio"]):
            fig_type = "forest"
        elif any(k in combined for k in ["waterfall","spider","tumor"]):
            fig_type = "waterfall"
        elif any(k in combined for k in ["bar","barplot","frequency","incidence"]):
            fig_type = "bar"
        elif any(k in combined for k in ["box","boxplot","distribution"]):
            fig_type = "box"
        elif any(k in combined for k in ["scatter","correlation"]):
            fig_type = "scatter"
        elif any(k in combined for k in ["line","mean over time","profile","trend","change from baseline"]):
            fig_type = "line"
        else:
            fig_type = "line"  # clinical default

        # ── Extract visit order from shell ────────────────────────────────
        import re as _re2
        visit_matches = _re2.findall(
            r'(?:Baseline|Screening|Day\s*\d+|Week\s*\d+|Month\s*\d+|Cycle\s*\d+|End of (?:Study|Treatment))',
            shell_text, _re2.IGNORECASE
        )
        # Deduplicate preserving order
        seen = set()
        visit_order = [v for v in visit_matches if not (v.lower() in seen or seen.add(v.lower()))]
        visit_order_r = "c(" + ", ".join(f'"{v}"' for v in visit_order) + ")" if visit_order else 'NULL'

        # ── Detect requirements from shell ────────────────────────────────
        needs_se        = any(k in shell_lower for k in ["± se","±se","mean ± se","se error","standard error","error bar"])
        needs_sd        = any(k in shell_lower for k in ["± sd","±sd","mean ± sd","sd error"])
        needs_ci        = any(k in shell_lower for k in ["95% ci","confidence interval","ci"])
        needs_ref_zero  = any(k in shell_lower for k in ["y=0","reference line","ref line","horizontal line at 0","dashed line at 0"])
        needs_line      = fig_type == "line" or any(k in shell_lower for k in ["connect","line","trend"])
        error_type      = "SE" if needs_se else ("SD" if needs_sd else ("CI" if needs_ci else "SE" if fig_type=="line" else ""))

        # ── Build visit order injection — runs BEFORE LLM code ───────────
        visit_inject = ""
        if visit_order:
            levels_r = "c(" + ", ".join(f'"{v}"' for v in visit_order) + ")"
            visit_inject = f"""
# Enforce clinical visit order — only if visit column exists with matching values
tryCatch({{
  .visit_cols <- intersect(c("VISIT","AVISIT","Visit","visit"), names(df))
  if (length(.visit_cols) > 0) {{
    .vc           <- .visit_cols[1]
    .visit_levels <- {levels_r}
    .existing     <- intersect(.visit_levels, unique(as.character(df[[.vc]])))
    if (length(.existing) > 0) {{
      df[[.vc]] <- factor(df[[.vc]], levels=.visit_levels)
    }}
  }}
}}, error=function(e) invisible(NULL))
"""

        # ── Build error bar summary — runs BEFORE LLM code ───────────────
        # error_inject: only for line figures with a time/visit axis
        error_inject = ""
        if error_type and fig_type == "line":
            error_inject = f"""
# Pre-compute summary stats for error bars ({error_type})
.x_col <- if ("VISIT" %in% names(df)) "VISIT" else if ("AVISIT" %in% names(df)) "AVISIT" else NULL
.y_col <- if ("CHG"   %in% names(df)) "CHG"   else if ("AVAL"   %in% names(df)) "AVAL"   else if ("MEAN" %in% names(df)) "MEAN" else names(df)[3]
.g_col <- if ("{groupby}" %in% names(df)) "{groupby}" else if ("TRT01P" %in% names(df)) "TRT01P" else if ("ARM" %in% names(df)) "ARM" else names(df)[1]

if (!is.null(.x_col) && .x_col != .g_col) {{
  df_sum <- df %>%
    group_by(.data[[.x_col]], .data[[.g_col]]) %>%
    summarise(
      MEAN = mean(.data[[.y_col]], na.rm=TRUE),
      N    = sum(!is.na(.data[[.y_col]])),
      SE   = if(sum(!is.na(.data[[.y_col]])) > 1)
               sd(.data[[.y_col]], na.rm=TRUE) / sqrt(sum(!is.na(.data[[.y_col]])))
             else 0,
      .groups="drop"
    )
  names(df_sum)[names(df_sum)==.x_col] <- "VISIT"
  names(df_sum)[names(df_sum)==.g_col] <- "{groupby}"
  df_sum$LOWER <- df_sum$MEAN - df_sum$SE
  df_sum$UPPER <- df_sum$MEAN + df_sum$SE
  df_sum$SE[is.na(df_sum$SE)] <- 0
  df_sum$LOWER[is.na(df_sum$LOWER)] <- df_sum$MEAN[is.na(df_sum$LOWER)]
  df_sum$UPPER[is.na(df_sum$UPPER)] <- df_sum$MEAN[is.na(df_sum$UPPER)]
  if (is.factor(df[[.x_col]])) df_sum$VISIT <- factor(df_sum$VISIT, levels=levels(df[[.x_col]]))
}} else {{
  df_sum <- df
}}
"""

        prompt = f"""You are an expert clinical R programmer. Generate ggplot2 code for this clinical figure.

TITLE: {title}
FIGURE TYPE: {fig_type}
TREATMENT COLUMN: {groupby}

REQUIREMENTS DETECTED FROM SHELL:
- needs_line_geom:  {needs_line}
- needs_error_bars: {bool(error_type)} ({error_type})
- needs_ref_zero:   {needs_ref_zero}
- visit_order:      {visit_order if visit_order else 'auto-detect'}

PRE-COMPUTED DATA AVAILABLE:
- df      = raw data (already loaded, VISIT already factored in correct order)
- df_sum  = summary with columns: VISIT, {groupby}, MEAN, SE, LOWER, UPPER {"(already computed)" if error_type else "(not computed — use df directly)"}

MANDATORY RULES — all must be in output:
1. Do NOT add library() or ggsave() or read.csv().
2. Use theme_classic() + theme(legend.position="top right") for clinical look.
3. MUST use df_sum as data source (not df) when error bars are needed.
4. MUST include geom_line(data=df_sum, aes(x=VISIT, y=MEAN, group={groupby}, color={groupby}), linewidth=1)
5. MUST include geom_point(data=df_sum, aes(x=VISIT, y=MEAN, color={groupby}), size=2)
6. {"MUST include geom_errorbar(data=df_sum, aes(x=VISIT, ymin=LOWER, ymax=UPPER, color="+groupby+"), width=0.2)" if error_type else "No error bars needed."}
7. {"MUST include geom_hline(yintercept=0, linetype='dashed', color='gray40', linewidth=0.8)" if needs_ref_zero else "No reference line needed."}
8. MUST include labs(title="{title}", x="Visit", y="Mean Change from Baseline", color="Treatment")
9. MUST include theme(legend.position=c(0.85, 0.95)) for top-right legend.
10. DO NOT sort visits — VISIT is already a factor in correct order.
11. NEVER write df$VISIT <- or df[["VISIT"]] <- or any assignment to a VISIT column. Visit ordering is handled externally.
12. Last line MUST assign plot to p: p <- ggplot(...) + ...
13. Return ONLY R code. No markdown. No explanations.

Generate complete ggplot2 code now:"""

        raw = _call_llm(prompt)
        raw = re.sub(r'```[rR]?\n?', '', raw)
        raw = re.sub(r'```', '', raw).strip()
        raw = re.sub(r'\+?\s*ggsave\s*\([^)]*\)', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'^\s*library\s*\([^)]+\)\s*$', '', raw, flags=re.MULTILINE).strip()

        # ── For line figures: build guaranteed skeleton, discard ALL LLM ggplot code ─
        if fig_type == "line":
            # Extract only color scale from LLM output — discard everything else
            color_line = ""
            for line in raw.split('\n'):
                ls = line.strip()
                if ls.startswith('scale_color_manual') or ls.startswith('scale_color_brewer'):
                    color_line = ls.rstrip(' +,')
                    break

            # Default clinical color scale
            if not color_line:
                color_line = 'scale_color_manual(values=c("Placebo"="#E45252","Drug A 10mg"="#3A8FC8","Drug A"="#3A8FC8","Active Drug"="#2CA02C"))'

            labs_line  = f'labs(title="{title}", x="Visit", y="Mean Change from Baseline", color="Treatment")'
            theme_line = 'theme(legend.position="right", legend.background=element_rect(fill="white", color="gray80", linewidth=0.3))'

            errorbar_geom = ""
            if bool(error_type):
                errorbar_geom = f'  geom_errorbar(aes(ymin=LOWER, ymax=UPPER), width=0.15, linewidth=0.6, fill=NA) +'

            # Always add reference line for line figures (clinical standard)
            hline_geom = '  geom_hline(yintercept=0, linetype="dashed", color="gray50", linewidth=0.7) +'

            # Completely replace raw with Python-built skeleton — no LLM ggplot code
            raw = f"""p <- ggplot(data=df_sum, aes(x=VISIT, y=MEAN, color={groupby}, group={groupby})) +
  geom_hline(yintercept=0, linetype="dashed", color="gray50", linewidth=0.7) +
  geom_line(linewidth=1.1) +
  geom_point(size=2.5, show.legend=FALSE) +
{errorbar_geom}
  {color_line} +
  {labs_line} +
  theme_classic() +
  {theme_line}
"""
        else:
            # Non-line figures: strip LLM visit factor assignments and wrong geoms
            raw = re.sub(r'df\$VISIT\s*<-\s*factor\([^)]*\)',     '', raw)
            raw = re.sub(r'df\[\["VISIT"\]\]\s*<-\s*factor\([^)]*\)', '', raw)
            raw = re.sub(r'df\[\[.x_col\]\]\s*<-\s*factor\([^)]*\)', '', raw)
            raw = re.sub(r'\+?\s*geom_bar\s*\([^)]*\)',  '', raw, flags=re.DOTALL)
            raw = re.sub(r'\+?\s*geom_col\s*\([^)]*\)',  '', raw, flags=re.DOTALL)
            raw = re.sub(r'\+?\s*geom_area\s*\([^)]*\)', '', raw, flags=re.DOTALL)

            if needs_ref_zero and "geom_hline" not in raw:
                raw += '\np <- p + geom_hline(yintercept=0, linetype="dashed", color="gray40", linewidth=0.8)'

        # Always define df_sum — error_inject only sets it for line figures
        visit_inject = visit_inject + "\nif (!exists('df_sum')) df_sum <- df\n"

        # Prepend visit ordering + error bar summary BEFORE everything
        raw = visit_inject + "\n" + error_inject + "\n" + raw.strip()

        # Ensure last line assigns to p
        lines = [l for l in raw.strip().split('\n') if l.strip()]
        if lines and not any(lines[-1].strip().startswith(x) for x in ['p <-','p=']):
            raw += '\np <- last_plot()'

        # Store requirements for validator
        state["_fig_requirements"] = {
            "needs_line":       needs_line,
            "needs_error_bars": bool(error_type),
            "needs_ref_zero":   needs_ref_zero,
            "visit_order":      visit_order,
            "fig_type":         fig_type,
            "title":            title,
        }

    # Dispatch to template
    if table_type == "figure":
        pass  # raw already set above
    elif table_type == "demog":
        raw = _build_demog_r_code(spec, has_adam)
    elif table_type == "ae_summary":
        raw = _build_ae_summary_r_code(spec, has_adam)
    elif table_type == "ae_socpt":
        raw = _build_ae_socpt_r_code(spec, has_adam)
    elif table_type in ("lab", "vitals"):
        raw = _build_lab_r_code(spec, has_adam)
    elif table_type == "disposition":
        raw = _build_disposition_r_code(spec, has_adam)
    elif table_type == "efficacy":
        raw = _build_efficacy_r_code(spec, has_adam)
    elif table_type == "listing":
        raw = _build_listing_r_code(spec, has_adam)
    else:
        # LLM fallback for Figures and custom tables
        prompt = f"""You are an expert clinical R programmer generating production-ready TLF code.

SPEC:
{spec}
{adam_hint}

RULES:
1. Use gt package for Tables, ggplot2 for Figures.
2. If ADaM data is provided, read from df (already loaded). If not, create realistic dummy data.
3. Do NOT add library() calls — dplyr, tidyr, gt, ggplot2 are prepended automatically.
4. TABLE STRUCTURE: Treatment groups are ALWAYS columns. Statistics are ALWAYS rows.
5. Use bind_rows() + pivot_wider(names_from=TRT01P) — never mutate(col=c(...)) on grouped data.
6. ALL data.frame column names must be non-empty strings.
7. VALID gt functions only: gt(), tab_header(), tab_footnote(), cols_label(), tab_row_group(),
   tab_style(), cell_text(), cells_row_groups(), cells_title(), cells_column_labels(),
   cells_body(), as_raw_html().
   NEVER USE: column_labels(), set_column_labels(), tab_column_label().
8. For Tables: last line MUST be: cat(as_raw_html(tbl))
9. For Figures: last line MUST be: print(p)  — no ggsave.
10. Return ONLY R code. No markdown fences. No explanations.

Generate complete R code now:"""
        raw = _call_llm(prompt)
        raw = re.sub(r'```[rR]?\n?', '', raw)
        raw = re.sub(r'```', '', raw).strip()
        raw = re.sub(r'\+?\s*ggsave\s*\([^)]*\)', '', raw, flags=re.DOTALL).strip()

    state["generated_code"] = raw
    state["detected_type"]  = table_type

    # ── AI instructions enhancement pass ──────────────────────────────────
    # If user provided extra instructions AND a template was used (not LLM),
    # run a quick LLM enhancement pass to apply the customisations.
    ai_instr = state.get("ai_instructions", "").strip()
    if ai_instr and table_type != "llm":
        enhance_prompt = f"""You are an R clinical TLF code editor. Apply ONLY the requested changes below.

EXISTING R CODE:
{raw}

USER INSTRUCTIONS:
{ai_instr}

RULES:
- Touch ONLY what the instructions ask. Preserve all other logic exactly.
- Do NOT add library(), read.csv, or ggsave — these are handled externally.
- Valid gt functions: gt(), tab_header(), tab_footnote(), cols_label(), tab_row_group(),
  tab_style(), cell_text(), cells_row_groups(), cells_title(), cells_column_labels(),
  cells_body(), as_raw_html().
- Last line must remain: cat(as_raw_html(tbl))
- Return ONLY complete R code. No markdown fences. No explanations.
"""
        try:
            enhanced = _call_llm(enhance_prompt)
            enhanced = re.sub(r'```[rR]?\n?', '', enhanced)
            enhanced = re.sub(r'```', '', enhanced).strip()
            if enhanced:
                state["generated_code"] = enhanced
        except Exception:
            pass  # silently keep template code if enhancement fails

    # ── Post-generation sanitiser ────────────────────────────────────────
    state["generated_code"] = _sanitise_r_code(state["generated_code"])
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Executor
# ══════════════════════════════════════════════════════════════════════════════
def node_execute(state: ShellTLFState) -> ShellTLFState:
    """Run R code, capture stdout as execution_output.
    For Figures: uses execute_graph() from graph_builder (tested, reliable).
    For Tables/Listings: uses subprocess with gt prefix.
    """
    spec        = state["parsed_spec"]
    output_type = spec.get("output_type", "Table")
    detected    = state.get("detected_type", "")

    # ── FIGURE: self-contained subprocess — guaranteed ggplot2 loading ───
    if output_type == "Figure" or detected == "figure":
        with tempfile.TemporaryDirectory() as d:
            script_path = os.path.join(d, "figure_script.R")
            plot_path   = os.path.join(d, "figure.png")
            inp_path    = os.path.join(d, "data.csv")

            # Build / save data
            if state.get("adam_csv"):
                df = pd.read_csv(io.StringIO(state["adam_csv"]))
                # Drop rows where all visit columns are NA
                visit_cols = [c for c in df.columns if c in ["VISIT","AVISIT","Visit"]]
                for vc in visit_cols:
                    df = df[df[vc].notna()]
            else:
                # Get visit order from figure requirements
                fig_reqs   = state.get("_fig_requirements", {})
                visit_order = fig_reqs.get("visit_order", ["Baseline","Week 4","Week 8","Week 12"])
                if not visit_order:
                    visit_order = ["Baseline","Week 4","Week 8","Week 12"]
                groupby = state.get("parsed_spec",{}).get("groupby_var","TRT01P") or "TRT01P"
                trts    = ["Placebo","Drug A 10mg"]

                import numpy as np
                np.random.seed(42)
                n_subj  = 15  # subjects per treatment — enough for SD
                rows    = []
                # Expected mean changes per visit (realistic clinical values)
                trt_means = {
                    trts[0]: {v: chg for v, chg in zip(visit_order, [0, -1.5, -2.1, -2.9, -3.0][:len(visit_order)])},
                    trts[1]: {v: chg for v, chg in zip(visit_order, [0, -3.2, -5.8, -8.7, -10.5][:len(visit_order)])} if len(trts)>1 else {}
                }
                for trt_i, trt in enumerate(trts):
                    means = trt_means.get(trt, {})
                    for subj_i in range(n_subj):
                        for v_i, v in enumerate(visit_order):
                            mean_chg = means.get(v, -v_i * 1.5 * (trt_i + 1))
                            chg = mean_chg + np.random.normal(0, 0.6)  # small SD → small SE
                            rows.append({
                                "USUBJID": f"S{trt_i}{subj_i:03d}",
                                groupby:   trt,
                                "TRT01P":  trt,
                                "VISIT":   v,
                                "AVISIT":  v,
                                "AVAL":    round(50 + chg, 1),
                                "CHG":     round(chg, 1),
                            })
                df = pd.DataFrame(rows)
            df.to_csv(inp_path, index=False)

            # Strip ggsave / library() from generated code
            r_code = state["generated_code"]
            r_code = re.sub(r'\+?\s*ggsave\s*\([^)]*\)', '', r_code, flags=re.DOTALL)
            r_code = re.sub(r'^\s*library\s*\([^)]+\)\s*$', '', r_code, flags=re.MULTILINE)
            # Aggressively strip ALL VISIT/AVISIT factor/level assignments from LLM code
            r_code = re.sub(r'[^\n]*\$VISIT\s*<-[^\n]*', '', r_code)
            r_code = re.sub(r'[^\n]*\$AVISIT\s*<-[^\n]*', '', r_code)
            r_code = re.sub(r'[^\n]*\[\[.VISIT.\]\]\s*<-[^\n]*', '', r_code)
            r_code = re.sub(r'[^\n]*\[\[.AVISIT.\]\]\s*<-[^\n]*', '', r_code)
            r_code = re.sub(r'[^\n]*factor\([^\n]*VISIT[^\n]*\)[^\n]*<-[^\n]*', '', r_code)
            r_code = re.sub(r'[^\n]*<-[^\n]*factor\([^\n]*levels\s*=[^\n]*VISIT[^\n]*\)', '', r_code)
            r_code = r_code.strip()
            # Ensure last line assigns to p
            lines = [l for l in r_code.split('\n') if l.strip()]
            if lines and not any(lines[-1].strip().startswith(x) for x in ['p <-','p=']):
                r_code += '\nif (!exists("p")) p <- last_plot()'

            full_script = f"""
user_lib <- path.expand('~/R/library')
dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))
options(warn=-1)
for (.pkg in c('ggplot2','dplyr','tidyr','scales','stringr')) {{
  if (!requireNamespace(.pkg, quietly=TRUE)) {{
    install.packages(.pkg, lib=user_lib, repos='https://cloud.r-project.org', quiet=TRUE)
  }}
}}
suppressMessages(suppressWarnings({{
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
}}))
df <- read.csv("{inp_path}", stringsAsFactors=FALSE)

{r_code}

suppressMessages(ggsave("{plot_path}", plot=p, width=10, height=6, dpi=150))
"""
            with open(script_path, "w") as f:
                f.write(full_script)

            try:
                res = subprocess.run(
                    ["Rscript", script_path],
                    capture_output=True, text=True, timeout=60
                )
            except subprocess.TimeoutExpired:
                state["execution_error"]  = "Figure script timed out (>60s)"
                state["execution_output"] = ""
                return state

            if res.returncode != 0:
                state["execution_error"]  = res.stderr
                state["execution_output"] = ""
                return state

            if os.path.exists(plot_path):
                with open(plot_path, "rb") as f:
                    state["execution_output"] = f.read()
                state["execution_error"] = ""
            else:
                state["execution_error"]  = "Figure file not created.\n" + res.stderr
                state["execution_output"] = ""
        return state

    # ── TABLE / LISTING: use subprocess with gt prefix ────────────────────
    with tempfile.TemporaryDirectory() as d:
        script_path = os.path.join(d, "tlf_script.R")

        prefix_lines = [
            "user_lib <- path.expand('~/R/library')",
            ".libPaths(c(user_lib, .libPaths()))",
            "options(warn=-1)",
            # Install any missing packages
            "for (.pkg in c('dplyr','tidyr','gt','ggplot2','tibble','stringr','scales','flextable')) {",
            "  if (!requireNamespace(.pkg, quietly=TRUE)) {",
            "    install.packages(.pkg, lib=user_lib, repos='https://cloud.r-project.org', quiet=TRUE)",
            "  }",
            "}",
            # Explicit library() calls AFTER install loop — errors are now visible
            # suppressMessages only, NOT suppressWarnings, so load failures surface
            "suppressMessages({",
            "  library(dplyr)",
            "  library(tidyr)",
            "  library(ggplot2)",   # must be loaded before any ggplot() call
            "  library(gt)",
            "  library(tibble)",
            "  library(stringr)",
            "  library(scales)",
            "})",
        ]

        if state.get("adam_csv"):
            inp = os.path.join(d, "adam.csv")
            with open(inp, "w") as f:
                f.write(state["adam_csv"])
            try:
                df_cols = pd.read_csv(io.StringIO(state["adam_csv"]), nrows=0).columns.tolist()
                trt_col = next(
                    (c for c in ["TRT01P","TRT01A","TRTP","TRTPN","ARM","ACTARM"] if c in df_cols),
                    None
                )
            except Exception:
                trt_col = None

            prefix_lines.append(f'df <- read.csv("{inp}", stringsAsFactors=FALSE)')
            if trt_col and trt_col != "TRT01P":
                prefix_lines.append(f'df$TRT01P <- df${trt_col}')
            elif not trt_col:
                prefix_lines.append('if (!"TRT01P" %in% names(df)) df$TRT01P <- "Total"')
            # Always materialise the _trt_ placeholder so template code using
            # backtick-quoted `_trt_` never fails with a tidyselect "unknown column" error.
            # Use the best trt column available (already aliased to TRT01P above).
            prefix_lines.append(
                'df[["_trt_"]] <- df[[ if ("TRT01P" %in% names(df)) "TRT01P" else names(df)[1] ]]'
            )

        full_script = "\n".join(prefix_lines + [state["generated_code"]])

        # When real ADaM data is present the treatment column is materialised as
        # both TRT01P and `_trt_` (see prefix_lines above), but backtick names
        # can still confuse tidyselect inside pivot_wider/group_by.  Replace every
        # remaining occurrence of the placeholder with the plain column name.
        if state.get("adam_csv"):
            full_script = full_script.replace('`_trt_`', 'TRT01P').replace('"_trt_"', '"TRT01P"')

        with open(script_path, "w") as f:
            f.write(full_script)

        try:
            res = subprocess.run(
                ["Rscript", script_path],
                capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            state["execution_error"]  = "R script timed out (>60s)"
            state["execution_output"] = ""
            return state

        if res.returncode != 0:
            stderr_lines = res.stderr.splitlines()
            error_lines  = [
                l for l in stderr_lines
                if any(kw in l for kw in ["Error", "error", "Execution halted", "object '"])
                and not any(kw in l for kw in ["masked from", "Attaching", "summarise()", "grouped by"])
            ]
            clean_error = "\n".join(error_lines) if error_lines else res.stderr
            state["execution_error"]  = clean_error
            state["execution_output"] = ""
            return state

        state["execution_error"]  = ""
        state["execution_output"] = res.stdout

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Validator
# ══════════════════════════════════════════════════════════════════════════════
def node_validate(state: ShellTLFState) -> ShellTLFState:
    """
    Validates output against spec.
    For Figures: inspects generated R code for required elements.
    For Tables: checks structural output.
    """
    if state.get("execution_error") and not state.get("execution_output"):
        state["validation_result"] = f"fail: R execution error — {state['execution_error'][:300]}"
        return state

    output = state["execution_output"]
    if not output:
        state["validation_result"] = "fail: empty output"
        return state

    spec        = state["parsed_spec"]
    output_type = spec.get("output_type", "Table")
    code        = state.get("generated_code", "")

    # ── FIGURE validation — inspect R code for required elements ─────────
    if output_type == "Figure" or state.get("detected_type") == "figure":
        reqs   = state.get("_fig_requirements", {})
        issues = []

        # 1. Title present
        if reqs.get("title") and reqs["title"] not in ["Clinical Figure",""] :
            title_safe = reqs["title"].replace('"',"").replace("'","")[:30]
            if title_safe.lower() not in code.lower() and "ggtitle" not in code and 'title=' not in code:
                issues.append("title missing from plot code")

        # 2. geom_line required for line figures
        if reqs.get("needs_line") and "geom_line" not in code:
            issues.append("geom_line() missing — visits not connected with lines")

        # 3. Error bars required
        if reqs.get("needs_error_bars") and "geom_errorbar" not in code and "geom_ribbon" not in code:
            issues.append("error bars (geom_errorbar) missing")

        # 4. Reference line at Y=0
        if reqs.get("needs_ref_zero") and "geom_hline" not in code:
            issues.append("reference line at Y=0 (geom_hline) missing")

        # 5. Visit order — check alphabetical sort is not used
        visit_order = reqs.get("visit_order", [])
        if visit_order and "sort(" in code and "VISIT" in code:
            issues.append("VISIT appears to be sorted alphabetically — must use factor with clinical order")

        # 6. Treatment group coloring
        groupby = spec.get("groupby_var") or spec.get("groupby") or "TRT01P"
        if groupby not in code and "color" not in code.lower() and "fill" not in code.lower():
            issues.append(f"treatment grouping ({groupby}) missing from aesthetics")

        if issues:
            state["validation_result"] = "fail: figure missing — " + "; ".join(issues)
        else:
            state["validation_result"] = "pass"
        return state

    # ── TABLE / LISTING validation ────────────────────────────────────────
    issues    = []
    output_str = output if isinstance(output, str) else ""

    cols = spec.get("columns", [])
    for col in cols[:3]:
        if col and col.lower() not in output_str.lower():
            issues.append(f"column '{col}' not found in output")

    if issues:
        state["validation_result"] = "fail: " + "; ".join(issues)
    else:
        state["validation_result"] = "pass"

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Fix
# ══════════════════════════════════════════════════════════════════════════════
def node_fix(state: ShellTLFState) -> ShellTLFState:
    """
    Fix broken/incomplete R code.
    For figures: applies targeted rule-based patches first (no LLM needed),
    then falls back to LLM for remaining issues.
    For tables: LLM-based fix.
    """
    code        = state.get("generated_code", "")
    val_result  = state.get("validation_result", "")
    spec        = state.get("parsed_spec", {})
    output_type = spec.get("output_type", "Table")
    reqs        = state.get("_fig_requirements", {})

    # ── FIGURE: targeted rule-based patches ──────────────────────────────
    if output_type == "Figure" or state.get("detected_type") == "figure":
        patched = False

        # Fix 1: geom_line missing
        if "geom_line() missing" in val_result and "geom_line" not in code:
            groupby = spec.get("groupby_var") or spec.get("groupby") or "TRT01P"
            # Insert geom_line after geom_point or after ggplot line
            if "geom_point" in code:
                code = code.replace(
                    "geom_point(",
                    f"geom_line(aes(group={groupby}), linewidth=1) +\n  geom_point("
                )
            else:
                code = code.replace(
                    "p <- ggplot(",
                    f"p <- ggplot("
                )
                code += f'\np <- p + geom_line(aes(group={groupby}), linewidth=1)'
            patched = True

        # Fix 2: error bars missing
        if "geom_errorbar" in val_result and "geom_errorbar" not in code:
            if "df_sum" in code:
                # df_sum already computed — just add errorbar geom
                code += '\np <- p + geom_errorbar(aes(ymin=LOWER, ymax=UPPER), width=0.2, alpha=0.7)'
            patched = True

        # Fix 3: reference line missing
        if "geom_hline" in val_result and "geom_hline" not in code:
            code += '\np <- p + geom_hline(yintercept=0, linetype="dashed", color="gray50", linewidth=0.8)'
            patched = True

        # Fix 4: visit alphabetical sort — add factor ordering
        if "sorted alphabetically" in val_result or ("sort(" in code and "VISIT" in code):
            visit_order = reqs.get("visit_order", [])
            if visit_order:
                levels_r = "c(" + ", ".join(f'"{v}"' for v in visit_order) + ")"
                visit_fix = f"""
# Fix: convert VISIT to ordered factor
for (.vc in c("VISIT","AVISIT")) {{
  if (.vc %in% names(df_sum)) df_sum[[.vc]] <- factor(df_sum[[.vc]], levels={levels_r})
  if (.vc %in% names(df))     df[[.vc]]     <- factor(df[[.vc]],     levels={levels_r})
}}
"""
                code = visit_fix + code
                patched = True

        if patched:
            state["generated_code"] = code
            state["retry_count"]    = state.get("retry_count", 0) + 1
            return state

        # LLM fallback for figures — targeted prompt
        prompt = f"""You are an R ggplot2 expert fixing a clinical figure.

CURRENT CODE:
{code}

VALIDATION FAILURES:
{val_result}

FIGURE REQUIREMENTS:
{reqs}

FIX RULES:
- geom_line missing → add geom_line(aes(group=TRT01P), linewidth=1)
- Visit alphabetical → add factor(VISIT, levels=c("Baseline","Week 2","Week 4",...)) BEFORE ggplot call
- Error bars missing → add geom_errorbar(aes(ymin=LOWER, ymax=UPPER), width=0.2)
- Reference line missing → add geom_hline(yintercept=0, linetype="dashed", color="gray50")
- Do NOT add library() or ggsave() calls
- Last line must be: p <- <plot object>
- Return ONLY complete corrected R code. No markdown. No explanations.
"""

    else:
        # ── TABLE: LLM-based fix ──────────────────────────────────────────
        prompt = f"""You are an R clinical programmer fixing broken TLF code.

ORIGINAL CODE:
{code}

ERROR / VALIDATION FAILURE:
{state.get('execution_error') or val_result}

SPEC:
{spec}

RULES:
- Fix ONLY what caused the error. Preserve all other logic.
- Do NOT add library(), read.csv, or ggsave — handled externally.
- Do NOT use mutate(col = c("a","b","c","d")) on grouped data — use bind_rows() pattern.
- Do NOT use column_labels() — use cols_label() for renaming gt columns.
- ALL column names in data.frame must be non-empty strings.
- Return ONLY corrected R code. No markdown. No explanation.
"""

    try:
        raw = _call_llm(prompt)
    except LLMRateLimitError as e:
        state["execution_error"] = (
            str(state.get("execution_error") or "") +
            f"\n[LLM rate-limited — auto-fix skipped: {e}]"
        )
        state["llm_unavailable"] = True
        state["retry_count"]     = MAX_RETRIES
        return state

    raw = re.sub(r'```[rR]?\n?', '', raw)
    raw = re.sub(r'```', '', raw).strip()
    if output_type == "Figure":
        raw = re.sub(r'\+?\s*ggsave\s*\([^)]*\)', '', raw, flags=re.DOTALL).strip()
        lines = [l for l in raw.split('\n') if l.strip()]
        if lines and not lines[-1].strip().startswith('p <-'):
            raw += '\np <- last_plot()'

    state["generated_code"] = raw
    state["retry_count"]    = state.get("retry_count", 0) + 1
    return state


# ══════════════════════════════════════════════════════════════════════════════
# REAL LANGGRAPH PIPELINE
# Graph:  parse_shell → generate_code → execute → validate
#                                          ↑              |
#                                          └──── fix ←────┘  (on fail, up to MAX_RETRIES)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from langgraph.graph import StateGraph, END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


def _should_fix_or_end(state: ShellTLFState) -> str:
    """
    Conditional edge after validate node.
    Returns "fix" if validation failed and retries remain, else "end".
    Also returns "end" immediately if LLM is rate-limited / unavailable.
    """
    if state["validation_result"] == "pass":
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "end"
    if state.get("llm_unavailable"):
        return "end"
    return "fix"


def _build_langgraph() -> "StateGraph":
    """
    Build and compile the TLF state graph.

    Nodes
    ─────
    parse_shell   : LLM extracts structured spec from raw shell text
    generate_code : template router + optional LLM enhancement → R code
    execute       : Rscript subprocess → stdout / PNG bytes
    validate      : structural checks on output vs spec
    fix           : LLM patches broken code using error + spec context

    Edges
    ─────
    parse_shell → generate_code → execute → validate
                                                │
                         ┌─── "fix" ──── fix ←─┘  (validation_result != "pass"
                         │                         AND retry_count < MAX_RETRIES)
                         └─── "end" ──── END       (pass  OR  retries exhausted)
    """
    graph = StateGraph(ShellTLFState)

    # Register nodes
    graph.add_node("parse_shell",   node_parse_shell)
    graph.add_node("generate_code", node_generate_code)
    graph.add_node("execute",       node_execute)
    graph.add_node("validate",      node_validate)
    graph.add_node("fix",           node_fix)

    # Linear edges
    graph.set_entry_point("parse_shell")
    graph.add_edge("parse_shell",   "generate_code")
    graph.add_edge("generate_code", "execute")
    graph.add_edge("execute",       "validate")

    # Conditional edge: validate → fix → execute  OR  validate → END
    graph.add_conditional_edges(
        "validate",
        _should_fix_or_end,
        {"fix": "fix", "end": END}
    )
    graph.add_edge("fix", "execute")   # fix always goes back to execute

    return graph.compile()


# Compiled graph singleton (built once, reused)
_TLF_GRAPH = None

def _get_graph():
    global _TLF_GRAPH
    if _TLF_GRAPH is None:
        _TLF_GRAPH = _build_langgraph()
    return _TLF_GRAPH


def run_shell_pipeline(
    shell_text: str,
    adam_csv: Optional[str] = None,
    ai_instructions: str = "",
    on_node: Optional[callable] = None,
) -> ShellTLFState:
    """
    Execute the TLF agentic pipeline.

    Uses real LangGraph when available; falls back to manual loop if
    langgraph is not installed (e.g. first deploy before requirements update).

    Parameters
    ----------
    shell_text      : raw mock shell text
    adam_csv        : optional ADaM CSV string
    ai_instructions : extra user instructions forwarded to generate_code node
    on_node         : optional callback(node_name, state) called after each node
                      — used by the UI to stream progress updates
    """
    init_state: ShellTLFState = {
        "shell_text":        shell_text,
        "adam_csv":          adam_csv,
        "parsed_spec":       {},
        "generated_code":    "",
        "execution_output":  "",
        "execution_error":   "",
        "validation_result": "",
        "retry_count":       0,
        "final_r_code":      "",
        "final_output":      "",
        "detected_type":     "",
        "ai_instructions":   ai_instructions,
        "llm_unavailable":   False,
    }

    if _LANGGRAPH_AVAILABLE:
        # ── Real LangGraph execution ─────────────────────────────────────
        # graph.stream() yields {node_name: output_state} dicts, not tuples
        graph = _get_graph()
        state = init_state
        for chunk in graph.stream(init_state):
            # chunk is {node_name: state_dict}
            for node_name, node_output in chunk.items():
                state = node_output
                if on_node:
                    on_node(node_name, state)
    else:
        # ── Fallback: manual loop (no langgraph installed) ────────────────
        state = init_state
        state = node_parse_shell(state)
        if on_node: on_node("parse_shell", state)
        state = node_generate_code(state)
        if on_node: on_node("generate_code", state)
        for _ in range(MAX_RETRIES + 1):
            state = node_execute(state)
            if on_node: on_node("execute", state)
            state = node_validate(state)
            if on_node: on_node("validate", state)
            if state["validation_result"] == "pass" or state["retry_count"] >= MAX_RETRIES:
                break
            state = node_fix(state)
            if on_node: on_node("fix", state)

    state["final_r_code"] = state["generated_code"]
    state["final_output"]  = state["execution_output"]
    return state


# ══════════════════════════════════════════════════════════════════════════════
# DIFF HELPER (same as graph_builder)
# ══════════════════════════════════════════════════════════════════════════════
def _show_code_diff(old_code: str, new_code: str):
    import difflib
    diff = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm='')
    html = ["<pre style='font-family:monospace; font-size:13px; line-height:1.5;'>"]
    for line in diff:
        if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
            continue
        elif line.startswith('+'):
            html.append(f"<span style='background:#1a4a1a;color:#90ee90;display:block'>{line}</span>")
        elif line.startswith('-'):
            html.append(f"<span style='background:#4a1a1a;color:#ff9999;display:block;text-decoration:line-through'>{line}</span>")
        else:
            html.append(f"<span style='color:#ccc;display:block'>{line}</span>")
    html.append("</pre>")
    st.markdown("".join(html), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT TAB RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_shell_tlf_tab():
    _ensure_r_packages()   # install gt, dplyr, tidyr, ggplot2 etc. if missing
    st.title("📋 TLF from Mock Shell")
    st.caption("Paste or upload a mock shell → AI parses spec → generates R code → executes → validates → auto-fixes")
    st.divider()

    # ── Session state init (all keys prefixed ms_) ────────────────────────
    _defaults = {
        "ms_shell_text":             "",
        "ms_adam_csv":               None,
        "ms_parsed_spec":            None,
        "ms_r_code":                 "",
        "ms_r_code_pending":         None,
        "ms_r_code_original":        None,
        "ms_preview_html":           None,
        "ms_preview_html_before":    None,
        "ms_output_before_enhance":  None,
        "ms_output":                 None,
        "ms_output_type":            "Table",
        "ms_error":                  None,
        "ms_validation":             "",
        "ms_retry_count":            0,
        "ms_pipeline_done":          False,
        "ms_run_now":                False,
        "ms_agent_log":              [],
        "ms_backend":                "LangGraph",
        "ms_ai_instructions":        "",
        "ms_enhance_text":           "",
    }
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def _clear():
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Mock Shell Input
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📄 Mock Shell Input")

    shell_tab1, shell_tab2 = st.tabs(["📋 Paste Shell", "📁 Upload Shell File"])

    with shell_tab1:
        shell_pasted = st.text_area(
            "Paste your mock shell here",
            height=200,
            placeholder="""Table 14.1.1  Demographic and Baseline Characteristics
Safety Population

                                    Placebo        Drug A        Total
                                    (N=XX)         (N=XX)        (N=XX)
                                    ──────────     ──────────    ──────────
Age (years)
  n
  Mean (SD)
  Median
  Min, Max

Sex, n (%)
  Male
  Female

a. Source: ADSL
b. Note: xx""",
            key="ms_shell_paste_area"
        )
        if shell_pasted.strip():
            st.session_state["ms_shell_text"] = shell_pasted

    with shell_tab2:
        uploaded_shell = st.file_uploader(
            "Upload shell (.txt, .rtf, .csv, .xlsx, .xls, .docx)",
            type=["txt", "rtf", "csv", "xlsx", "xls", "docx"],
            key="ms_shell_upload"
        )
        if uploaded_shell:
            try:
                ext = os.path.splitext(uploaded_shell.name)[1].lower()

                if ext in (".xlsx", ".xls"):
                    import openpyxl
                    wb    = openpyxl.load_workbook(uploaded_shell, data_only=True)
                    ws    = wb.active
                    lines = []
                    for row in ws.iter_rows():
                        vals = [str(c.value).strip() if c.value is not None else "" for c in row]
                        if any(v for v in vals):
                            lines.append("  ".join(vals))
                    shell_content = "\n".join(lines)

                elif ext == ".docx":
                    try:
                        import docx as python_docx
                        doc = python_docx.Document(uploaded_shell)
                        shell_content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    except ImportError:
                        st.error("python-docx not installed. Add 'python-docx' to requirements.txt")
                        st.stop()

                else:
                    shell_content = uploaded_shell.read().decode("utf-8", errors="ignore")

                # Always runs regardless of file type
                st.session_state["ms_shell_text"] = shell_content
                st.success(f"✅ Loaded shell: {uploaded_shell.name} ({len(shell_content)} chars)")
                with st.expander("👁️ Shell Preview"):
                    st.text(shell_content[:1000])

            except Exception as e:
                st.error(f"Failed to read file: {e}")

    # Show current shell
    if st.session_state["ms_shell_text"]:
        with st.expander("✅ Current Shell (click to view)", expanded=False):
            st.text(st.session_state["ms_shell_text"][:800])

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — ADaM Dataset (optional)
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📊 ADaM Dataset (Optional)")
    st.caption("If not provided, AI will generate realistic dummy data matching the shell spec.")

    adam_tab1, adam_tab2 = st.tabs(["📁 Upload ADaM CSV", "📋 Paste CSV"])

    with adam_tab1:
        uploaded_adam = st.file_uploader(
            "Upload ADaM CSV (ADSL, ADAE, ADVS, etc.)",
            type=["csv", "xlsx", "xls"],
            key="ms_adam_upload"
        )
        if uploaded_adam:
            try:
                ext = os.path.splitext(uploaded_adam.name)[1].lower()
                df_adam = pd.read_excel(uploaded_adam) if ext in (".xlsx", ".xls") else pd.read_csv(uploaded_adam)
                csv_str = df_adam.to_csv(index=False)
                st.session_state["ms_adam_csv"] = csv_str
                st.success(f"✅ Loaded — {df_adam.shape[0]} rows × {df_adam.shape[1]} cols")
                with st.expander("👁️ Data Preview"):
                    st.dataframe(df_adam.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load ADaM: {e}")

    with adam_tab2:
        adam_pasted = st.text_area(
            "Paste ADaM CSV here",
            height=100,
            key="ms_adam_paste_area"
        )
        if adam_pasted.strip():
            try:
                df_adam = pd.read_csv(io.StringIO(adam_pasted))
                st.session_state["ms_adam_csv"] = adam_pasted
                st.success(f"✅ Parsed — {df_adam.shape[0]} rows × {df_adam.shape[1]} cols")
                st.dataframe(df_adam.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"CSV parse error: {e}")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — AI Instructions box (same pattern as graph_builder)
    # ════════════════════════════════════════════════════════════════════════
    ai_instructions = st.text_area(
        "✨ Additional AI Instructions (optional)",
        placeholder="e.g. Use gt package with blue header, round to 1 decimal, add p-value column, apply ICH E3 footnote format...",
        height=80,
        key="ms_ai_instructions"
    )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Generate / Clear buttons
    # ════════════════════════════════════════════════════════════════════════
    btn_col1, btn_col2 = st.columns([4, 1])
    with btn_col1:
        generate_btn = st.button(
            "🤖 Generate TLF from Shell",
            type="primary",
            use_container_width=True,
            key="ms_generate_btn"
        )
    with btn_col2:
        st.button("🗑️ Clear", on_click=_clear, use_container_width=True, key="ms_clear_btn")

    # ── Validate inputs before running ───────────────────────────────────
    if generate_btn:
        if not st.session_state["ms_shell_text"].strip():
            st.error("⚠️ Please paste or upload a mock shell first.")
            st.stop()

        # Append any extra instructions to shell text for parser
        shell_for_pipeline = st.session_state["ms_shell_text"]
        if ai_instructions.strip():
            shell_for_pipeline += f"\n\nADDITIONAL REQUIREMENTS:\n{ai_instructions}"

        # ── Run real LangGraph pipeline with streaming progress ──────────
        agent_log = []
        progress  = st.progress(0, text="🧠 Starting agentic pipeline...")

        _node_progress = {
            "parse_shell":   (15,  "🔍 Node 1 — Parsing mock shell..."),
            "generate_code": (35,  "⚙️ Node 2 — Generating R code..."),
            "execute":       (60,  "▶️ Node 3 — Executing R..."),
            "validate":      (75,  "🔍 Node 4 — Validating output..."),
            "fix":           (85,  "🔧 Node 5 — AI fixing code..."),
        }
        _template_map = {
            "demog":      "📊 Demographics template",
            "ae_summary": "🔴 AE Summary template",
            "ae_socpt":   "🔴 AE SOC/PT template",
            "lab":        "🧪 Lab Values template",
            "vitals":     "💓 Vital Signs template",
            "efficacy":   "📈 Efficacy template",
            "listing":    "📋 Listing template",
            "figure":     "📊 Figure (ggplot2 via graph_builder)",
            "llm":        "🤖 LLM generated",
        }

        def _on_node(node_name: str, state: ShellTLFState):
            """Called by run_shell_pipeline after each node completes."""
            pct, text = _node_progress.get(node_name, (50, f"⚙️ {node_name}..."))
            # Increment pct for repeated execute/validate/fix cycles
            retry = state.get("retry_count", 0)
            if retry > 0:
                pct = min(pct + retry * 5, 95)
                text = text.replace("...", f" (retry {retry})...")
            progress.progress(pct, text=text)

            # Build agent log entry
            if node_name == "parse_shell":
                agent_log.append(("✅ Shell Parsed", str(state["parsed_spec"])[:300]))
            elif node_name == "generate_code":
                detected = state.get("detected_type", "unknown")
                ai_note  = " + AI customised" if ai_instructions.strip() and detected != "llm" else ""
                agent_log.append(("✅ Code Generated",
                    f"{_template_map.get(detected, detected)}{ai_note} — {len(state['generated_code'])} chars"))
            elif node_name == "execute":
                if state["execution_error"]:
                    agent_log.append((f"⚠️ Execute Failed",  state["execution_error"][:200]))
                else:
                    agent_log.append((f"✅ Execute OK", f"Output: {len(str(state['execution_output']))} chars"))
            elif node_name == "validate":
                agent_log.append((f"🔍 Validate", state["validation_result"]))
            elif node_name == "fix":
                if state.get("llm_unavailable"):
                    agent_log.append((
                        f"⚠️ Fix Skipped",
                        "LLM rate limit reached — returning best available output. "
                        "Try again in ~30 minutes or upgrade Groq tier."
                    ))
                else:
                    agent_log.append((f"🔧 Fix Applied (retry {retry})", "Code patched by LLM"))

        try:
            with st.spinner(""):
                backend = "LangGraph" if _LANGGRAPH_AVAILABLE else "fallback loop"
                progress.progress(5, text=f"🧠 Using {backend}...")

                final_state = run_shell_pipeline(
                    shell_text      = shell_for_pipeline,
                    adam_csv        = st.session_state.get("ms_adam_csv"),
                    ai_instructions = ai_instructions.strip(),
                    on_node         = _on_node,
                )

                # Persist to session state
                st.session_state["ms_r_code"]       = final_state["final_r_code"]
                st.session_state["ms_output"]        = final_state["final_output"]
                st.session_state["ms_error"]         = final_state["execution_error"] or None
                st.session_state["ms_validation"]    = final_state["validation_result"]
                st.session_state["ms_retry_count"]   = final_state["retry_count"]
                st.session_state["ms_parsed_spec"]   = final_state["parsed_spec"]
                st.session_state["ms_output_type"]   = final_state["parsed_spec"].get("output_type", "Table")
                st.session_state["ms_pipeline_done"] = True
                st.session_state["ms_agent_log"]     = agent_log
                st.session_state["ms_backend"]       = backend

                progress.progress(100, text="✅ Pipeline complete!")

        except Exception as e:
            progress.empty()
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            st.stop()

        st.rerun()

    # ── Re-run from edited code ───────────────────────────────────────────
    if st.session_state.get("ms_run_now"):
        st.session_state["ms_run_now"] = False
        spec = st.session_state.get("ms_parsed_spec") or {}
        run_state: ShellTLFState = {
            "shell_text":        st.session_state["ms_shell_text"],
            "adam_csv":          st.session_state.get("ms_adam_csv"),
            "parsed_spec":       spec,
            "generated_code":    st.session_state["ms_r_code"],
            "execution_output":  "",
            "execution_error":   "",
            "validation_result": "",
            "retry_count":       0,
            "final_r_code":      "",
            "final_output":      "",
            "detected_type":     "",
            "ai_instructions":   "",
        }
        with st.spinner("⚙️ Running R..."):
            run_state = node_execute(run_state)
        st.session_state["ms_output"] = run_state["final_output"] or run_state["execution_output"]
        st.session_state["ms_error"]  = run_state["execution_error"] or None
        st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Output Display
    # ════════════════════════════════════════════════════════════════════════
    if not st.session_state.get("ms_pipeline_done"):
        return

    st.divider()
    st.subheader("📤 Output")

    # ── Agent log expander ────────────────────────────────────────────────
    agent_log = st.session_state.get("ms_agent_log", [])
    if agent_log:
        backend = st.session_state.get("ms_backend", "")
        badge   = "🟢 LangGraph" if "LangGraph" in backend else "🟡 fallback loop"
        with st.expander(f"🤖 Agent Pipeline Log  [{badge}]", expanded=False):
            if "LangGraph" not in backend:
                st.info(
                    "LangGraph not installed — running sequential fallback. "
                    "Add `langgraph` to requirements.txt to enable the real graph."
                )
            for step, detail in agent_log:
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(f"**{step}**")
                with col_b:
                    st.caption(detail)

    # ── Parsed spec summary ───────────────────────────────────────────────
    spec = st.session_state.get("ms_parsed_spec") or {}
    if spec:
        with st.expander("🔍 Parsed Shell Spec", expanded=False):
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.markdown(f"**Type:** {spec.get('output_type','Table')}")
                st.markdown(f"**TLF #:** {spec.get('tlf_number','')}")
            with meta_col2:
                st.markdown(f"**Dataset:** {spec.get('dataset_hint','')}")
                st.markdown(f"**Population:** {spec.get('population','')}")
            with meta_col3:
                st.markdown(f"**Pop Flag:** {spec.get('pop_flag','')}")
                st.markdown(f"**Group By:** {spec.get('groupby','')}")
            if spec.get("title"):
                st.markdown(f"**Title:** {spec['title']}")
            if spec.get("columns"):
                st.markdown(f"**Columns:** {', '.join(spec['columns'])}")
            if spec.get("footnotes"):
                for fn in spec["footnotes"]:
                    st.caption(f"ᵃ {fn}")

    # ── Validation badge ─────────────────────────────────────────────────
    val = st.session_state.get("ms_validation", "")
    retries = st.session_state.get("ms_retry_count", 0)
    if val == "pass":
        st.success(f"✅ Validation passed {'(after ' + str(retries) + ' retries)' if retries else '(first attempt)'}")
    elif val:
        st.warning(f"⚠️ Validation: {val}")

    # ── Pending enhancement diff — ABOVE tabs, mirrors graph_builder.py ────
    if st.session_state.get("ms_r_code_pending"):
        st.warning("⚠️ Enhancement applied — review changes and confirm:")
        st.markdown("**Code Changes** (🟢 added | 🔴 removed):")
        _show_code_diff(
            st.session_state.get("ms_r_code_original", ""),
            st.session_state["ms_r_code_pending"]
        )
        _dc1, _dc2, _dc3 = st.columns(3)
        with _dc1:
            if st.button("✅ Accept Changes", use_container_width=True, key="ms_apply"):
                st.session_state["ms_r_code"]                = st.session_state["ms_r_code_pending"]
                st.session_state["ms_r_code_original"]       = None
                st.session_state["ms_r_code_pending"]        = None
                st.session_state["ms_preview_html"]          = None
                st.session_state["ms_output_before_enhance"] = None
                # Output already updated inline when enhancement ran — no re-execute needed
                st.rerun()
        with _dc2:
            if st.button("👁️ Preview", use_container_width=True, key="ms_preview_btn"):
                with st.spinner("Generating preview..."):
                    try:
                        _prev_state: ShellTLFState = {
                            "shell_text":        st.session_state.get("ms_shell_text", ""),
                            "adam_csv":          st.session_state.get("ms_adam_csv"),
                            "parsed_spec":       st.session_state.get("ms_parsed_spec", {}),
                            "generated_code":    st.session_state["ms_r_code_pending"],
                            "execution_output":  "", "execution_error":   "",
                            "validation_result": "", "retry_count":       0,
                            "final_r_code":      "", "final_output":      "",
                            "detected_type":     "", "ai_instructions":   "",
                            "llm_unavailable":   False,
                        }
                        _prev_state = node_execute(_prev_state)
                        if _prev_state["execution_error"]:
                            st.error(f"Preview error: {_prev_state['execution_error']}")
                        else:
                            st.session_state["ms_preview_html"] = _prev_state["execution_output"]
                            # Left tab = original output captured BEFORE enhancement ran
                            # (ms_output_before_enhance set when Apply Enhancement was clicked)
                            st.session_state["ms_preview_html_before"] = (
                                st.session_state.get("ms_output_before_enhance") or
                                st.session_state.get("ms_output", "")
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(f"Preview failed: {e}")
        with _dc3:
            if st.button("❌ Reject Changes", use_container_width=True, key="ms_reject"):
                orig_code   = st.session_state.get("ms_r_code_original") or st.session_state.get("ms_r_code", "")
                orig_output = st.session_state.get("ms_output_before_enhance", "")
                st.session_state["ms_r_code"]                = orig_code
                st.session_state["ms_r_code_pending"]        = None
                st.session_state["ms_preview_html"]          = None
                st.session_state["ms_output_before_enhance"] = None
                # Restore original output directly — no re-execution needed
                if orig_output:
                    st.session_state["ms_output"] = orig_output
                    st.session_state["ms_error"]  = None
                st.rerun()

        # ── Side-by-side before/after HTML preview ────────────────────────
        if st.session_state.get("ms_preview_html"):
            st.markdown("**👁️ Preview (not applied yet):**")
            _col_b, _col_a = st.columns(2)
            with _col_b:
                st.markdown("**⬅️ Before (original):**")
                _before = st.session_state.get("ms_preview_html_before", "")
                if _before:
                    st.components.v1.html(_before, height=400, scrolling=True)
                else:
                    st.info("No original output to compare.")
            with _col_a:
                st.markdown("**➡️ After (enhanced):**")
                st.components.v1.html(
                    st.session_state["ms_preview_html"], height=400, scrolling=True
                )
        st.divider()

    # ── Tabs: TLF Output | R Code ─────────────────────────────────────────
    output_type = st.session_state.get("ms_output_type", "Table")
    out_tab1, out_tab2 = st.tabs([
        "📊 TLF Output",
        "💻 R Code"
    ])

    with out_tab1:
        output = st.session_state.get("ms_output")
        error  = st.session_state.get("ms_error")

        if error:
            st.error(f"R Error:\n{error}")

        if output:
            if output_type == "Figure" and isinstance(output, (bytes, bytearray)):
                st.image(output, use_container_width=True)
                st.download_button(
                    "⬇️ Download Figure PNG",
                    data=output,
                    file_name="figure.png",
                    mime="image/png"
                )
            elif isinstance(output, str) and output.strip().startswith("<"):
                # HTML table from gt
                st.components.v1.html(output, height=600, scrolling=True)
                st.download_button(
                    "⬇️ Download HTML Table",
                    data=output,
                    file_name="tlf_output.html",
                    mime="text/html"
                )
            else:
                # Plain text listing
                st.code(output if isinstance(output, str) else str(output), language="")
                if isinstance(output, str):
                    st.download_button(
                        "⬇️ Download Listing",
                        data=output,
                        file_name="listing.txt",
                        mime="text/plain"
                    )

    with out_tab2:
        # ── Editable code ─────────────────────────────────────────────────
        current_code = st.session_state.get("ms_r_code", "")
        edited = st.text_area(
            "Edit R Code",
            value=current_code,
            height=350,
            key=f"ms_code_editor_{hash(current_code)}"
        )

        btn_a, btn_b = st.columns(2)
        with btn_a:
            if st.button("▶️ Run Edited Code", type="primary", use_container_width=True, key="ms_run_edit"):
                st.session_state["ms_r_code"] = edited
                st.session_state["ms_run_now"] = True
                st.rerun()
        with btn_b:
            st.download_button(
                "⬇️ Download R Code",
                data=edited,
                file_name="tlf_from_shell.R",
                mime="text/plain",
                use_container_width=True
            )

    # ── Custom enhancement box — mirrors graph_builder.py pattern exactly ──
    # Only shown after a table has been generated (same guard as graph_builder)
    if st.session_state.get("ms_pipeline_done") and st.session_state.get("ms_r_code"):
        st.divider()
        enhance_text = st.text_area(
            "✨ Custom Enhancement (optional)",
            placeholder="e.g. Add p-value column, change header color to navy, bold the Total column, add risk difference row...",
            height=80,
            key="ms_enhance_text",
        )

        ecol1, ecol2 = st.columns([4, 1])
        with ecol1:
            apply_enhance = st.button(
                "🔧 Apply Enhancement", type="primary",
                use_container_width=True, key="ms_enhance_btn"
            )
        with ecol2:
            if st.button("↩️ Revert", use_container_width=True, key="ms_enhance_revert"):
                orig = st.session_state.get("ms_r_code_original", "")
                if orig:
                    st.session_state["ms_r_code"]         = orig
                    st.session_state["ms_r_code_pending"] = None
                    st.session_state["ms_r_code_original"]= ""
                    st.session_state["ms_run_now"]        = True
                    st.rerun()
                else:
                    st.info("Nothing to revert.")

        if apply_enhance:
            if not enhance_text.strip():
                st.warning("Enter enhancement instructions first.")
            else:
                # Build on currently accepted code (cumulative enhancements preserved)
                existing_code = st.session_state.get("ms_r_code", "")
                if not existing_code.strip():
                    st.error("No R code to enhance yet — generate a table first.")
                    st.stop()

                enhance_prompt = (
                    f"You are an R clinical TLF code editor. "
                    f"Apply ONLY the requested change to the existing code.\n\n"
                    f"EXISTING CODE:\n```r\n{existing_code}\n```\n\n"
                    f"REQUEST: {enhance_text}\n\n"
                    f"RULES:\n"
                    f"- Touch ONLY what the request asks. Preserve everything else exactly as in EXISTING CODE.\n"
                    f"- MERGE new settings into existing tab_style()/tab_options() blocks — never rewrite the whole block.\n"
                    f"- Keep all column headers, footnotes, population label, and grouping from EXISTING CODE unless request explicitly changes them.\n"
                    f"- Never add: read.csv, hardcoded file paths, ggsave, or new library() calls not already present.\n"
                    f"- Before outputting, verify: is every tab_options(), tab_style(), cols_label() from EXISTING CODE still present?\n"
                    f"- Return ONLY complete R code. No explanations, no markdown fences.\n"
                )

                with st.spinner("🤖 Applying enhancement..."):
                    raw = None
                    try:
                        # Groq first (fast), Gemini fallback — same order as graph_builder
                        res = _groq.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": enhance_prompt}],
                            temperature=0
                        )
                        raw = res.choices[0].message.content
                    except Exception:
                        try:
                            raw = _gemini.models.generate_content(
                                model="gemini-2.0-flash", contents=enhance_prompt
                            ).text
                        except Exception:
                            st.warning("⚠️ Enhancement failed — using base code.")

                    if raw:
                        raw = re.sub(r'```[rR]?\n?', '', raw)
                        raw = re.sub(r'```', '', raw).strip()
                        raw = _sanitise_r_code(raw)

                        # ── Snapshot BEFORE running — must happen here, inline, ──
                        # before any execution overwrites ms_output.
                        # Do NOT use ms_run_now (it fires before this code runs
                        # on the next render, wiping ms_output first).
                        before_output = st.session_state.get("ms_output", "")

                        # Execute enhanced code inline right now
                        _enh_state: ShellTLFState = {
                            "shell_text":        st.session_state.get("ms_shell_text", ""),
                            "adam_csv":          st.session_state.get("ms_adam_csv"),
                            "parsed_spec":       st.session_state.get("ms_parsed_spec", {}),
                            "generated_code":    raw,
                            "execution_output":  "", "execution_error":   "",
                            "validation_result": "", "retry_count":       0,
                            "final_r_code":      "", "final_output":      "",
                            "detected_type":     "", "ai_instructions":   "",
                            "llm_unavailable":   False,
                        }
                        with st.spinner("⚙️ Running enhanced R..."):
                            _enh_state = node_execute(_enh_state)

                        # Store snapshot (original) and new output separately
                        st.session_state["ms_output_before_enhance"] = before_output
                        st.session_state["ms_r_code_original"]       = existing_code
                        st.session_state["ms_r_code_pending"]        = raw
                        st.session_state["ms_r_code"]                = raw
                        # Update live output with enhanced result
                        if not _enh_state["execution_error"]:
                            st.session_state["ms_output"] = (
                                _enh_state["final_output"] or _enh_state["execution_output"]
                            )
                            st.session_state["ms_error"] = None
                        else:
                            st.session_state["ms_error"] = _enh_state["execution_error"]
                        st.rerun()
