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

# ─── Clients (same helper as graph_builder) ──────────────────────────────────
def _get_secret(key):
    try:    return st.secrets[key]
    except Exception: return os.environ.get(key, "")

_gemini = genai.Client(api_key=_get_secret("GEMINI_API_KEY"))
_groq   = Groq(api_key=_get_secret("GROQ_API_KEY"))

MAX_RETRIES = 3

# ─── LangGraph-style State ────────────────────────────────────────────────────
class ShellTLFState(TypedDict):
    shell_text:        str            # raw mock shell pasted/uploaded
    adam_csv:          Optional[str]  # CSV string of ADaM dataset
    parsed_spec:       dict           # extracted by parse_shell node
    generated_code:    str            # R code from generate_code node
    execution_output:  str            # stdout / table HTML
    execution_error:   str            # stderr if failed
    validation_result: str            # "pass" | "fail: <reason>"
    retry_count:       int
    final_r_code:      str
    final_output:      str            # rendered HTML table or message
    detected_type:     str            # demog|ae_summary|ae_socpt|lab|vitals|efficacy|listing|llm
    ai_instructions:   str            # extra user instructions for LLM enhancement


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Shell Parser
# ══════════════════════════════════════════════════════════════════════════════
def _call_llm(prompt: str) -> str:
    """Gemini primary, Groq fallback."""
    try:
        return _gemini.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        ).text
    except Exception:
        try:
            res = _groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return res.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Both LLMs failed: {e}")


def node_parse_shell(state: ShellTLFState) -> ShellTLFState:
    """
    Extract structured spec from the mock shell text.
    Returns parsed_spec dict with keys:
      title, footnotes, population, columns, row_stubs, dataset_hint,
      output_type (Table/Listing/Figure), tlf_number
    """
    prompt = f"""You are a clinical programmer. Parse this mock shell and extract a JSON spec.

MOCK SHELL:
{state['shell_text']}

Return ONLY valid JSON, no markdown, no explanation. Schema:
{{
  "tlf_number":   "e.g. Table 14.1.1 or empty string",
  "title":        "full title text",
  "footnotes":    ["list of footnote strings"],
  "population":   "e.g. Safety Population or empty string",
  "pop_flag":     "e.g. SAFFL or ITTFL or empty string",
  "output_type":  "Table" | "Listing" | "Figure",
  "dataset_hint": "e.g. ADSL, ADAE, ADVS or empty string",
  "columns":      ["col1", "col2", ...],
  "row_stubs":    ["row label 1", "row label 2", ...],
  "statistics":   ["n", "mean", "sd", "median", "min", "max", "pct"] (subset relevant),
  "groupby":      "treatment variable name or empty string"
}}"""
    raw = _call_llm(prompt)
    raw = re.sub(r'```json|```', '', raw).strip()
    import json
    try:
        spec = json.loads(raw)
    except Exception:
        # Fallback minimal spec
        spec = {
            "tlf_number": "", "title": "Table", "footnotes": [],
            "population": "", "pop_flag": "", "output_type": "Table",
            "dataset_hint": "", "columns": [], "row_stubs": [],
            "statistics": ["n", "pct"], "groupby": ""
        }
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
    title      = spec.get("title", "Summary of Demographic and Baseline Characteristics")
    pop_flag   = spec.get("pop_flag", "SAFFL") or "SAFFL"
    footnotes  = spec.get("footnotes", [])
    groupby    = spec.get("groupby_var") or spec.get("groupby") or "TRT01P"
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
    pivot_wider(names_from={groupby}, values_from=val)
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

    bind_call = "tbl_data <- bind_rows(\n  " + ",\n  ".join(bind_vars) + "\n)"
    param_order_r = "c(" + ",".join(f'"{p}"' for p in param_names) + ")"

    code = f"""{dummy_data}
{pop_filter}

{"".join(param_blocks)}

{bind_call}

tbl_data <- tbl_data %>% select(Parameter, Statistic, everything())
tbl_data$Statistic <- gsub("^[ \\t\\r\\n]+|[ \\t\\r\\n]+$","",tbl_data$Statistic)

# Dynamic N headers using actual groupby column
trts      <- sort(unique(df${groupby}))
n_per_trt <- sapply(trts, function(t) n_distinct(df$USUBJID[df${groupby}==t]))
n_total   <- n_distinct(df$USUBJID)
col_labels <- setNames(
  c(paste0(trts,"\\n(N=",n_per_trt,")"), paste0("Total\\n(N=",n_total,")")),
  c(trts,"Total")
)

# Enforce shell parameter order
tbl_data$Parameter <- factor(tbl_data$Parameter, levels={param_order_r})
tbl_data <- tbl_data[order(tbl_data$Parameter),]
tbl_data$Parameter <- as.character(tbl_data$Parameter)

tbl <- gt(tbl_data, groupname_col="Parameter") %>%
  tab_header(title="{title}", subtitle="{pop_label}") %>%
  {fn_lines}  cols_label(.list=as.list(col_labels)) %>%
  cols_label(Statistic="") %>%
  cols_hide("Parameter") %>%
  tab_style(style=cell_text(weight="bold"), locations=cells_row_groups()) %>%
  tab_style(style=cell_text(weight="bold"), locations=cells_column_labels()) %>%
  tab_style(style=cell_text(indent=px(20)), locations=cells_body(columns="Statistic")) %>%
  cols_align(align="left", columns="Statistic") %>%
  tab_options(
    table.width=pct(100),
    row_group.background.color="#f5f5f5",
    heading.subtitle.font.size=px(13)
  )

cat(as_raw_html(tbl))
"""
    return code


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: AE Summary (incidence by treatment)
# ══════════════════════════════════════════════════════════════════════════════
def _build_ae_summary_r_code(spec: dict, has_adam: bool) -> str:
    title     = spec.get("title", "Summary of Adverse Events")
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
  sapply(trts, function(t) length(unique(df$USUBJID[df$TRT01P == t]))),
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
  tab_style(style=cell_text(weight="bold"), locations=cells_column_labels()) %>%
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
    title     = spec.get("title", "Adverse Events by System Organ Class and Preferred Term")
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
  tab_style(style=cell_text(weight="bold"), locations=cells_column_labels()) %>%
  tab_options(table.width=pct(100))

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: Lab Values / Vital Signs Summary
# ══════════════════════════════════════════════════════════════════════════════
def _build_lab_r_code(spec: dict, has_adam: bool) -> str:
    title     = spec.get("title", "Summary of Laboratory Values")
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
  tab_style(style=cell_text(weight="bold"), locations=cells_column_labels()) %>%
  cols_align(align="left", columns="Statistic") %>%
  tab_options(table.width=pct(100), row_group.background.color="#f5f5f5")

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: Efficacy / Primary Endpoint
# ══════════════════════════════════════════════════════════════════════════════
def _build_efficacy_r_code(spec: dict, has_adam: bool) -> str:
    title     = spec.get("title", "Summary of Efficacy")
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
  tab_style(style=cell_text(weight="bold"), locations=cells_column_labels()) %>%
  cols_align(align="left", columns="Statistic") %>%
  tab_options(table.width=pct(100), row_group.background.color="#f5f5f5")

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE: Generic Clinical Listing
# ══════════════════════════════════════════════════════════════════════════════
def _build_listing_r_code(spec: dict, has_adam: bool) -> str:
    title     = spec.get("title", "Clinical Listing")
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
  {fn_lines}  tab_style(style=cell_text(weight="bold"), locations=cells_column_labels()) %>%
  tab_options(table.width=pct(100))

cat(as_raw_html(tbl))
"""


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE ROUTER — detect table type and dispatch
# ══════════════════════════════════════════════════════════════════════════════
def _detect_table_type(spec: dict) -> str:
    """
    Returns one of: demog | ae_summary | ae_socpt | lab | vitals | efficacy | listing | llm
    Based on title keywords and row_stubs.
    """
    title      = spec.get("title", "").lower()
    row_stubs  = [r.lower() for r in spec.get("row_stubs", [])]
    all_text   = title + " " + " ".join(row_stubs)

    if any(k in all_text for k in ["demographic", "baseline characteristic", "age", "sex, n", "race"]):
        return "demog"
    if any(k in all_text for k in ["system organ class", "preferred term", "soc", "pt", "by body system"]):
        return "ae_socpt"
    if any(k in all_text for k in ["adverse event", "teae", "ae summary", "incidence of ae"]):
        return "ae_summary"
    if any(k in all_text for k in ["vital sign", "blood pressure", "pulse", "weight", "height", "temperature"]):
        return "vitals"
    if any(k in all_text for k in ["laborator", "lab value", "haematol", "hematol", "chemistry", "alt", "ast", "creatinine"]):
        return "lab"
    if any(k in all_text for k in ["efficacy", "primary endpoint", "response rate", "responder", "change from baseline"]):
        return "efficacy"
    if spec.get("output_type", "Table") == "Listing":
        return "listing"
    if not spec.get("row_stubs"):
        return "demog"   # safe default
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

    # Figures always go to LLM
    if output_type == "Figure":
        table_type = "llm"
    else:
        table_type = _detect_table_type(spec)

    # Dispatch to template
    if table_type == "demog":
        raw = _build_demog_r_code(spec, has_adam)
    elif table_type == "ae_summary":
        raw = _build_ae_summary_r_code(spec, has_adam)
    elif table_type == "ae_socpt":
        raw = _build_ae_socpt_r_code(spec, has_adam)
    elif table_type in ("lab", "vitals"):
        raw = _build_lab_r_code(spec, has_adam)
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
3. Do NOT load any libraries — dplyr, tidyr, gt, ggplot2 are already loaded.
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

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Executor
# ══════════════════════════════════════════════════════════════════════════════
def node_execute(state: ShellTLFState) -> ShellTLFState:
    """Run R code, capture stdout as execution_output."""
    spec        = state["parsed_spec"]
    output_type = spec.get("output_type", "Table")

    with tempfile.TemporaryDirectory() as d:
        script_path = os.path.join(d, "tlf_script.R")
        plot_path   = os.path.join(d, "figure.png")

        # Prefix: library path + preload packages silently + optional data load
        prefix_lines = [
            "user_lib <- path.expand('~/R/library')",
            "if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))",
            "options(warn=-1)",
            "suppressMessages(suppressWarnings({",
            "  library(dplyr)",
            "  library(tidyr)",
            "  library(gt)",
            "  library(ggplot2)",
            "}))",
        ]

        if state.get("adam_csv"):
            inp = os.path.join(d, "adam.csv")
            with open(inp, "w") as f:
                f.write(state["adam_csv"])
            # Auto-detect treatment column name from actual data
            try:
                df_cols = pd.read_csv(io.StringIO(state["adam_csv"]), nrows=0).columns.tolist()
                trt_col = next(
                    (c for c in ["TRT01P","TRT01A","TRTP","TRTPN","ARM","ACTARM"] if c in df_cols),
                    None
                )
            except Exception:
                trt_col = None

            prefix_lines.append(f'df <- read.csv("{inp}", stringsAsFactors=FALSE)')

            # If detected trt col differs from what template uses, alias it
            if trt_col and trt_col != "TRT01P":
                prefix_lines.append(f'df$TRT01P <- df${trt_col}')
            elif not trt_col:
                # No known treatment col — create placeholder so group_by doesn't crash
                prefix_lines.append('if (!"TRT01P" %in% names(df)) df$TRT01P <- "Total"')

        suffix_lines = []
        if output_type == "Figure":
            suffix_lines = [f'suppressMessages(ggsave("{plot_path}", width=10, height=6, dpi=150))']

        full_script = "\n".join(prefix_lines + [state["generated_code"]] + suffix_lines)

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
            # Extract actual R error — skip dplyr/package warning noise
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

        # returncode == 0 — success even if stderr has dplyr noise
        state["execution_error"] = ""

        if output_type == "Figure":
            if os.path.exists(plot_path):
                with open(plot_path, "rb") as f:
                    state["execution_output"] = f.read()   # bytes
            else:
                state["execution_error"]  = "Figure file not created.\n" + res.stderr
                state["execution_output"] = ""
        else:
            state["execution_output"] = res.stdout

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Validator
# ══════════════════════════════════════════════════════════════════════════════
def node_validate(state: ShellTLFState) -> ShellTLFState:
    """
    Check output against spec. Simple rule-based + LLM sanity check.
    Sets validation_result = "pass" or "fail: <reason>"
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

    if output_type == "Figure":
        # If we got bytes it's a PNG — assume pass
        state["validation_result"] = "pass"
        return state

    # For Table/Listing — quick structural checks
    issues = []
    output_str = output if isinstance(output, str) else ""

    # Check at least some columns mentioned appear in output
    cols = spec.get("columns", [])
    for col in cols[:3]:  # check first 3 columns only
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
    """LLM reads error + code and patches it."""
    prompt = f"""You are an R clinical programmer fixing broken TLF code.

ORIGINAL CODE:
{state['generated_code']}

ERROR / VALIDATION FAILURE:
{state.get('execution_error') or state.get('validation_result', '')}

SPEC:
{state['parsed_spec']}

RULES:
- Fix ONLY what caused the error. Preserve all other logic.
- Do NOT add library(), read.csv, or ggsave — these are handled externally.
- Do NOT use mutate(col = c("a","b","c","d")) on grouped data — use bind_rows() pattern instead.
- Do NOT use column_labels() — use cols_label() for renaming gt columns.
- ALL column names in data.frame must be non-empty strings.
- Never produce an empty string "" as a column name.
- Return ONLY corrected R code. No markdown. No explanation.
"""
    raw = _call_llm(prompt)
    raw = re.sub(r'```[rR]?\n?', '', raw)
    raw = re.sub(r'```', '', raw).strip()
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
    """
    if state["validation_result"] == "pass":
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
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
    }

    if _LANGGRAPH_AVAILABLE:
        # ── Real LangGraph execution ─────────────────────────────────────
        graph  = _get_graph()
        # stream() yields (node_name, output_state) after each node completes
        state  = init_state
        for node_name, node_output in graph.stream(init_state):
            state = node_output   # last value is the complete state after that node
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
    st.title("📋 TLF from Mock Shell")
    st.caption("Paste or upload a mock shell → AI parses spec → generates R code → executes → validates → auto-fixes")
    st.divider()

    # ── Session state init (all keys prefixed ms_) ────────────────────────
    _defaults = {
        "ms_shell_text":       "",
        "ms_adam_csv":         None,
        "ms_parsed_spec":      None,
        "ms_r_code":           "",
        "ms_r_code_pending":   None,
        "ms_r_code_original":  None,
        "ms_output":           None,
        "ms_output_type":      "Table",
        "ms_error":            None,
        "ms_validation":       "",
        "ms_retry_count":      0,
        "ms_pipeline_done":    False,
        "ms_run_now":          False,
        "ms_agent_log":        [],
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
        # ── Pending diff review (same pattern as graph_builder) ───────────
        if st.session_state.get("ms_r_code_pending"):
            st.warning("⚠️ AI wants to modify your code. Review and confirm:")
            st.markdown("**Code Changes** (🟢 added | 🔴 removed):")
            _show_code_diff(
                st.session_state["ms_r_code_original"],
                st.session_state["ms_r_code_pending"]
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ Apply Changes", use_container_width=True, key="ms_apply"):
                    st.session_state["ms_r_code"]          = st.session_state["ms_r_code_pending"]
                    st.session_state["ms_r_code_original"] = None
                    st.session_state["ms_r_code_pending"]  = None
                    st.session_state["ms_run_now"]         = True
                    st.rerun()
            with c2:
                if st.button("❌ Reject", use_container_width=True, key="ms_reject"):
                    st.session_state["ms_r_code_pending"] = None
                    st.rerun()
            with c3:
                pass  # reserve for future preview

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

    # ── Custom enhancement box (same pattern as graph_builder) ────────────
    st.divider()
    enhance_text = st.text_area(
        "✨ Custom Enhancement (optional)",
        placeholder="e.g. Add p-value column, change header color to navy, add risk difference row...",
        height=80,
        key="ms_enhance_text"
    )

    if st.button("🔧 Apply Enhancement", use_container_width=True, key="ms_enhance_btn"):
        if not enhance_text.strip():
            st.warning("Enter enhancement instructions first.")
        else:
            existing_code = st.session_state.get("ms_r_code", "")
            enhance_prompt = (
                f"You are an R clinical TLF code editor. Apply ONLY the requested change.\n\n"
                f"EXISTING CODE:\n```r\n{existing_code}\n```\n\n"
                f"REQUEST: {enhance_text}\n\n"
                f"RULES:\n"
                f"- Touch ONLY what the request asks. Preserve everything else exactly.\n"
                f"- Never add read.csv, hardcoded file paths, or ggsave.\n"
                f"- Return ONLY complete R code. No explanations, no markdown fences.\n"
            )
            with st.spinner("🤖 Applying enhancement..."):
                try:
                    raw = _call_llm(enhance_prompt)
                    raw = re.sub(r'```[rR]?\n?', '', raw)
                    raw = re.sub(r'```', '', raw).strip()
                    st.session_state["ms_r_code_pending"]  = raw
                    st.session_state["ms_r_code_original"] = existing_code
                    st.rerun()
                except Exception as e:
                    st.error(f"Enhancement failed: {e}")
