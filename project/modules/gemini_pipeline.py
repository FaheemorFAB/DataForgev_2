"""
DataForge — Gemini Query Pipeline  [NEW SINGLE-CALL VERSION — NO LANGCHAIN]
Replace: project/modules/gemini_pipeline.py
"""
import os, sys, re, json, time, logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_HERE, _PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# Startup confirmation — visible in Flask console
print("[gemini_pipeline] NEW single-call pipeline loaded (no LangChain)", flush=True)

_MODEL_FALLBACKS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
_GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_OK        = bool(_GEMINI_API_KEY)
MAX_CODE_RETRIES = 2
SAMPLE_ROWS      = 6


def is_available() -> bool:
    return GEMINI_OK


# ── Direct REST call to Gemini — no LangChain ────────────────────────────────

def _gemini_call(prompt: str, model: str, temperature: float = 0.1, timeout: int = 25) -> str:
    import urllib.request, urllib.error
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={_GEMINI_API_KEY}")
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} from Gemini ({model}): {body[:300]}") from e


def _gemini(prompt: str, temperature: float = 0.1, timeout: int = 25) -> str:
    """Try each model in fallback order, log errors clearly."""
    last_err = None
    for model in _MODEL_FALLBACKS:
        try:
            result = _gemini_call(prompt, model=model, temperature=temperature, timeout=timeout)
            logger.info("[gemini_pipeline] success with model=%s", model)
            return result
        except Exception as exc:
            last_err = exc
            logger.warning("[gemini_pipeline] model %s failed: %s", model, exc)
            if "429" in str(exc) or "quota" in str(exc).lower():
                time.sleep(3)
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


# ── Schema summary for prompt ─────────────────────────────────────────────────

def _schema(df: pd.DataFrame) -> str:
    rows = []
    for col in df.columns:
        pct    = round(df[col].isnull().mean() * 100, 1)
        sample = df[col].dropna().head(3).tolist()
        rows.append(f"  {col!r}: {df[col].dtype}, {pct}% null, sample={sample}")
    return "\n".join(rows)


# ── Single Gemini call: returns structured JSON ───────────────────────────────

def _ask_gemini(query: str, df: pd.DataFrame) -> dict:
    col_list = ", ".join(f'"{c}"' for c in df.columns)
    prompt = f"""You are a senior data analyst. Answer the user query about a pandas DataFrame called `df`.

DATAFRAME: {df.shape[0]:,} rows x {df.shape[1]} columns
COLUMNS:
{_schema(df)}

SAMPLE DATA ({SAMPLE_ROWS} rows):
{df.head(SAMPLE_ROWS).to_string(index=False, max_cols=20)}

USER QUERY: "{query}"

TASK:
1. Write pandas/numpy code that answers the query. Assign the final answer to `result`.
2. Write a clear 2-5 sentence plain-text answer with specific numbers from the data.
3. Pick the best visualization: bar_chart, line_chart, scatter_chart, histogram, table, metric, or summary.

RULES:
- Use only df, pd, np — no imports, no print(), no plt
- Always assign to result
- Plain text answer only — no markdown, no asterisks, no bullet points
- Return ONLY valid JSON — no markdown fences, no extra text

JSON FORMAT:
{{
  "code": "result = ...",
  "answer": "plain text answer with numbers",
  "intent": "bar_chart|line_chart|scatter_chart|histogram|table|metric|summary",
  "x_col": "column name or null",
  "y_col": "column name or null",
  "top_n": 10
}}

Available columns: {col_list}"""

    raw  = _gemini(prompt, temperature=0.1, timeout=22)
    text = raw.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$",       "", text, flags=re.MULTILINE).strip()
    # Extract first JSON object
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e <= s:
        raise ValueError(f"No JSON found in Gemini response: {text[:200]}")
    return json.loads(text[s:e])


# ── Execute pandas code ───────────────────────────────────────────────────────

def _run_code(code: str, df: pd.DataFrame):
    ns = {"df": df.copy(), "pd": pd, "np": np, "result": None}
    exec(_clean_code(code), {}, ns)  # noqa: S102
    return ns.get("result")


# ── Build chart/table/metric payload ─────────────────────────────────────────

def _build_chart(intent, x_col, y_col, top_n, raw, df):
    try:
        if intent == "scatter_chart":
            nc = df.select_dtypes(include=np.number).columns.tolist()
            x  = (x_col if x_col and x_col in df.columns else nc[0] if nc else None)
            y  = (y_col if y_col and y_col in df.columns else nc[1] if len(nc) > 1 else None)
            if x and y:
                pts = [{"x": (float(r[x]) if pd.notna(r[x]) else None),
                        "y": (float(r[y]) if pd.notna(r[y]) else None)}
                       for r in df[[x, y]].dropna().head(500).to_dict("records")]
                return {"type": "scatter_chart", "points": pts, "x_label": x, "y_label": y}

        if intent in ("bar_chart", "histogram", "line_chart"):
            if isinstance(raw, pd.Series):
                series, lx, ly = raw, str(raw.index.name or x_col or "category"), str(y_col or "value")
            elif isinstance(raw, pd.DataFrame) and raw.shape[1] >= 2:
                series = raw.iloc[:, 1]
                series.index = raw.iloc[:, 0].values
                lx, ly = str(raw.columns[0]), str(raw.columns[1])
            elif x_col and x_col in df.columns:
                if y_col and y_col in df.columns:
                    series = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                else:
                    series = df[x_col].value_counts()
                lx, ly = x_col, (y_col or "count")
            else:
                return None
            if top_n:
                series = series.head(int(top_n))
            t = "line_chart" if intent == "line_chart" else "bar_chart"
            return {"type": t,
                    "labels":  [str(i) for i in series.index],
                    "values":  [float(v) if pd.notna(v) else 0 for v in series],
                    "x_label": lx, "y_label": ly}

        if intent == "table":
            tbl = raw if isinstance(raw, pd.DataFrame) else df.head(50)
            return {"type": "table", "headers": tbl.columns.tolist(),
                    "rows": _safe_rows(tbl.head(100).to_dict("records")), "total": len(tbl)}

        if intent == "metric":
            val = raw
            if isinstance(val, (pd.Series, pd.DataFrame)):
                val = val.iloc[0] if len(val) else None
            if val is not None:
                try:
                    return {"type": "metric", "value": round(float(val), 4)}
                except (TypeError, ValueError):
                    pass
    except Exception as exc:
        logger.warning("[gemini_pipeline] chart build error: %s", exc)
    return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_query_pipeline(query: str, df: pd.DataFrame) -> dict:
    """Gemini-first, deterministic fallback. Single API call, ~2-4s."""

    if GEMINI_OK:
        try:
            gd     = _ask_gemini(query, df)
            code   = gd.get("code", "")
            answer = _strip(gd.get("answer") or "")
            intent = gd.get("intent", "summary")
            x_col  = gd.get("x_col")
            y_col  = gd.get("y_col")
            top_n  = gd.get("top_n")

            raw = None
            if code:
                for attempt in range(MAX_CODE_RETRIES + 1):
                    try:
                        raw = _run_code(code, df)
                        break
                    except Exception as err:
                        if attempt < MAX_CODE_RETRIES:
                            logger.warning("[gemini_pipeline] code attempt %d failed: %s", attempt+1, err)
                            fix = _gemini(
                                f"Fix this pandas code:\n```python\n{code}\n```\n"
                                f"Error: {err}\nColumns available: {list(df.columns)}\n"
                                "Return ONLY corrected Python code, no explanation.", timeout=12)
                            code = _clean_code(fix)
                        else:
                            logger.warning("[gemini_pipeline] code exec failed after retries: %s", err)

            if not answer and raw is not None:
                answer = str(raw)[:500]

            result = None
            if intent != "summary":
                result = _build_chart(intent, x_col, y_col, top_n, raw, df)
            if result is None:
                result = {"type": "summary", "text": answer or "No result."}

            return {
                "error":   None,
                "answer":  answer or "No result.",
                "result":  result,
                "insight": "",
                "intent":  {"type": intent},
                "engine":  "llm",
            }

        except Exception as exc:
            logger.error("[gemini_pipeline] Gemini failed, falling back: %s", exc, exc_info=True)

    # Deterministic fallback
    try:
        det = _load_det()
        dr  = det.run_deterministic_pipeline(query, df)
        if not dr.get("error"):
            ri = dr.get("intent", "summary")
            return {"error": None, "answer": dr["answer"], "result": dr["result"],
                    "insight": dr.get("insight", ""),
                    "intent": {"type": ri} if isinstance(ri, str) else ri,
                    "engine": "deterministic"}
    except Exception as e:
        logger.error("[gemini_pipeline] deterministic failed: %s", e)

    return {"error": "Query failed. Check GEMINI_API_KEY in your .env file."}


# ── Deterministic engine loader ───────────────────────────────────────────────

_DET = None

def _load_det():
    global _DET
    if _DET: return _DET
    import importlib.util as ilu
    try:
        import deterministic_engine as m
        _DET = m; return m
    except ModuleNotFoundError:
        pass
    for d in [_HERE, _PARENT, os.getcwd(), os.path.join(os.getcwd(), "modules")]:
        p = os.path.join(d, "deterministic_engine.py")
        if os.path.isfile(p):
            spec = ilu.spec_from_file_location("deterministic_engine", p)
            mod  = ilu.module_from_spec(spec); sys.modules["deterministic_engine"] = mod
            spec.loader.exec_module(mod); _DET = mod; return mod
    raise ImportError("deterministic_engine.py not found")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_rows(rows):
    out = []
    for row in rows:
        r = {}
        for k, v in row.items():
            if isinstance(v, np.integer):    v = int(v)
            elif isinstance(v, np.floating): v = None if np.isnan(v) else float(v)
            elif isinstance(v, np.bool_):    v = bool(v)
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)): v = None
            r[str(k)] = v
        out.append(r)
    return out

def _clean_code(t: str) -> str:
    if not t: return ""
    t = re.sub(r"^```(?:python)?\s*", "", t.strip(), flags=re.MULTILINE)
    return re.sub(r"\s*```\s*$", "", t.strip()).strip()

def _strip(text: str) -> str:
    if not text: return text
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`",      r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*",  r"\1", text)
    text = re.sub(r"^#{1,6}\s+",     "",    text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[*\-]\s+",   "",    text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# Alias used in some older imports
strip_markdown = _strip