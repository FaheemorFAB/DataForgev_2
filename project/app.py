"""
DataForge — Flask Application
New in this version
───────────────────
  • SQLite database via Flask-SQLAlchemy (users / uploads / analyses tables)
  • Google OAuth 2.0 via Authlib
  • Flask-Login for session management
  • /login, /logout, /auth/google/callback routes
  • /dashboard route (login required)
  • DB logging on every upload / clean / eda / automl / query action
  • Auth is additive — workspace still works without login (DB logging skipped)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os, io, uuid, json, pickle, tempfile, traceback
from datetime import datetime
from pathlib import Path
from functools import wraps

import pandas as pd
import numpy as np
from flask import (Flask, render_template, request, jsonify, session,
                   send_file, redirect, url_for, Response, flash)
from dotenv import load_dotenv

load_dotenv(override=True)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "instance" / "dataforge.db"
DB_PATH.parent.mkdir(exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

from modules.models import db, User, Upload, Analysis
db.init_app(app)

# ── Flask-Login ────────────────────────────────────────────────────────────────
from flask_login import (LoginManager, login_user, logout_user,
                          login_required, current_user)

login_manager = LoginManager(app)
login_manager.login_view = "login_page"
login_manager.login_message = ""

# ── API calls return JSON 401 instead of HTML redirect ────────────────────────
@login_manager.unauthorized_handler
def unauthorized():
    if (request.path.startswith("/api/") or
            "application/json" in request.headers.get("Accept", "") or
            request.headers.get("Content-Type", "").startswith("application/json")):
        return jsonify({"error": "Authentication required", "redirect": "/login"}), 401
    return redirect(url_for("index") + "?login=1")

@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))

# ── Google OAuth via Authlib ───────────────────────────────────────────────────
from authlib.integrations.flask_client import OAuth

_GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
_GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_AUTH_ENABLED   = bool(_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET)

oauth = OAuth(app)
if GOOGLE_AUTH_ENABLED:
    # Hardcode Google endpoints — avoids a live HTTPS discovery request at
    # startup/request time that Windows Firewall often blocks.
    oauth.register(
        name="google",
        client_id=_GOOGLE_CLIENT_ID,
        client_secret=_GOOGLE_CLIENT_SECRET,
        # Static endpoints (stable — Google has not changed these in years)
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        access_token_url="https://oauth2.googleapis.com/token",
        jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
        userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
        client_kwargs={
            "scope": "openid email profile",
            "token_endpoint_auth_method": "client_secret_post",
        },
    )

# Temp storage dir for large objects (DataFrames, models)
STORE_DIR = Path(tempfile.gettempdir()) / "dataforge_store"
STORE_DIR.mkdir(exist_ok=True)

import math
PROJECTS_DIR = Path(__file__).parent / "instance" / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

def _persist(upload_id: int, key: str, obj):
    d = PROJECTS_DIR / str(upload_id); d.mkdir(exist_ok=True)
    with open(d / key, "wb") as f: pickle.dump(obj, f)

def _load_persisted(upload_id: int, key: str):
    p = PROJECTS_DIR / str(upload_id) / key
    if not p.exists(): return None
    with open(p, "rb") as f: return pickle.load(f)

def _project_meta(upload_id: int) -> dict:
    d = PROJECTS_DIR / str(upload_id)
    return {
        "has_raw":   (d / "df_raw").exists(),
        "has_clean": (d / "df_clean").exists(),
        "has_eda":   (d / "eda_html").exists(),
        "has_model": (d / "model_pkl").exists(),
        "has_chat":  (d / "chat_history").exists(),
    }

from flask.json.provider import DefaultJSONProvider
class _SafeJSON(DefaultJSONProvider):
    def dumps(self, obj, **kw):
        def _fix(o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
            if isinstance(o, dict):  return {k: _fix(v) for k, v in o.items()}
            if isinstance(o, list):  return [_fix(v) for v in o]
            return o
        return super().dumps(_fix(obj), **kw)
app.json_provider_class = _SafeJSON
app.json = _SafeJSON(app)

# ── Module imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from modules.data_cleaner   import run_cleaning_pipeline
from modules.eda_report     import generate_eda_report
from modules.automl_trainer import run_automl, _detect_task
from modules.gemini_pipeline import run_query_pipeline, is_available as gemini_available

with app.app_context():
    db.create_all()
    from sqlalchemy import text
    with db.engine.connect() as _c:
        for _s in [
            "ALTER TABLE uploads ADD COLUMN chat_history TEXT",
            "ALTER TABLE uploads ADD COLUMN clean_meta_json TEXT",
            "ALTER TABLE uploads ADD COLUMN automl_meta_json TEXT",
        ]:
            try: _c.execute(text(_s)); _c.commit()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HELPERS  — DataFrames too large for cookie; store on disk
# ══════════════════════════════════════════════════════════════════════════════
def _sid() -> str:
    if "store_id" not in session:
        session["store_id"] = uuid.uuid4().hex
    return session["store_id"]

def _path(key: str) -> Path:
    return STORE_DIR / f"{_sid()}_{key}"

def _save(key: str, obj):
    with open(_path(key), "wb") as f:
        pickle.dump(obj, f)

def _load(key: str):
    p = _path(key)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

def _clear_store():
    sid = _sid()
    for p in STORE_DIR.glob(f"{sid}_*"):
        p.unlink(missing_ok=True)

def _require_df(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _load("df_raw") is None:
            return jsonify({"error": "No dataset uploaded. Please upload a CSV first."}), 400
        return fn(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# DB LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _db_log_upload(profile: dict) -> int | None:
    """Save an upload record to DB. Returns upload.id or None."""
    if not current_user.is_authenticated:
        return None
    try:
        up = Upload(
            user_id     = current_user.id,
            filename    = profile.get("filename", ""),
            rows        = profile.get("rows", 0),
            cols        = profile.get("cols", 0),
            missing_pct = profile.get("missing_pct", 0.0),
        )
        db.session.add(up)
        db.session.commit()
        session["db_upload_id"] = up.id
        return up.id
    except Exception:
        db.session.rollback()
        return None


def _db_log_analysis(type_: str, summary: str = ""):
    """Save an analysis record to DB."""
    if not current_user.is_authenticated:
        return
    try:
        an = Analysis(
            user_id   = current_user.id,
            upload_id = session.get("db_upload_id"),
            type      = type_,
            summary   = summary,
        )
        db.session.add(an)
        db.session.commit()
    except Exception:
        db.session.rollback()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _df_profile(df: pd.DataFrame, filename: str = "") -> dict:
    missing     = int(df.isnull().sum().sum())
    numeric_cnt = int(len(df.select_dtypes(include=np.number).columns))
    total_cells = df.shape[0] * df.shape[1]
    miss_pct    = round(missing / max(total_cells, 1) * 100, 1)
    columns = []
    for col, dtype in zip(df.columns, df.dtypes):
        null_pct = round(df[col].isnull().mean() * 100, 1)
        columns.append({"name": col, "dtype": str(dtype),
                        "null_pct": null_pct, "quality": round(100 - null_pct, 1)})
    return {"filename": filename, "rows": df.shape[0], "cols": df.shape[1],
            "numeric": numeric_cnt, "missing": missing, "missing_pct": miss_pct,
            "columns": columns}

def _safe_json_value(v):
    if isinstance(v, (np.integer,)):   return int(v)
    if isinstance(v, (np.floating,)):  return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):        return bool(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if pd.isna(v):                     return None
    return v

def _df_to_json_rows(df: pd.DataFrame, limit: int = 500) -> dict:
    df = df.head(limit).replace([np.inf, -np.inf], None)
    headers = [str(c) for c in df.columns]
    rows = [[_safe_json_value(v) for v in row] for _, row in df.iterrows()]
    return {"headers": headers, "rows": rows, "total": len(df)}

def _time_ago(dt: datetime) -> str:
    diff = datetime.utcnow() - dt
    s = int(diff.total_seconds())
    if s < 60:    return "just now"
    if s < 3600:  return f"{s//60}m ago"
    if s < 86400: return f"{s//3600}h ago"
    if s < 604800: return f"{s//86400}d ago"
    return dt.strftime("%b %d")


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/login")
def login_page():
    # No standalone login page — redirect to landing which shows the modal
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("index") + "?login=1")


@app.route("/login/google")
def login_google():
    if not GOOGLE_AUTH_ENABLED:
        return redirect(url_for("index") + "?login=1")
    redirect_uri = url_for("auth_google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    if not GOOGLE_AUTH_ENABLED:
        return redirect(url_for("index") + "?login=1")
    try:
        token = oauth.google.authorize_access_token()
        # Without server_metadata_url, userinfo won't be in the token automatically.
        # Call the userinfo endpoint explicitly using the access token.
        resp     = oauth.google.get("https://openidconnect.googleapis.com/v1/userinfo")
        userinfo = resp.json()
    except Exception as e:
        app.logger.error(f"Google OAuth callback error: {e}")
        return redirect(url_for("index") + "?login=1")

    google_id = userinfo.get("sub")
    if not google_id:
        return redirect(url_for("index") + "?login=1")

    # Upsert user
    user = User.query.filter_by(google_id=google_id).first()
    if user is None:
        user = User(
            google_id = google_id,
            email     = userinfo.get("email"),
            name      = userinfo.get("name"),
            avatar    = userinfo.get("picture"),
        )
        db.session.add(user)
    else:
        user.name       = userinfo.get("name", user.name)
        user.avatar     = userinfo.get("picture", user.avatar)
        user.last_login = datetime.utcnow()

    db.session.commit()
    login_user(user, remember=True)
    return redirect(url_for("dashboard"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("upload.html", user=current_user, google_enabled=GOOGLE_AUTH_ENABLED)


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user

    # Recent uploads (last 10)
    recent_uploads = (
        Upload.query
        .filter_by(user_id=user.id)
        .order_by(Upload.uploaded_at.desc())
        .limit(10)
        .all()
    )

    # Recent analyses (last 20)
    recent_analyses = (
        Analysis.query
        .filter_by(user_id=user.id)
        .order_by(Analysis.created_at.desc())
        .limit(20)
        .all()
    )

    # Serialize for Jinja
    uploads_data = [
        {
            "id":          u.id,
            "filename":    u.filename,
            "rows":        u.rows,
            "cols":        u.cols,
            "missing_pct": u.missing_pct,
            "time_ago":    _time_ago(u.uploaded_at),
        }
        for u in recent_uploads
    ]

    analyses_data = [
        {
            "id":       a.id,
            "type":     a.type,
            "label":    a.label,
            "icon":     a.icon,
            "summary":  a.summary or "",
            "filename": a.upload.filename if a.upload else "",
            "time_ago": _time_ago(a.created_at),
        }
        for a in recent_analyses
    ]

    stats = {
        "uploads":  user.total_uploads,
        "analyses": user.total_analyses,
        "models":   user.total_models,
        "queries":  user.total_queries,
    }

    return render_template(
        "dashboard.html",
        user           = user,
        stats          = stats,
        recent_uploads = uploads_data,
        recent_analyses= analyses_data,
    )


@app.route("/workspace")
def workspace():
    profile = _load("profile")
    if not profile:
        return redirect(url_for("index"))
    return render_template(
        "workspace.html",
        profile   = profile,
        gemini_ok = gemini_available(),
        user      = current_user,
    )


# ══════════════════════════════════════════════════════════════════════════════
# API: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/upload", methods=["POST"])
def api_upload():
    # Require login for upload — return JSON 401 so the frontend shows the modal
    if not current_user.is_authenticated:
        return jsonify({"error": "login_required"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    _clear_store()
    _save("df_raw", df)
    profile = _df_profile(df, filename=f.filename)
    _save("profile", profile)
    session["filename"] = f.filename

    upload_id = _db_log_upload(profile)
    if upload_id:
        _persist(upload_id, "df_raw",  df)
        _persist(upload_id, "profile", profile)
    return jsonify({"ok": True, "profile": profile})


# ══════════════════════════════════════════════════════════════════════════════
# API: PROFILE + PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/profile")
@login_required
@_require_df
def api_profile():
    profile       = _load("profile") or {}
    clean_profile = _load("clean_profile")
    return jsonify({"raw": profile, "cleaned": clean_profile})


@app.route("/api/preview")
@login_required
@_require_df
def api_preview():
    use_clean = request.args.get("clean", "false").lower() == "true"
    df        = _load("df_clean") if use_clean else _load("df_raw")
    if df is None:
        df = _load("df_raw")
    limit = int(request.args.get("limit", 200))
    return jsonify(_df_to_json_rows(df, limit=limit))


# ══════════════════════════════════════════════════════════════════════════════
# API: AI QUERY
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/query", methods=["POST"])
@login_required
@_require_df
def api_query():
    body  = request.get_json(force=True)
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    df_clean = _load("df_clean")
    df_raw   = _load("df_raw")
    df       = df_clean if df_clean is not None else df_raw

    result = run_query_pipeline(query, df)
    _db_log_analysis("query", f"Query: {query[:80]}")
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
# API: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/clean", methods=["POST"])
@login_required
@_require_df
def api_clean():
    df_raw = _load("df_raw")
    try:
        result   = run_cleaning_pipeline(df_raw)
        df_clean = result["df_clean"]
        _save("df_clean", df_clean)
        clean_profile = _df_profile(df_clean, filename=session.get("filename", ""))
        _save("clean_profile", clean_profile)

        clean_meta = {
            "stats":          result["stats"],
            "missing_log":    result["missing_log"],
            "struct_actions": result["struct_actions"],
            "clean_profile":  clean_profile,
        }
        _save("clean_meta", clean_meta)

        # Save to DB (always works even if file persistence fails)
        uid = session.get("db_upload_id")
        if uid:
            try:
                _up = Upload.query.get(uid)
                if _up:
                    _up.clean_meta_json = json.dumps(clean_meta)
                    db.session.commit()
            except Exception: db.session.rollback()
            _persist(uid, "df_clean",      df_clean)
            _persist(uid, "clean_profile", clean_profile)
            _persist(uid, "clean_meta",    clean_meta)

        stats = result["stats"]
        _db_log_analysis(
            "clean",
            f"{stats['rows_removed']:,} rows removed · {stats['cols_removed']} cols removed"
        )

        return jsonify({
            "ok":             True,
            "stats":          stats,
            "missing_log":    result["missing_log"],
            "struct_actions": result["struct_actions"],
            "clean_profile":  clean_profile,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/clean/download")
@login_required
@_require_df
def api_clean_download():
    df = _load("df_clean")
    if df is None:
        return jsonify({"error": "Run cleaning first"}), 400
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    fname = "cleaned_" + session.get("filename", "data.csv")
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=fname)


# ══════════════════════════════════════════════════════════════════════════════
# API: EDA REPORT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/eda", methods=["POST"])
@login_required
@_require_df
def api_eda():
    body     = request.get_json(force=True) or {}
    minimal  = bool(body.get("minimal", True))
    sample_n = int(body.get("sample_n", 5000))

    df_clean = _load("df_clean")
    df_raw   = _load("df_raw")
    df       = df_clean if df_clean is not None else df_raw

    result = generate_eda_report(df, minimal=minimal, sample_n=sample_n)
    if result["error"]:
        return jsonify({"error": result["error"]}), 500

    _save("eda_html", result["html"])
    uid = session.get("db_upload_id")
    if uid: _persist(uid, "eda_html", result["html"])
    _db_log_analysis("eda", f"{result['rows_profiled']:,} rows profiled · minimal={minimal}")
    return jsonify({"ok": True, "rows_profiled": result["rows_profiled"]})


@app.route("/api/eda/report")
@login_required
def api_eda_report():
    html = _load("eda_html")
    if not html:
        return Response(
            """<!DOCTYPE html><html data-theme="dark"><head><style>
              html,body{background:#0A0A0B;color:#66666a;font-family:monospace;margin:0;
                display:flex;align-items:center;justify-content:center;height:100vh}
              </style></head><body><p>No EDA report yet — click "Generate Report".</p></body></html>
            """, mimetype="text/html", status=404)
    return Response(html, mimetype="text/html")


# ══════════════════════════════════════════════════════════════════════════════
# API: AUTOML
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/automl/detect-task", methods=["POST"])
@login_required
@_require_df
def api_automl_detect():
    body       = request.get_json(force=True) or {}
    target_col = body.get("target_col")
    df_clean   = _load("df_clean")
    df_raw     = _load("df_raw")
    df         = df_clean if df_clean is not None else df_raw

    if not target_col or target_col not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    task     = _detect_task(df[target_col])
    n_unique = int(df[target_col].nunique())
    return jsonify({"task": task, "n_unique": n_unique})


@app.route("/api/automl/train", methods=["POST"])
@login_required
@_require_df
def api_automl_train():
    body        = request.get_json(force=True) or {}
    target_col  = body.get("target_col")
    task_choice = body.get("task_choice", "auto-detect")
    time_budget = int(body.get("time_budget", 120))
    test_size   = int(body.get("test_size", 20)) / 100.0

    df_clean = _load("df_clean")
    df_raw   = _load("df_raw")
    df       = df_clean if df_clean is not None else df_raw

    if not target_col or target_col not in df.columns:
        return jsonify({"error": f"Target column '{target_col}' not found"}), 400

    try:
        result = run_automl(df, target_col, task_choice, time_budget, test_size)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    if result.get("error"):
        return jsonify({"error": result["error"]}), 500

    model_pkl = result.pop("model_pkl", None)
    if model_pkl:
        _save("model_pkl",      model_pkl)
        _save("best_estimator", result["best_estimator"])
        _save("automl_meta",    result)

        uid = session.get("db_upload_id")
        if uid:
            try:
                _up = Upload.query.get(uid)
                if _up:
                    _up.automl_meta_json = json.dumps(result)
                    db.session.commit()
            except Exception: db.session.rollback()
            _persist(uid, "model_pkl",      model_pkl)
            _persist(uid, "best_estimator", result["best_estimator"])
            _persist(uid, "automl_meta",    result)

    # Build summary: best model + top metric
    metrics     = result.get("metrics", {})
    top_metric  = next(iter(metrics.items()), ("", ""))
    summary     = (f"{result['best_estimator']} · "
                   f"{top_metric[0]}={top_metric[1]} · "
                   f"target={target_col}")
    _db_log_analysis("automl", summary)

    return jsonify({"ok": True, **result})


@app.route("/api/automl/download")
@login_required
def api_automl_download():
    model_pkl = _load("model_pkl")
    if not model_pkl:
        return jsonify({"error": "No trained model available"}), 400
    best_estimator = _load("best_estimator") or "model"
    buf = io.BytesIO(model_pkl)
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream",
                     as_attachment=True,
                     download_name=f"best_model_{best_estimator}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# API: USER PROFILE UPDATE
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/user/update", methods=["POST"])
@login_required
def api_user_update():
    """Update the current user's display name and/or avatar URL."""
    data = request.get_json(silent=True) or {}
    user = current_user

    new_name   = (data.get("name")   or "").strip()
    new_avatar = (data.get("avatar") or "").strip()

    if new_name:
        user.name = new_name[:256]
    if new_avatar:
        # Accept data-URIs (base64 uploads) or plain URLs
        user.avatar = new_avatar[:2048]

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok":     True,
        "name":   user.name,
        "avatar": user.avatar,
    })


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PROJECTS
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/projects")
@login_required
def projects_page():
    return render_template("projects.html", user=current_user)


@app.route("/api/projects")
@login_required
def api_projects():
    uploads = (Upload.query.filter_by(user_id=current_user.id)
               .order_by(Upload.uploaded_at.desc()).limit(50).all())
    result = []
    for u in uploads:
        meta = _project_meta(u.id)
        # has_clean / has_model also true if stored in DB
        if not meta["has_clean"] and u.clean_meta_json:  meta["has_clean"] = True
        if not meta["has_model"] and u.automl_meta_json: meta["has_model"] = True
        if not meta["has_chat"]  and u.chat_history:     meta["has_chat"]  = True
        analyses = (Analysis.query.filter_by(upload_id=u.id)
                    .order_by(Analysis.created_at.desc()).all())
        result.append({
            "id": u.id, "filename": u.filename,
            "rows": u.rows, "cols": u.cols,
            "missing_pct": u.missing_pct,
            "time_ago": _time_ago(u.uploaded_at),
            "analyses": [{"type": a.type, "label": a.label,
                          "summary": a.summary or "",
                          "time_ago": _time_ago(a.created_at)} for a in analyses],
            **meta,
        })
    return jsonify(result)


@app.route("/api/restore/<int:upload_id>", methods=["POST"])
@login_required
def api_restore(upload_id: int):
    upload = Upload.query.filter_by(id=upload_id, user_id=current_user.id).first()
    if not upload:
        return jsonify({"error": "Project not found"}), 404

    _clear_store()

    # Restore binary files to session
    for key in ["df_raw", "df_clean", "profile", "clean_profile",
                "eda_html", "model_pkl", "best_estimator",
                "clean_meta", "automl_meta"]:
        obj = _load_persisted(upload_id, key)
        if obj is not None:
            _save(key, obj)

    session["filename"]     = upload.filename
    session["db_upload_id"] = upload_id

    # ── Load result dicts: DB is authoritative, files are fallback ────────────
    clean_meta = None
    if upload.clean_meta_json:
        try: clean_meta = json.loads(upload.clean_meta_json)
        except Exception: pass
    if clean_meta is None:
        clean_meta = _load("clean_meta")  # from persisted file loaded above

    automl_meta = None
    if upload.automl_meta_json:
        try: automl_meta = json.loads(upload.automl_meta_json)
        except Exception: pass
    if automl_meta is None:
        automl_meta = _load("automl_meta")

    chat_history = []
    if upload.chat_history:
        try: chat_history = json.loads(upload.chat_history)
        except Exception: pass

    meta = _project_meta(upload_id)
    # Supplement meta flags from DB
    if not meta["has_clean"] and clean_meta:   meta["has_clean"] = True
    if not meta["has_model"] and automl_meta:  meta["has_model"] = True
    if not meta["has_eda"]:
        meta["has_eda"] = (PROJECTS_DIR / str(upload_id) / "eda_html").exists()

    profile = _load("profile") or {}

    return jsonify({
        "ok":           True,
        "profile":      profile,
        "chat_history": chat_history,
        "clean_meta":   clean_meta,
        "automl_meta":  automl_meta,
        **meta,
    })


@app.route("/api/chat/save", methods=["POST"])
@login_required
def api_chat_save():
    data      = request.get_json(force=True) or {}
    messages  = data.get("messages", [])
    upload_id = session.get("db_upload_id")
    if not upload_id:
        return jsonify({"ok": False})
    upload = Upload.query.filter_by(id=upload_id, user_id=current_user.id).first()
    if not upload:
        return jsonify({"ok": False})
    try:
        upload.chat_history = json.dumps(messages)
        db.session.commit()
        _persist(upload_id, "chat_history", json.dumps(messages).encode())
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)