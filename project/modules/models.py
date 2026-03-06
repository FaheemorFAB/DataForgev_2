"""
Database models for DataForge.
Tables: users, uploads, analyses
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id         = db.Column(db.Integer, primary_key=True)
    google_id  = db.Column(db.String(128), unique=True, nullable=False)
    email      = db.Column(db.String(256), unique=True, nullable=True)
    name       = db.Column(db.String(256), nullable=True)
    avatar     = db.Column(db.String(512), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

    uploads  = db.relationship("Upload",   backref="user", lazy="dynamic")
    analyses = db.relationship("Analysis", backref="user", lazy="dynamic")

    # ── aggregate helpers ──────────────────────────────────────────────────────
    @property
    def total_uploads(self) -> int:
        return self.uploads.count()

    @property
    def total_analyses(self) -> int:
        return self.analyses.count()

    @property
    def total_models(self) -> int:
        return self.analyses.filter_by(type="automl").count()

    @property
    def total_queries(self) -> int:
        return self.analyses.filter_by(type="query").count()


class Upload(db.Model):
    __tablename__ = "uploads"

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename    = db.Column(db.String(256))
    rows        = db.Column(db.Integer)
    cols        = db.Column(db.Integer)
    missing_pct = db.Column(db.Float, default=0.0)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    analyses        = db.relationship("Analysis", backref="upload", lazy="dynamic")
    chat_history    = db.Column(db.Text, nullable=True)   # JSON [{role,content,...}]
    clean_meta_json = db.Column(db.Text, nullable=True)   # JSON clean result dict
    automl_meta_json= db.Column(db.Text, nullable=True)   # JSON automl result dict


class Analysis(db.Model):
    __tablename__ = "analyses"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"),  nullable=False)
    upload_id  = db.Column(db.Integer, db.ForeignKey("uploads.id"), nullable=True)
    type       = db.Column(db.String(64))   # eda | automl | clean | query
    summary    = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ── display helpers ────────────────────────────────────────────────────────
    TYPE_LABELS = {
        "eda":    ("EDA Report",    ""),
        "automl": ("AutoML",        ""),
        "clean":  ("Data Cleaning", ""),
        "query":  ("AI Query",      ""),
    }

    @property
    def label(self) -> str:
        entry = self.TYPE_LABELS.get(self.type or "")
        if isinstance(entry, tuple) and len(entry) >= 1:
            return entry[0]
        return str(self.type or "Unknown")

    @property
    def icon(self) -> str:
        entry = self.TYPE_LABELS.get(self.type or "")
        if isinstance(entry, tuple) and len(entry) >= 2:
            return entry[1]
        return ""