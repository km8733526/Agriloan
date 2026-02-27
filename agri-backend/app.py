"""
AgriLoan Credit Scorer — Flask REST API
"""

from __future__ import annotations

import os
import sys
import pickle
import traceback
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Setup paths
# ─────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from database.init_db import (
    init_db,
    get_connection,
    row_to_dict,
    DB_PATH,
)

from services.soil_service import get_soil_data
from services.weather_service import get_weather_data
from services.scoring_service import calculate_trust_score
from services.feature_engineering import (
    build_feature_vector,
    get_feature_summary,
)

# ─────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────

_MODEL_PATH = os.path.join(_BASE_DIR, "model", "credit_model.pkl")
_model_cache = None


# ─────────────────────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)

    # ✅ Proper CORS handling
    CORS(app, supports_credentials=True)

    app.config["SECRET_KEY"] = os.getenv(
        "AGRILOAN_SECRET", os.urandom(24).hex()
    )

    init_db()

    register_routes(app)
    register_errors(app)

    return app


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_db():
    if "db" not in g:
        g.db = get_connection()
    return g.db


def _utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_json(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 415
        return f(*args, **kwargs)
    return wrapper


def _load_model():
    global _model_cache

    if _model_cache is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError("Model not trained.")
        with open(_MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)

    return _model_cache


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

def register_routes(app: Flask):

    @app.route("/health")
    def health():
        return jsonify({
            "status": "running",
            "timestamp": _utcnow()
        })

    @app.route("/evaluate", methods=["POST"])
    @_require_json
    def evaluate():
        try:
            data = request.get_json()

            # Basic validation
            required = [
                "fullName", "aadhaar", "phone",
                "district", "village",
                "surveyNumber", "landSize",
                "irrigation", "ownership",
                "primaryCrop", "cropDiversity",
                "latitude", "longitude"
            ]

            missing = [f for f in required if not data.get(f)]
            if missing:
                return jsonify({"error": f"Missing: {', '.join(missing)}"}), 422

            lat = float(data["latitude"])
            lng = float(data["longitude"])

            soil = get_soil_data(lat, lng)
            weather = get_weather_data(lat, lng)

            features = build_feature_vector(data, soil, weather)

            model = _load_model()
            prob = float(model.predict_proba(features)[0][1])

            scoring = calculate_trust_score(prob)

            db = _get_db()
            with db:
                farmer_id = db.execute(
                    """INSERT INTO farmers 
                       (full_name, aadhaar, phone, district, village)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        data["fullName"],
                        data["aadhaar"],
                        data["phone"],
                        data["district"],
                        data["village"],
                    ),
                ).lastrowid

            return jsonify({
                "trust_score": scoring["trust_score"],
                "risk_level": scoring["risk_level"],
                "status": scoring["status"],
                "repayment_probability": scoring["repayment_probability"],
                "evaluated_at": _utcnow()
            }), 201

        except Exception:
            return jsonify({
                "error": "Server error",
                "details": traceback.format_exc()
            }), 500


# ─────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────

def register_errors(app: Flask):

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Route not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("AGRILOAN_PORT", 5000))
    app.run(host="0.0.0.0", port=port)
