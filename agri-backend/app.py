"""
app.py
------
AgriLoan Credit Scorer — Flask REST API (Production-ready)

Endpoints:
    GET  /health                  Liveness / readiness probe
    POST /evaluate                Submit loan application + get ML evaluation
    GET  /applications            List all applications (dashboard)
    GET  /applications/<int:id>   Get one application detail (result page)

CORS:
    All origins allowed via manual headers (flask-cors installed separately).
    In production, replace "*" with your frontend domain.

Startup:
    Development :  python app.py
    Production  :  gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"

Environment variables (optional):
    AGRILOAN_PORT      – listening port (default: 5000)
    AGRILOAN_DEBUG     – set to "1" to enable Flask debug mode
    AGRILOAN_SECRET    – Flask secret key (auto-generated if not set)
"""

from __future__ import annotations

import os
import sys
import pickle
import traceback
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify, g

from dotenv import load_dotenv
load_dotenv()

# ── Ensure project root is importable from any CWD ───────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from database.init_db import (init_db, get_connection,
                               row_to_dict, rows_to_list, DB_PATH)
from services.soil_service        import get_soil_data
from services.weather_service     import get_weather_data
from services.scoring_service     import calculate_trust_score
from services.feature_engineering import build_feature_vector, get_feature_summary

# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_PATH  = os.path.join(_BASE_DIR, "model", "credit_model.pkl")
_model_cache = None   # loaded once on first use


def create_app() -> Flask:
    """Application factory. Called by gunicorn and tests."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.config["SECRET_KEY"]     = os.getenv("AGRILOAN_SECRET", os.urandom(24).hex())

    # Initialise database tables at startup
    try:
        init_db()
    except Exception as exc:
        app.logger.critical("Database init failed: %s", exc)
        raise

    # Register blueprints / routes
    _register_routes(app)
    _register_error_handlers(app)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# CORS middleware (manual – works without flask-cors package installed)
# ─────────────────────────────────────────────────────────────────────────────

def _add_cors(app: Flask) -> None:
    @app.after_request
    def _cors_headers(response):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-Requested-With"
        )
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Max-Age"]       = "86400"
        return response

    @app.before_request
    def _preflight():
        if request.method == "OPTIONS":
            return jsonify({}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Request-scoped DB connection
# ─────────────────────────────────────────────────────────────────────────────

def _get_db():
    """Return (and cache per request) a SQLite connection."""
    if "db" not in g:
        g.db = get_connection()
    return g.db


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    """Load the trained sklearn Pipeline from disk. Cached after first call."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {_MODEL_PATH}. "
                "Run `python train_model.py` first to train and save the model."
            )
        with open(_MODEL_PATH, "rb") as fh:
            _model_cache = pickle.load(fh)
    return _model_cache


# ─────────────────────────────────────────────────────────────────────────────
# Decorators / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_json(f):
    """Decorator: reject requests whose Content-Type is not application/json."""
    @wraps(f)
    def _wrapper(*args, **kwargs):
        if not request.is_json:
            return _err("Request Content-Type must be application/json", 415)
        return f(*args, **kwargs)
    return _wrapper


def _err(message: str, code: int = 400):
    """Return a standardised JSON error response."""
    return jsonify({
        "error":     message,
        "status":    code,
        "timestamp": _utcnow(),
    }), code


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt_app_id(db_id: int) -> str:
    """Format a DB integer ID as the display application ID."""
    return f"#AL{db_id:07d}"


# ─────────────────────────────────────────────────────────────────────────────
# Database write helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_farmer(db, data: dict) -> int:
    """Insert a farmer record and return the new ID."""
    cur = db.execute(
        """INSERT INTO farmers (full_name, aadhaar, phone, district, village)
           VALUES (?, ?, ?, ?, ?)""",
        (
            data["fullName"],
            data["aadhaar"],
            data["phone"],
            data["district"],
            data["village"],
        ),
    )
    return cur.lastrowid


def _save_land(db, farmer_id: int, data: dict) -> int:
    """Insert a land_details record and return the new ID."""
    cur = db.execute(
        """INSERT INTO land_details
               (farmer_id, survey_number, land_size, irrigation,
                ownership, latitude, longitude)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            farmer_id,
            data["surveyNumber"],
            float(data["landSize"]),
            data["irrigation"],
            data["ownership"],
            float(data["latitude"]),
            float(data["longitude"]),
        ),
    )
    return cur.lastrowid


def _save_application(db, farmer_id: int, data: dict,
                      soil: dict, weather: dict, scoring: dict) -> int:
    """Insert a loan_applications record and return the new ID."""
    cur = db.execute(
        """INSERT INTO loan_applications
               (farmer_id, primary_crop, crop_diversity,
                yield_1, yield_2, yield_3,
                trust_score, risk_level, repayment_probability, status,
                soil_type, soil_ph, annual_rainfall, drought_risk)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            farmer_id,
            data["primaryCrop"],
            int(data.get("cropDiversity") or 1),
            _safe_float(data.get("yield1")),
            _safe_float(data.get("yield2")),
            _safe_float(data.get("yield3")),
            scoring["trust_score"],
            scoring["risk_level"],
            scoring["repayment_probability"],
            scoring["status"],
            soil.get("soil_type"),
            soil.get("soil_ph"),
            weather.get("annual_rainfall"),
            weather.get("drought_risk"),
        ),
    )
    return cur.lastrowid


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for empty/zero values."""
    if val in (None, "", "0", 0):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Route registration
# ─────────────────────────────────────────────────────────────────────────────

def _register_routes(app: Flask) -> None:

    # Apply CORS to every response
    _add_cors(app)

    # Tear down DB connection after each request
    @app.teardown_appcontext
    def _close_db(exc=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    # ── Try to install flask-cors if available ────────────────────────────
    try:
        from flask_cors import CORS
        CORS(app, origins="*")
        app.logger.info("flask-cors enabled")
    except ImportError:
        app.logger.info("flask-cors not installed – using manual CORS headers")

    # =========================================================================
    # GET /health
    # =========================================================================
    @app.route("/health", methods=["GET"])
    def health():
        """
        Liveness + readiness probe.

        Returns 200 if the API is running, DB exists, and model file exists.
        Returns 503 if model has not been trained yet.
        """
        model_ready = os.path.exists(_MODEL_PATH)
        db_ready    = os.path.exists(DB_PATH)

        body = {
            "status":      "running",
            "timestamp":   _utcnow(),
            "db_ready":    db_ready,
            "model_ready": model_ready,
            "db_path":     DB_PATH,
            "model_path":  _MODEL_PATH,
        }

        if not model_ready:
            body["message"] = "Model not trained. Run `python train_model.py` first."
            return jsonify(body), 503

        return jsonify(body), 200

    # =========================================================================
    # POST /evaluate
    # =========================================================================
    @app.route("/evaluate", methods=["POST", "OPTIONS"])
    @_require_json
    def evaluate():
        """
        Accept a complete loan application JSON, run the full evaluation
        pipeline, persist results, and return the credit decision.

        Expected JSON body fields:
            fullName, aadhaar, phone, district, village,
            surveyNumber, landSize, irrigation, ownership,
            primaryCrop, cropDiversity,
            yield1, yield2, yield3,
            latitude, longitude

        Returns 201 with credit evaluation on success.
        Returns 422 if required fields are missing or invalid.
        Returns 503 if the model file has not been trained.
        Returns 500 for unexpected server errors.
        """
        data = request.get_json(silent=True)
        if not data:
            return _err("Request body is empty or not valid JSON", 400)

        # ── Validate required fields ──────────────────────────────────────
        required_fields = [
            "fullName", "aadhaar", "phone", "district", "village",
            "surveyNumber", "landSize", "irrigation", "ownership",
            "primaryCrop", "cropDiversity", "latitude", "longitude",
        ]
        missing = [f for f in required_fields if not str(data.get(f, "")).strip()]
        if missing:
            return _err(f"Missing or empty required fields: {', '.join(missing)}", 422)

        # ── Validate numeric coordinates ──────────────────────────────────
        try:
            lat = float(data["latitude"])
            lng = float(data["longitude"])
        except (ValueError, TypeError):
            return _err("'latitude' and 'longitude' must be valid numbers", 422)

        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return _err("Coordinates out of valid range", 422)

        app.logger.info("Evaluating application for: %s", data.get("fullName"))

        # ── Step 1: Fetch soil and weather enrichment ─────────────────────
        try:
            soil_data    = get_soil_data(lat, lng)
            weather_data = get_weather_data(lat, lng)
        except Exception as exc:
            app.logger.error("Enrichment error: %s", exc)
            return _err("Failed to fetch soil/weather data", 500)

        # ── Step 2: Build ML feature vector ──────────────────────────────
        try:
            feature_vec = build_feature_vector(data, soil_data, weather_data)
        except Exception as exc:
            app.logger.error("Feature engineering error: %s", exc)
            return _err("Feature engineering failed", 500)

        # ── Step 3: Load model and predict repayment probability ──────────
        try:
            model      = _load_model()
            prob_array = model.predict_proba(feature_vec)
            # Index 1 = probability of repayment (positive class)
            repayment_prob = float(prob_array[0][1])
        except FileNotFoundError as exc:
            return _err(str(exc), 503)
        except Exception as exc:
            app.logger.error("Model prediction error:\n%s", traceback.format_exc())
            return _err("Model prediction failed", 500)

        # ── Step 4: Calculate trust score and risk decision ───────────────
        scoring = calculate_trust_score(repayment_prob)

        # ── Step 5: Persist everything to SQLite ──────────────────────────
        try:
            db = _get_db()
            with db:                              # auto-commit / rollback
                farmer_id      = _save_farmer(db, data)
                _save_land(db, farmer_id, data)
                application_id = _save_application(
                    db, farmer_id, data, soil_data, weather_data, scoring
                )
        except Exception as exc:
            app.logger.error("DB write error:\n%s", traceback.format_exc())
            return _err("Database write failed", 500)

        app.logger.info(
            "Application %s evaluated: score=%d status=%s",
            _fmt_app_id(application_id),
            scoring["trust_score"],
            scoring["status"],
        )

        # ── Step 6: Build response ────────────────────────────────────────
        return jsonify({
            "application_id":        _fmt_app_id(application_id),
            "db_id":                 application_id,
            "trust_score":           scoring["trust_score"],
            "risk_level":            scoring["risk_level"],
            "status":                scoring["status"],
            "repayment_probability": scoring["repayment_probability"],
            "loan_terms":            scoring["loan_terms"],
            "emi_monthly":           scoring["emi_monthly"],
            "soil_summary": {
                "soil_type":        soil_data["soil_type"],
                "soil_ph":          soil_data["soil_ph"],
                "nitrogen_level":   soil_data["nitrogen_level"],
                "phosphorus_level": soil_data["phosphorus_level"],
                "potassium_level":  soil_data["potassium_level"],
                "organic_carbon":   soil_data["organic_carbon"],
            },
            "weather_summary": {
                "annual_rainfall":     weather_data["annual_rainfall"],
                "temperature_avg":     weather_data["temperature_avg"],
                "temperature_min":     weather_data["temperature_min"],
                "temperature_max":     weather_data["temperature_max"],
                "humidity":            weather_data["humidity"],
                "drought_risk":        weather_data["drought_risk"],
                "monsoon_reliability": weather_data["monsoon_reliability"],
            },
            "features_used":         get_feature_summary(data, soil_data, weather_data),
            "evaluated_at":          _utcnow(),
        }), 201

    # =========================================================================
    # GET /applications
    # =========================================================================
    @app.route("/applications", methods=["GET"])
    def list_applications():
        """
        Return all loan applications, newest first.

        Optional query parameters:
            status   – filter by 'Approved' | 'Review' | 'Rejected'
            district – filter by district name
            page     – page number (1-based, default 1)
            per_page – records per page (default 50, max 200)
            search   – fuzzy search on farmer full_name

        Returns:
            200 with JSON:
                {
                  "applications": [...],
                  "total": <int>,
                  "page":  <int>,
                  "per_page": <int>,
                  "pages": <int>
                }
        """
        # ── Query parameters ──────────────────────────────────────────────
        status_filter   = request.args.get("status",   "").strip()
        district_filter = request.args.get("district", "").strip()
        search_term     = request.args.get("search",   "").strip()
        try:
            page     = max(1, int(request.args.get("page",     1)))
            per_page = min(200, max(1, int(request.args.get("per_page", 50))))
        except ValueError:
            return _err("'page' and 'per_page' must be integers", 422)

        offset = (page - 1) * per_page

        # ── Build dynamic WHERE clause ────────────────────────────────────
        where_clauses: list[str] = []
        params: list             = []

        if status_filter:
            where_clauses.append("la.status = ?")
            params.append(status_filter)
        if district_filter:
            where_clauses.append("LOWER(f.district) = LOWER(?)")
            params.append(district_filter)
        if search_term:
            where_clauses.append("LOWER(f.full_name) LIKE LOWER(?)")
            params.append(f"%{search_term}%")

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # ── Query ─────────────────────────────────────────────────────────
        try:
            db = _get_db()

            # Total count (for pagination)
            count_sql = f"""
                SELECT COUNT(*) AS cnt
                FROM loan_applications la
                JOIN farmers f ON f.id = la.farmer_id
                {where_sql}
            """
            total = db.execute(count_sql, params).fetchone()["cnt"]

            # Paginated results
            data_sql = f"""
                SELECT
                    la.id                    AS db_id,
                    la.id                    AS application_id_raw,
                    f.full_name              AS farmer_name,
                    f.district,
                    f.village,
                    f.phone,
                    ld.land_size,
                    ld.irrigation,
                    ld.ownership,
                    ld.latitude,
                    ld.longitude,
                    la.primary_crop,
                    la.crop_diversity,
                    la.yield_1,
                    la.yield_2,
                    la.yield_3,
                    la.trust_score,
                    la.risk_level,
                    la.repayment_probability,
                    la.status,
                    la.soil_type,
                    la.soil_ph,
                    la.annual_rainfall,
                    la.drought_risk,
                    la.created_at
                FROM loan_applications la
                JOIN farmers     f  ON f.id  = la.farmer_id
                LEFT JOIN land_details ld ON ld.farmer_id = la.farmer_id
                {where_sql}
                ORDER BY la.created_at DESC
                LIMIT ? OFFSET ?
            """
            rows = db.execute(data_sql, params + [per_page, offset]).fetchall()

        except Exception as exc:
            app.logger.error("DB read error:\n%s", traceback.format_exc())
            return _err("Database read failed", 500)

        applications = []
        for row in rows:
            rec = row_to_dict(row)
            rec["application_id"] = _fmt_app_id(rec["db_id"])
            applications.append(rec)

        return jsonify({
            "applications": applications,
            "total":        total,
            "page":         page,
            "per_page":     per_page,
            "pages":        max(1, -(-total // per_page)),  # ceiling division
        }), 200

    # =========================================================================
    # GET /applications/<id>
    # =========================================================================
    @app.route("/applications/<int:app_id>", methods=["GET"])
    def get_application(app_id: int):
        """
        Return the full detail of a single loan application.

        Path parameter:
            app_id (int) – the database integer ID of the application.

        Returns:
            200 with detailed application JSON.
            404 if not found.
        """
        try:
            db = _get_db()
            row = db.execute(
                """
                SELECT
                    la.id                    AS db_id,
                    f.full_name              AS farmer_name,
                    f.aadhaar,
                    f.phone,
                    f.district,
                    f.village,
                    ld.survey_number,
                    ld.land_size,
                    ld.irrigation,
                    ld.ownership,
                    ld.latitude,
                    ld.longitude,
                    la.primary_crop,
                    la.crop_diversity,
                    la.yield_1,
                    la.yield_2,
                    la.yield_3,
                    la.trust_score,
                    la.risk_level,
                    la.repayment_probability,
                    la.status,
                    la.soil_type,
                    la.soil_ph,
                    la.annual_rainfall,
                    la.drought_risk,
                    la.created_at
                FROM loan_applications la
                JOIN farmers     f  ON f.id  = la.farmer_id
                LEFT JOIN land_details ld ON ld.farmer_id = la.farmer_id
                WHERE la.id = ?
                """,
                (app_id,),
            ).fetchone()
        except Exception as exc:
            app.logger.error("DB read error:\n%s", traceback.format_exc())
            return _err("Database read failed", 500)

        if row is None:
            return _err(f"Application with id={app_id} not found", 404)

        result = row_to_dict(row)
        result["application_id"] = _fmt_app_id(result["db_id"])
        return jsonify(result), 200

    # =========================================================================
    # GET /stats   (bonus – for dashboard summary cards)
    # =========================================================================
    @app.route("/stats", methods=["GET"])
    def stats():
        """
        Return aggregate statistics for the lender dashboard summary cards.

        Returns:
            200 with:
                {
                  "total":    <int>,
                  "approved": <int>,
                  "review":   <int>,
                  "rejected": <int>,
                  "avg_trust_score": <float>
                }
        """
        try:
            db  = _get_db()
            row = db.execute(
                """
                SELECT
                    COUNT(*)                                   AS total,
                    SUM(CASE WHEN status='Approved' THEN 1 ELSE 0 END) AS approved,
                    SUM(CASE WHEN status='Review'   THEN 1 ELSE 0 END) AS review,
                    SUM(CASE WHEN status='Rejected' THEN 1 ELSE 0 END) AS rejected,
                    ROUND(AVG(trust_score), 1)                AS avg_trust_score
                FROM loan_applications
                """
            ).fetchone()
        except Exception as exc:
            app.logger.error("Stats query error:\n%s", traceback.format_exc())
            return _err("Database read failed", 500)

        return jsonify(dict(row)), 200


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

def _register_error_handlers(app: Flask) -> None:

    @app.errorhandler(404)
    def not_found(e):
        return _err("Route not found", 404)

    @app.errorhandler(405)
    def method_not_allowed(e):
        return _err("Method not allowed", 405)

    @app.errorhandler(500)
    def internal_error(e):
        return _err("Internal server error", 500)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (development server)
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    port  = int(os.getenv("AGRILOAN_PORT",  5000))
    debug = os.getenv("AGRILOAN_DEBUG", "0") == "1"

    print(f"\n{'='*60}")
    print("  AgriLoan Credit Scorer API")
    print(f"  Running on http://0.0.0.0:{port}")
    print(f"  Debug mode : {debug}")
    print(f"  DB path    : {DB_PATH}")
    print(f"  Model path : {_MODEL_PATH}")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=port, debug=debug)