"""
services/feature_engineering.py
---------------------------------
Feature Engineering — transforms raw form data + soil/weather enrichment
into the exact numeric feature vector the trained ML model expects.

IMPORTANT: Feature order here MUST match the training column order in
train_model.py exactly:
    [land_size, crop_diversity, average_yield, soil_ph, rainfall,
     irrigation_score, ownership_score, drought_risk_score,
     nitrogen_score, yield_trend]

Usage:
    from services.feature_engineering import build_feature_vector
    X = build_feature_vector(form_data, soil_data, weather_data)
    prob = model.predict_proba(X)[0][1]
"""

import numpy as np


# ── Categorical → numeric encodings (must match training data) ────────────────

IRRIGATION_SCORES: dict[str, int] = {
    "drip":     4,   # Most efficient – lowest risk
    "canal":    3,
    "borewell": 2,
    "rain-fed": 1,   # Least reliable – highest risk
}

OWNERSHIP_SCORES: dict[str, int] = {
    "owner":  3,   # Full collateral value
    "lease":  2,
    "tenant": 1,   # No collateral – highest risk
}


def build_feature_vector(form_data: dict, soil_data: dict,
                         weather_data: dict) -> np.ndarray:
    """
    Build the ML model's input feature vector from application + enrichment.

    Args:
        form_data    (dict): Raw fields from the POST /evaluate JSON payload.
        soil_data    (dict): Output of soil_service.get_soil_data().
        weather_data (dict): Output of weather_service.get_weather_data().

    Returns:
        np.ndarray: Shape (1, 10) – ready for model.predict_proba().
    """
    # ── 1. Land size (acres) ─────────────────────────────────────────────
    land_size = float(form_data.get("landSize") or 0)

    # ── 2. Crop diversity (number of distinct crops) ──────────────────────
    crop_diversity = float(form_data.get("cropDiversity") or 1)
    crop_diversity = max(1.0, crop_diversity)   # floor at 1

    # ── 3. Average yield over last 3 years (quintals/acre) ────────────────
    raw_yields = []
    for key in ("yield1", "yield2", "yield3"):
        val = form_data.get(key)
        if val not in (None, "", "0", 0):
            try:
                raw_yields.append(float(val))
            except (TypeError, ValueError):
                pass
    # Use regional average of 15 quintals/acre if yields not provided
    average_yield = float(np.mean(raw_yields)) if raw_yields else 15.0

    # ── 4. Soil pH ────────────────────────────────────────────────────────
    soil_ph = float(soil_data.get("soil_ph") or 7.0)

    # ── 5. Annual rainfall (mm) ───────────────────────────────────────────
    rainfall = float(weather_data.get("annual_rainfall") or 700)

    # ── 6. Irrigation score (1–4) ─────────────────────────────────────────
    irrigation_type  = str(form_data.get("irrigation") or "rain-fed").lower()
    irrigation_score = float(IRRIGATION_SCORES.get(irrigation_type, 1))

    # ── 7. Ownership score (1–3) ──────────────────────────────────────────
    ownership_type  = str(form_data.get("ownership") or "tenant").lower()
    ownership_score = float(OWNERSHIP_SCORES.get(ownership_type, 1))

    # ── 8. Drought risk score (1=Low, 2=Medium, 3=High) ───────────────────
    drought_risk_score = float(weather_data.get("drought_risk_score") or 2)

    # ── 9. Nitrogen score (1–3) ───────────────────────────────────────────
    nitrogen_score = float(soil_data.get("nitrogen_score") or 2)

    # ── 10. Yield trend – slope of 3-year yield (positive = improving) ────
    if len(raw_yields) >= 2:
        trend = float(np.polyfit(range(len(raw_yields)), raw_yields, 1)[0])
    else:
        trend = 0.0

    feature_vector = np.array([[
        land_size,
        crop_diversity,
        average_yield,
        soil_ph,
        rainfall,
        irrigation_score,
        ownership_score,
        drought_risk_score,
        nitrogen_score,
        trend,
    ]])

    return feature_vector


def get_feature_summary(form_data: dict, soil_data: dict,
                        weather_data: dict) -> dict:
    """
    Return a labelled dict of all engineered features (for API responses
    and debugging). Does not affect the model prediction.

    Returns:
        dict mapping feature name → computed numeric value.
    """
    vec = build_feature_vector(form_data, soil_data, weather_data)
    feature_names = [
        "land_size", "crop_diversity", "average_yield", "soil_ph",
        "rainfall", "irrigation_score", "ownership_score",
        "drought_risk_score", "nitrogen_score", "yield_trend",
    ]
    return {name: round(float(val), 4)
            for name, val in zip(feature_names, vec[0])}