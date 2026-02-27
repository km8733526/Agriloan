"""
services/soil_service.py
------------------------
Soil Service — returns simulated soil analysis data based on GPS coordinates.

In production this would call an external API such as:
  - SoilGrids (https://soilgrids.org/) by ISRIC
  - ICAR National Bureau of Soil Survey API

For this implementation, deterministic dummy logic based on Maharashtra
latitude bands is used so the ML model always gets consistent feature values.

Usage:
    from services.soil_service import get_soil_data
    soil = get_soil_data(19.99, 73.78)
"""


def get_soil_data(latitude: float, longitude: float) -> dict:
    """
    Return soil properties for the given GPS coordinate.

    Latitude bands (Maharashtra, India):
        >= 20.5  → Vidarbha / northern MH  → Black Cotton soil
        >= 19.0  → Nashik / Pune belt      → Red Laterite soil
        >= 17.5  → Konkan / river valleys  → Alluvial soil
        < 17.5   → Southern MH             → Sandy Loam soil

    Args:
        latitude  (float): Decimal-degree latitude.
        longitude (float): Decimal-degree longitude.

    Returns:
        dict with keys:
            soil_type        (str)   – descriptive name
            soil_ph          (float) – pH scale 0–14
            nitrogen_level   (str)   – 'Low' | 'Medium' | 'High'
            nitrogen_score   (int)   – numeric 1/2/3 (used by ML)
            phosphorus_level (str)
            phosphorus_score (int)
            potassium_level  (str)
            potassium_score  (int)
            organic_carbon   (float) – % organic carbon in topsoil
    """
    lat = float(latitude)
    lng = float(longitude)

    # ── Soil classification by Maharashtra latitude band ──────────────────
    if lat >= 20.5:
        # Vidarbha / northern Maharashtra – classic black cotton belt
        soil_type     = "Black Cotton"
        base_ph       = 7.8
        nitrogen      = "Medium"
        phosphorus    = "Medium"
        potassium     = "High"
        organic_c     = 0.62
    elif lat >= 19.0:
        # Nashik / Pune / Ahmednagar belt – mixed red-laterite soils
        soil_type     = "Red Laterite"
        base_ph       = 6.5
        nitrogen      = "Low"
        phosphorus    = "Low"
        potassium     = "Medium"
        organic_c     = 0.38
    elif lat >= 17.5:
        # Konkan / river-valley alluvial plains – highly fertile
        soil_type     = "Alluvial"
        base_ph       = 7.2
        nitrogen      = "High"
        phosphorus    = "High"
        potassium     = "High"
        organic_c     = 0.91
    else:
        # Southern Maharashtra – sandy loam, lower fertility
        soil_type     = "Sandy Loam"
        base_ph       = 6.8
        nitrogen      = "Low"
        phosphorus    = "Medium"
        potassium     = "Low"
        organic_c     = 0.28

    # Fine-tune pH using the fractional part of longitude (±0.1 variance)
    ph_tweak = (lng % 1.0) * 0.2 - 0.1
    soil_ph  = round(base_ph + ph_tweak, 2)
    soil_ph  = max(4.5, min(9.0, soil_ph))   # clamp to realistic range

    # Nutrient label → numeric score for ML feature engineering
    _score = {"Low": 1, "Medium": 2, "High": 3}

    return {
        "soil_type":        soil_type,
        "soil_ph":          soil_ph,
        "nitrogen_level":   nitrogen,
        "nitrogen_score":   _score[nitrogen],
        "phosphorus_level": phosphorus,
        "phosphorus_score": _score[phosphorus],
        "potassium_level":  potassium,
        "potassium_score":  _score[potassium],
        "organic_carbon":   organic_c,
    }