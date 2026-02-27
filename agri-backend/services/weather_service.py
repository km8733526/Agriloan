"""
services/weather_service.py
---------------------------
Weather Service — returns simulated climate/weather data based on GPS
coordinates.

In production this would call:
  - OpenWeatherMap API (https://openweathermap.org/api)
  - India Meteorological Department (IMD) API
  - NASA POWER API (https://power.larc.nasa.gov/)

For this implementation, deterministic dummy logic based on Maharashtra
coordinate ranges is used.

Usage:
    from services.weather_service import get_weather_data
    weather = get_weather_data(19.99, 73.78)
"""


def get_weather_data(latitude: float, longitude: float) -> dict:
    """
    Return climate/weather data for the given GPS coordinate.

    Climate zones (Maharashtra, India):
        lat >= 20.0 and lng <= 74.5  → Western Ghats / Nashik region
        lat >= 18.5 and lng >= 73.5  → Pune / Deccan plateau
        lat >= 17.5                  → Solapur / southern arid zone
        else                         → Konkan coastal high-rainfall zone

    Args:
        latitude  (float): Decimal-degree latitude.
        longitude (float): Decimal-degree longitude.

    Returns:
        dict with keys:
            annual_rainfall      (int)   – mm/year
            temperature_avg      (float) – °C annual average
            temperature_min      (float) – °C typical winter low
            temperature_max      (float) – °C typical summer high
            humidity             (int)   – % annual average
            drought_risk         (str)   – 'Low' | 'Medium' | 'High'
            drought_risk_score   (int)   – 1/2/3 (used by ML)
            monsoon_reliability  (str)   – 'Good' | 'Moderate' | 'Poor'
            frost_risk           (str)   – 'None' | 'Low' | 'Medium'
    """
    lat = float(latitude)
    lng = float(longitude)

    # ── Climate zone classification for Maharashtra ───────────────────────
    if lat >= 20.0 and lng <= 74.5:
        # Nashik / Western Ghats foothills – moderate rainfall, low drought
        annual_rainfall     = 850
        temperature_avg     = 25.5
        temperature_min     = 12.0
        temperature_max     = 38.0
        humidity            = 65
        drought_risk        = "Low"
        monsoon_reliability = "Good"
        frost_risk          = "Low"

    elif lat >= 18.5 and lng >= 73.5:
        # Pune / Deccan plateau – semi-arid, moderate drought risk
        annual_rainfall     = 710
        temperature_avg     = 27.0
        temperature_min     = 10.0
        temperature_max     = 40.0
        humidity            = 55
        drought_risk        = "Medium"
        monsoon_reliability = "Moderate"
        frost_risk          = "None"

    elif lat >= 17.5:
        # Solapur / Satara southern Deccan – arid, high drought risk
        annual_rainfall     = 520
        temperature_avg     = 28.5
        temperature_min     = 14.0
        temperature_max     = 43.0
        humidity            = 45
        drought_risk        = "High"
        monsoon_reliability = "Poor"
        frost_risk          = "None"

    else:
        # Konkan coastal / default – high rainfall, low drought risk
        annual_rainfall     = 2500
        temperature_avg     = 27.0
        temperature_min     = 18.0
        temperature_max     = 35.0
        humidity            = 80
        drought_risk        = "Low"
        monsoon_reliability = "Good"
        frost_risk          = "None"

    # Add small variation based on longitude fractional part (±25 mm)
    rainfall_tweak  = int((lng % 1.0) * 50 - 25)
    annual_rainfall = max(200, annual_rainfall + rainfall_tweak)

    _drought_score = {"Low": 1, "Medium": 2, "High": 3}

    return {
        "annual_rainfall":     annual_rainfall,
        "temperature_avg":     temperature_avg,
        "temperature_min":     temperature_min,
        "temperature_max":     temperature_max,
        "humidity":            humidity,
        "drought_risk":        drought_risk,
        "drought_risk_score":  _drought_score[drought_risk],
        "monsoon_reliability": monsoon_reliability,
        "frost_risk":          frost_risk,
    }