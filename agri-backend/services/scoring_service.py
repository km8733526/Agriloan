"""
services/scoring_service.py
----------------------------
Scoring Service — converts raw ML repayment probability (0.0–1.0) into a
trust score (0–100), risk tier, status decision, and loan term suggestion.

Risk rules (as per product specification):
    trust_score >= 80  →  Low Risk    →  Approved
    trust_score 60–79  →  Medium Risk →  Review
    trust_score < 60   →  High Risk   →  Rejected

Loan terms scale with risk tier:
    Approved  → ₹5,00,000 @ 7.5% p.a. for 5 years
    Review    → ₹2,50,000 @ 9.5% p.a. for 3 years
    Rejected  → No loan terms offered

Usage:
    from services.scoring_service import calculate_trust_score
    result = calculate_trust_score(0.82)
"""

import math


# ── Loan term look-up by status ───────────────────────────────────────────────
_LOAN_TERMS: dict[str, dict] = {
    "Approved": {
        "max_loan_amount_inr":  500000,
        "interest_rate_pct":     7.5,
        "tenure_years":          5,
    },
    "Review": {
        "max_loan_amount_inr":  250000,
        "interest_rate_pct":     9.5,
        "tenure_years":          3,
    },
    "Rejected": {
        "max_loan_amount_inr":  0,
        "interest_rate_pct":     None,
        "tenure_years":          None,
    },
}


def calculate_trust_score(repayment_probability: float) -> dict:
    """
    Convert a model repayment probability into a trust score and credit decision.

    Formula:
        trust_score = round(repayment_probability × 100)

    Args:
        repayment_probability (float): Model output in range [0.0, 1.0].

    Returns:
        dict with keys:
            trust_score           (int)         – 0 to 100
            risk_level            (str)         – 'Low' | 'Medium' | 'High'
            status                (str)         – 'Approved' | 'Review' | 'Rejected'
            repayment_probability (float)       – 4 decimal places
            loan_terms            (dict)        – amount, rate, tenure
            emi_monthly           (float|None)  – monthly instalment in INR
    """
    # Clamp to valid probability range
    prob        = max(0.0, min(1.0, float(repayment_probability)))
    trust_score = int(round(prob * 100))

    # ── Risk tier classification ───────────────────────────────────────────
    if trust_score >= 80:
        risk_level = "Low"
        status     = "Approved"
    elif trust_score >= 60:
        risk_level = "Medium"
        status     = "Review"
    else:
        risk_level = "High"
        status     = "Rejected"

    loan_terms  = _LOAN_TERMS[status].copy()
    emi_monthly = _calculate_emi(
        principal    = loan_terms["max_loan_amount_inr"],
        annual_rate  = loan_terms["interest_rate_pct"],
        tenure_years = loan_terms["tenure_years"],
    )

    return {
        "trust_score":            trust_score,
        "risk_level":             risk_level,
        "status":                 status,
        "repayment_probability":  round(prob, 4),
        "loan_terms":             loan_terms,
        "emi_monthly":            emi_monthly,
    }


def _calculate_emi(principal: float,
                   annual_rate: float | None,
                   tenure_years: int | None) -> float | None:
    """
    Calculate monthly EMI using the standard reducing-balance formula:
        EMI = P × r × (1+r)^n / ((1+r)^n - 1)

    where:
        P = principal amount
        r = monthly interest rate (annual_rate / 12 / 100)
        n = total number of months (tenure_years × 12)

    Returns None for Rejected applications (no loan offered).
    """
    if not principal or not annual_rate or not tenure_years:
        return None

    r   = annual_rate / (12.0 * 100.0)   # monthly interest rate
    n   = tenure_years * 12               # total months
    if r == 0:
        return round(principal / n, 2)

    emi = principal * r * math.pow(1 + r, n) / (math.pow(1 + r, n) - 1)
    return round(emi, 2)