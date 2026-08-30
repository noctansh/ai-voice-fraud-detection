def calculate_risk_index(synthetic_score: float, mismatch_score: float, urgency_score: float) -> float:
    # 50% Acoustic Deepfake + 30% Speaker Mismatch + 20% Urgency Threat
    composite = (0.50 * synthetic_score) + (0.30 * mismatch_score) + (0.20 * urgency_score)
    return round(composite * 100, 2)