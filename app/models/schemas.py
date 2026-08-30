from pydantic import BaseModel

class AnalysisResult(BaseModel):
    synthetic_score: float = 0.0     # Vansh
    speaker_mismatch: float = 0.0    # Keerthana
    urgency_score: float = 0.0       # Aakriti
    transcript: str = ""
    risk_score: float = 0.0          # 0-100%
    alert: bool = False              # > 75%