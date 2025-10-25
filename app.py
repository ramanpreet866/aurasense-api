
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Optional, Dict, List

MODEL_PATH = "export_artifacts/aurasense_stress_model.joblib"
FEATURES_EXPECTED = ["HR", "HRV", "BT", "C_HR", "C_HRV", "C_BT", "G_stress"]

app = FastAPI(title="AuraSense Stress Inference API", version="v2025.10.25")
pipe = joblib.load(MODEL_PATH)

class RawVitals(BaseModel):
    HR: float = Field(..., description="Heart Rate")
    HRV: float = Field(..., description="Heart Rate Variability")
    BT: float = Field(..., description="Body Temperature (°C)")
    # Optional per-user stats to compute C_*; if omitted, we approximate with single-sample std=1
    M_HR: Optional[float] = None
    D_HR: Optional[float] = None
    M_HRV: Optional[float] = None
    D_HRV: Optional[float] = None
    M_BT: Optional[float] = None
    D_BT: Optional[float] = None

class DirectFeatures(BaseModel):
    # Supply features exactly as used in training (any subset will be handled with imputation)
    features: Dict[str, float]

class PredictRequest(BaseModel):
    mode: str = Field(..., description="'raw' for HR/HRV/BT, 'direct' for feature dict")
    raw: Optional[RawVitals] = None
    direct: Optional[DirectFeatures] = None

def _compute_engineered(raw: RawVitals) -> Dict[str, float]:
    # Use provided per-user stats if present; otherwise, fall back to trivial z-scores with std=1
    M_HR  = raw.M_HR  if raw.M_HR  is not None else raw.HR
    D_HR  = raw.D_HR  if (raw.D_HR  is not None and raw.D_HR  != 0) else 1.0
    M_HRV = raw.M_HRV if raw.M_HRV is not None else raw.HRV
    D_HRV = raw.D_HRV if (raw.D_HRV is not None and raw.D_HRV != 0) else 1.0
    M_BT  = raw.M_BT  if raw.M_BT  is not None else raw.BT
    D_BT  = raw.D_BT  if (raw.D_BT  is not None and raw.D_BT  != 0) else 1.0

    C_HR  = (raw.HR  - M_HR)  / D_HR
    C_HRV = - (raw.HRV - M_HRV) / D_HRV
    C_BT  = abs((raw.BT  - M_BT)  / D_BT)
    G     = 0.5*C_HRV + 0.3*C_HR + 0.2*C_BT

    return {
        "HR": raw.HR, "HRV": raw.HRV, "BT": raw.BT,
        "C_HR": C_HR, "C_HRV": C_HRV, "C_BT": C_BT, "G_stress": G
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if req.mode not in ("raw", "direct"):
        raise HTTPException(status_code=400, detail="mode must be 'raw' or 'direct'")

    if req.mode == "raw":
        if req.raw is None:
            raise HTTPException(status_code=400, detail="Provide 'raw' payload.")
        row = _compute_engineered(req.raw)
    else:
        if req.direct is None:
            raise HTTPException(status_code=400, detail="Provide 'direct' payload.")
        row = req.direct.features

    # Build a single-row dataframe that includes any supplied fields
    df = pd.DataFrame([row])

    # Ensure all expected features exist (imputer will handle missing within pipeline)
    for f in FEATURES_EXPECTED:
        if f not in df.columns:
            df[f] = None

    # Predict
    pred = pipe.predict(df)[0]
    proba = None
    try:
        proba_arr = pipe.predict_proba(df)[0]
        classes = getattr(pipe.named_steps.get("clf", object()), "classes_", None)
        if classes is not None:
            proba = dict(zip(map(str, classes), map(float, proba_arr)))
    except Exception:
        pass

    return {
        "stress_level": str(pred),
        "probabilities": proba,
        "features_used": FEATURES_EXPECTED
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_path": MODEL_PATH}
