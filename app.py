# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import os, joblib, pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "export_artifacts", "aurasense_stress_model.joblib")

app = FastAPI(title="AuraSense Stress Inference API", version="v1")

# Lazy-load model once
pipe = None
load_error = None
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    load_error = str(e)

class RawVitals(BaseModel):
    HR: float = Field(..., description="Heart Rate")
    HRV: float = Field(..., description="Heart Rate Variability")
    BT: float = Field(..., description="Body Temperature (°C)")
    M_HR: Optional[float] = None
    D_HR: Optional[float] = None
    M_HRV: Optional[float] = None
    D_HRV: Optional[float] = None
    M_BT: Optional[float] = None
    D_BT: Optional[float] = None

class DirectFeatures(BaseModel):
    features: Dict[str, float]

class PredictRequest(BaseModel):
    mode: str = Field(..., description="'raw' or 'direct'")
    raw: Optional[RawVitals] = None
    direct: Optional[DirectFeatures] = None

FEATURES_EXPECTED = ["HR","HRV","BT","C_HR","C_HRV","C_BT","G_stress"]

def compute_engineered(raw: RawVitals) -> Dict[str, float]:
    M_HR  = raw.M_HR  if raw.M_HR  is not None else raw.HR
    D_HR  = raw.D_HR  if (raw.D_HR  not in (None,0)) else 1.0
    M_HRV = raw.M_HRV if raw.M_HRV is not None else raw.HRV
    D_HRV = raw.D_HRV if (raw.D_HRV not in (None,0)) else 1.0
    M_BT  = raw.M_BT  if raw.M_BT  is not None else raw.BT
    D_BT  = raw.D_BT  if (raw.D_BT  not in (None,0)) else 1.0

    C_HR  = (raw.HR  - M_HR)  / D_HR
    C_HRV = - (raw.HRV - M_HRV) / D_HRV
    C_BT  = abs((raw.BT  - M_BT)  / D_BT)
    G     = 0.5*C_HRV + 0.3*C_HR + 0.2*C_BT

    return {
        "HR": raw.HR, "HRV": raw.HRV, "BT": raw.BT,
        "C_HR": C_HR, "C_HRV": C_HRV, "C_BT": C_BT, "G_stress": G
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": pipe is not None, "load_error": load_error}

@app.post("/predict")
def predict(req: PredictRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    if req.mode not in ("raw","direct"):
        raise HTTPException(status_code=400, detail="mode must be 'raw' or 'direct'")

    if req.mode == "raw":
        if req.raw is None:
            raise HTTPException(status_code=400, detail="Provide 'raw' payload.")
        row = compute_engineered(req.raw)
    else:
        if req.direct is None:
            raise HTTPException(status_code=400, detail="Provide 'direct' payload.")
        row = req.direct.features

    df = pd.DataFrame([row])
    for f in FEATURES_EXPECTED:
        if f not in df.columns:
            df[f] = None

    pred = pipe.predict(df)[0]
    proba = None
    try:
        proba_arr = pipe.predict_proba(df)[0]
        classes = getattr(pipe.named_steps.get("clf", object()), "classes_", None)
        if classes is not None:
            proba = {str(k): float(v) for k, v in zip(classes, proba_arr)}
    except Exception:
        pass

    return {"stress_level": str(pred), "probabilities": proba, "features_used": FEATURES_EXPECTED}
