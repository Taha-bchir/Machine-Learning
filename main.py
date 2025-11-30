from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="RH Performance Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement
model = joblib.load("modele_rh_final.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features_list.pkl")

class Candidat(BaseModel):
    âge: int
    années_expérience: float
    score_test_technique: float
    score_softskills: float
    niveau_études: str
    spécialité: str
    secteur_précédent: str
    langues_parlées: int
    mobilité: int
    disponibilité_immédiate: int

@app.post("/predict")
def predict(candidat: Candidat):
    data = candidat.dict()
    df = pd.DataFrame([data])
    
    # One-hot comme dans le notebook
    df_encoded = pd.get_dummies(df, columns=['niveau_études','spécialité','secteur_précédent'])
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[features]
    
    X_scaled = scaler.transform(df_encoded)
    proba = model.predict_proba(X_scaled)[0][1]
    prediction = int(proba > 0.5)
    
    return {"probabilite_performance": round(proba*100, 2), "performant": prediction}