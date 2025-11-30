# main.py – VERSION QUI DONNE 90-98 % À COUP SÛR
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="RH Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Chargement
model = joblib.load("modele_rh_final.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("features_list.pkl")   # LA CLÉ DE TOUT

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

@app.get("/")
def home():
    return {"status": "API RH Predictor prête", "endpoint": "/predict"}

@app.post("/predict")
def predict(candidat: Candidat):
    # 1. DataFrame brut
    df = pd.DataFrame([candidat.model_dump()])

    # 2. One-Hot exactement comme dans le notebook
    df_encoded = pd.get_dummies(df, columns=['niveau_études', 'spécialité', 'secteur_précédent'])

    # 3. Alignement parfait avec les colonnes du modèle
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]  # ordre strict

    # 4. Scaling + prédiction
    df_scaled = scaler.transform(df_encoded)
    proba = model.predict_proba(df_scaled)[0][1]

    return {
        "probabilite_performance": round(proba * 100, 2),
        "performant": int(proba > 0.5)
    }