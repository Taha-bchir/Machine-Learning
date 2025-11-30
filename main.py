# main.py – VERSION QUI MARCHE À 100%
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Chargement
model = joblib.load("modele_rh_final.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("features_list.pkl")   # très important !

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
    return {"message": "RH Predictor API – utilisez POST /predict"}

@app.post("/predict")
def predict(candidat: Candidat):
    # 1. On crée un DataFrame avec les données brutes
    input_data = pd.DataFrame([candidat.dict()])

    # 2. One-Hot exactement comme dans le notebook
    input_encoded = pd.get_dummies(
        input_data,
        columns=['niveau_études', 'spécialité', 'secteur_précédent']
    )

    # 3. On rajoute TOUTES les colonnes attendues par le modèle (les manquantes = 0)
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[expected_columns]  # ordre exact !

    # 4. Scaling + prédiction
    input_scaled = scaler.transform(input_encoded)
    proba = model.predict_proba(input_scaled)[0][1]

    return {
        "probabilite_performance": round(float(proba * 100), 2),
        "performant": int(proba > 0.5)
    }