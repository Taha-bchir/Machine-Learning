# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

# ================== APP ==================
app = FastAPI(title="RH Performance Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod tu peux restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== CHARGEMENT DES OBJETS ==================
model = joblib.load("modele_rh_final.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("features_list.pkl")

# ================== SCHEMA D'ENTRÉE ==================
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

# ================== ROUTES ==================
@app.get("/")
def home():
    return {"message": "RH Predictor API – POST /predict pour tester"}

@app.post("/predict")
def predict(candidat: Candidat):
    # 1. DataFrame brut
    df = pd.DataFrame([candidat.model_dump()])

    # 2. One-Hot
    df_encoded = pd.get_dummies(df, columns=['niveau_études', 'spécialité', 'secteur_précédent'])

    # 3. Alignement parfait des colonnes
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_columns]

    # 4. Scaling + prédiction
    df_scaled = scaler.transform(df_encoded)
    proba = model.predict_proba(df_scaled)[0][1]

    return {
        "probabilite_performance": round(proba * 100, 2),
        "performant": int(proba > 0.5)
    }