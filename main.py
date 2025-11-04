from fastapi import FastAPI
import joblib
import gdown
import numpy as np
import json
import os
from pydantic import BaseModel
import uvicorn

# === Configuration ===
MODEL_ID = "1FBM4lYhm9pvEmlL4vmJV0YMKl-rIIXaJ"
MODEL_PATH = "bet_predict_model.pkl"
MAPPING_PATH = "team_league_mapping.json"

app = FastAPI(title="Betsmart Prediction API", version="1.0")

model = None
mapping = None


# === Modèle d’entrée API ===
class MatchInput(BaseModel):
    team1: str
    team2: str
    league: str
    odd1: float
    oddx: float
    odd2: float


# === Téléchargement du modèle depuis Google Drive ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Téléchargement du modèle depuis Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Modèle téléchargé avec succès.")


# === Chargement du modèle et du mapping ===
def load_resources():
    global model, mapping
    try:
        download_model()
        model = joblib.load(MODEL_PATH)
        with open(MAPPING_PATH, "r") as f:
            mapping = json.load(f)
        print("✅ Modèle et mapping chargés avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")


# === Endpoint de prédiction ===
@app.post("/predict")
async def predict_match(data: MatchInput):
    if model is None or mapping is None:
        return {"status": "error", "message": "Model or mapping not loaded"}

    try:
        league_id = mapping["LEAGUE"].get(data.league)
        team1_id = mapping["TEAM1"].get(data.team1)
        team2_id = mapping["TEAM2"].get(data.team2)

        if None in (league_id, team1_id, team2_id):
            return {"status": "error", "message": "Nom d'équipe ou ligue introuvable"}

        X_new = np.array([[league_id, team1_id, team2_id, data.odd1, data.oddx, data.odd2]])
        proba = model.predict_proba(X_new)[0]
        classes = model.classes_

        results = dict(zip(classes, [float(p) for p in proba]))
        best = max(results, key=results.get)

        return {
            "status": "ok",
            "prediction": best,
            "probabilities": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
def root():
    return {"message": "✅ Betsmart Prediction API is running!"}


# === Lancer localement ===
if __name__ == "__main__":
    load_resources()
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    load_resources()
