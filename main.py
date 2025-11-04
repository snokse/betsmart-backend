from fastapi import FastAPI
import joblib
import gdown
import numpy as np
import json
import os
from pydantic import BaseModel
import uvicorn

# === Configuration ===
MODEL_ID = os.environ.get("BETSMART_DRIVE_MODEL_ID", "1FBM4lYhm9pvEmlL4vmJV0YMKl-rIIXaJ")
MODEL_PATH = "/tmp/bet_predict_model.pkl"  # /tmp pour Render
MAPPING_PATH = "team_league_mapping.json"

app = FastAPI(title="Betsmart Prediction API", version="1.0")

# === Cache des ressources ===
model_cache = None
mapping_cache = None


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


# === Chargement du mapping ===
def load_mapping():
    global mapping_cache
    if mapping_cache is None:
        with open(MAPPING_PATH, "r") as f:
            mapping_cache = json.load(f)
        print("✅ Mapping chargé avec succès.")


# === Endpoint de prédiction avec chargement à la demande ===
@app.post("/predict")
async def predict_match(data: MatchInput):
    global model_cache
    try:
        # Charger le modèle seulement si besoin
        if model_cache is None:
            download_model()
            model_cache = joblib.load(MODEL_PATH)
            print("✅ Modèle chargé en mémoire.")

        load_mapping()

        league_id = mapping_cache["LEAGUE"].get(data.league)
        team1_id = mapping_cache["TEAM1"].get(data.team1)
        team2_id = mapping_cache["TEAM2"].get(data.team2)

        if None in (league_id, team1_id, team2_id):
            return {"status": "error", "message": "Nom d'équipe ou ligue introuvable"}

        X_new = np.array([[league_id, team1_id, team2_id, data.odd1, data.oddx, data.odd2]])
        proba = model_cache.predict_proba(X_new)[0]
        classes = model_cache.classes_

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
