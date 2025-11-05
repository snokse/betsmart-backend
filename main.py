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
MODEL_PATH = "/tmp/bet_predict_model.pkl"  # Utilise /tmp sur Render
MAPPING_PATH = "team_league_mapping.json"

app = FastAPI(title="Betsmart Prediction API", version="1.0")

# === Cache pour le mapping uniquement (léger)
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
    """Télécharge le modèle depuis Google Drive uniquement s'il n'existe pas déjà."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Téléchargement du modèle depuis Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Modèle téléchargé avec succès.")


# === Chargement du mapping JSON (léger, gardé en mémoire) ===
def load_mapping():
    """Charge le mapping une seule fois (léger)."""
    global mapping_cache
    if mapping_cache is None:
        with open(MAPPING_PATH, "r") as f:
            mapping_cache = json.load(f)
        print("✅ Mapping chargé avec succès.")


# === Endpoint /predict ===
@app.post("/predict")
async def predict_match(data: MatchInput):
    """
    Endpoint principal — calcule la prédiction pour un match.
    Le modèle est chargé temporairement pour éviter les crashs mémoire.
    """
    try:
        # Charger le mapping (léger)
        load_mapping()

        # Vérifier les identifiants d'équipes et de ligue
        league_id = mapping_cache["LEAGUE"].get(data.league)
        team1_id = mapping_cache["TEAM1"].get(data.team1)
        team2_id = mapping_cache["TEAM2"].get(data.team2)

        if None in (league_id, team1_id, team2_id):
            return {"status": "error", "message": "Nom d'équipe ou ligue introuvable"}

        # Télécharger et charger le modèle dans /tmp
        download_model()
        model = joblib.load(MODEL_PATH)
        print("✅ Modèle chargé temporairement pour prédiction.")

        # Effectuer la prédiction
        X_new = np.array([[league_id, team1_id, team2_id, data.odd1, data.oddx, data.odd2]])
        proba = model.predict_proba(X_new)[0]
        classes = model.classes_

        # Supprimer le modèle immédiatement pour libérer la mémoire
        del model
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            print("🧹 Modèle supprimé de /tmp pour libérer la mémoire.")

        # Construire la réponse
        results = dict(zip(classes, [float(p) for p in proba]))
        best = max(results, key=results.get)

        return {
            "status": "ok",
            "prediction": best,
            "probabilities": results
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# === Endpoint racine (test de vie) ===
@app.get("/")
def root():
    """Simple test pour vérifier que l'API fonctionne."""
    return {"message": "✅ Betsmart Prediction API is running!"}


# === Lancer localement (développement) ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
