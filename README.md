# 🧠 Betsmart Backend (FastAPI)

Ce backend sert à héberger le modèle de prédiction des matchs pour l'application **Betsmart**.  
Il télécharge automatiquement le modèle ML depuis Google Drive et le garde en mémoire pour traiter les requêtes de prédiction.

---

## 🚀 Fonctionnalités
- Téléchargement automatique du modèle depuis Google Drive
- API REST pour la prédiction
- Intégration simple avec Supabase et Flet
- Retour JSON avec les probabilités des issues (1, X, 2)

---

## 🛠️ Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/ton-compte/betsmart-backend.git
cd betsmart-backend

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le serveur
python main.py


# 4.Variable d’environnement
BETSMART_DRIVE_MODEL_ID=1FBM4lYhm9pvEmlL4vmJV0YMKl-rIIXaJ

# 5.serveur via
uvicorn main:app --host 0.0.0.0 --port 10000