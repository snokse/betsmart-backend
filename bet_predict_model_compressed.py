import joblib
import sys

# Charger ton modèle existant
try:
	model = joblib.load("bet_predict_model.pkl")
except ModuleNotFoundError as e:
	# Happens when unpickling requires a module not installed (e.g. sklearn)
	missing = getattr(e, "name", None)
	if missing and "sklearn" in missing:
		print("The model requires 'scikit-learn' to be installed to unpickle.")
		print()
		print("Install it with pip:")
		print("    python -m pip install scikit-learn")
		print()
		print("Or, with your Python executable (example):")
		print("    & C:/Python314/python.exe -m pip install scikit-learn")
	else:
		print("A required module is missing while loading the model:", e)
	sys.exit(1)

# Sauvegarder en version compressée (niveau 3 = bon équilibre)
joblib.dump(model, "bet_predict_model_compressed.pkl", compress=3)
