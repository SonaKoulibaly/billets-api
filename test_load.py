import joblib

# Liste des fichiers modèles/scaler
files = [
    "log_model_25_08_2025.sav",
    "knn_model_25_08_2025.sav",
    "rf_model_25_08_2025.sav",
    "kmeans_model_25_08_2025.sav",
    "standard_scaler.sav"
]

print("=== Vérification du chargement des modèles ===")

for f in files:
    try:
        print(f"Tentative de chargement : {f}")
        m = joblib.load(f)
        print(f"  ✅ OK - Type chargé : {type(m)}")
    except Exception as e:
        print(f"  ❌ Erreur lors du chargement de {f} : {e}")

print("=== Test terminé ===")

