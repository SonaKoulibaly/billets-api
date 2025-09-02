# app.py — API FastAPI pour billets Vrai/Faux

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from typing import Dict, Any
import joblib, pandas as pd, numpy as np
import io, csv, os, zipfile, shutil  # <-- ajout de shutil

# -----------------------------------------------------------------------------
# Config de l’app
# -----------------------------------------------------------------------------
app = FastAPI(title="Billets API", version="1.0.0")

# (Optionnel) autoriser les appels depuis Streamlit/local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restreins si besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Colonnes attendues (ordre utilisé à l’entraînement)
EXPECTED_COLS = ["length", "height_left", "height_right", "margin_low", "margin_up", "diagonal"]

# --- Chemins robustes (relatifs au fichier app.py) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")  # <-- sous-dossier 'models'
MODELS_ZIP = os.path.join(BASE_DIR, "models.zip")

# Décompression auto si models/ n’existe pas mais models.zip est présent
if not os.path.isdir(MODELS_DIR) and os.path.exists(MODELS_ZIP):
    with zipfile.ZipFile(MODELS_ZIP, "r") as z:
        z.extractall(MODELS_DIR)

MODEL_PATHS = {
    "logreg":  os.path.join(MODELS_DIR, "log_model_25_08_2025.sav"),
    "knn":     os.path.join(MODELS_DIR, "knn_model_25_08_2025.sav"),
    "rf":      os.path.join(MODELS_DIR, "rf_model_25_08_2025.sav"),
    "kmeans":  os.path.join(MODELS_DIR, "kmeans_model_25_08_2025.sav"),
    "scaler":  os.path.join(MODELS_DIR, "standard_scaler.sav"),
}

# --- Auto-unzip + flatten du dossier models ---
import zipfile, shutil

def _ensure_models_present():
    # 1) Si models/ est vide ou ne contient pas les .sav, tenter de dézipper models.zip
    need_unzip = False
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        need_unzip = True
    else:
        # cherche des .sav directement à la racine de models/
        has_sav = any(f.endswith(".sav") for f in os.listdir(MODELS_DIR))
        need_unzip = not has_sav

    zip_path = os.path.join(BASE_DIR, "models.zip")
    if need_unzip and os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(MODELS_DIR)
        except Exception as e:
            raise RuntimeError(f"Echec unzip de {zip_path}: {e}")

    # 2) Si après unzip on a un sous-dossier unique (ex: models/models/...), on remonte les .sav
    entries = [e for e in os.listdir(MODELS_DIR) if not e.startswith(".")]
    if len(entries) == 1:
        only = os.path.join(MODELS_DIR, entries[0])
        if os.path.isdir(only):
            # déplacer tous les .sav du sous-dossier vers MODELS_DIR
            for root, _, files in os.walk(only):
                for f in files:
                    if f.endswith(".sav"):
                        src = os.path.join(root, f)
                        dst = os.path.join(MODELS_DIR, f)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
            # on peut laisser le sous-dossier, ce n'est pas bloquant

    # 3) Vérification finale
    expected = [
        "log_model_25_08_2025.sav",
        "knn_model_25_08_2025.sav",
        "rf_model_25_08_2025.sav",
        "kmeans_model_25_08_2025.sav",
        "standard_scaler.sav",
    ]
    missing = [f for f in expected if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise RuntimeError(
            "Fichiers modèles manquants après préparation: "
            + ", ".join(missing)
            + f"\nMODELS_DIR={MODELS_DIR}\nContenu: {os.listdir(MODELS_DIR)}"
        )

# appelle la préparation AVANT de charger les modèles
_ensure_models_present()

LABELS = {"0": "Faux", "1": "Vrai"}  # mapping affichage

# -----------------------------------------------------------------------------
# Chargement des artefacts au démarrage
# -----------------------------------------------------------------------------
def _safe_load(path: str):
    if not os.path.exists(path):
        # aide au debug : montre où on cherche
        raise RuntimeError(
            f"Fichier introuvable: {path}\n"
            f"MODELS_DIR={MODELS_DIR}\n"
            f"Contenu MODELS_DIR: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'dossier inexistant'}"
        )
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Echec de chargement '{path}': {e}")

log_model    = _safe_load(MODEL_PATHS["logreg"])
knn_model    = _safe_load(MODEL_PATHS["knn"])
rf_model     = _safe_load(MODEL_PATHS["rf"])
kmeans_model = _safe_load(MODEL_PATHS["kmeans"])
scaler       = _safe_load(MODEL_PATHS["scaler"])

# -----------------------------------------------------------------------------
# Schéma d’entrée pour /predict_one
# -----------------------------------------------------------------------------
class Billet(BaseModel):
    length: float = Field(..., description="Longueur du billet (mm)")
    height_left: float = Field(..., description="Hauteur à gauche (mm)")
    height_right: float = Field(..., description="Hauteur à droite (mm)")
    margin_low: float = Field(..., description="Marge bas (mm)")
    margin_up: float = Field(..., description="Marge haut (mm)")
    diagonal: float = Field(..., description="Diagonale (mm)")

# -----------------------------------------------------------------------------
# Helpers (lecture CSV, nettoyage, scaling, prédiction)
# -----------------------------------------------------------------------------
def _detect_sep(sample_bytes: bytes) -> str:
    """Détecte le séparateur (',' ou ';') sur un échantillon."""
    try:
        sniff = csv.Sniffer().sniff(sample_bytes.decode("utf-8", errors="ignore"))
        return sniff.delimiter
    except Exception:
        return ","  # fallback

def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    """Lit le CSV uploadé avec détection de séparateur."""
    raw = file.file.read()
    if len(raw) == 0:
        raise ValueError("Fichier vide.")
    sep = _detect_sep(raw[:4096])
    buf = io.StringIO(raw.decode("utf-8", errors="ignore"))
    df = pd.read_csv(buf, sep=sep)
    return df

def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Garantit présence/ordre des colonnes, coercition numérique, NaN/Inf -> moyennes scaler."""
    # Colonnes manquantes -> NaN, puis réordonner
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[EXPECTED_COLS].copy()

    # Forcer numérique et nettoyer Inf
    for c in EXPECTED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remplir NaN avec la moyenne du scaler si disponible (cohérent avec entraînement)
    if hasattr(scaler, "mean_"):
        means = pd.Series(scaler.mean_, index=EXPECTED_COLS)
        df = df.fillna(means)
    else:
        df = df.fillna(df.mean(numeric_only=True))

    return df

def _scale(df_num: pd.DataFrame) -> np.ndarray:
    return scaler.transform(df_num.values)

def _predict_all(df_num: pd.DataFrame, X_scaled: np.ndarray) -> Dict[str, Any]:
    """Effectue toutes les prédictions + probabilités moyennes + mapping KMeans."""
    # Prédictions brutes (0/1)
    pred_log = log_model.predict(X_scaled)
    pred_knn = knn_model.predict(X_scaled)
    pred_rf  = rf_model.predict(df_num.values)          # RF entraîné sur X brut
    pred_km  = kmeans_model.predict(X_scaled)           # labels 0/1 arbitraires

    # Mapper KMeans pour « coller » aux classes supervisées (alignement sur RF)
    try:
        mask0 = (pred_km == 0)
        mask1 = ~mask0
        maj0 = np.round(np.mean(pred_rf[mask0])) if mask0.any() else 0
        maj1 = np.round(np.mean(pred_rf[mask1])) if mask1.any() else 1
        pred_km_mapped = (1 - pred_km) if maj0 > maj1 else pred_km
    except Exception:
        pred_km_mapped = pred_km

    # Probabilités moyennes (si disponibles)
    def avg_pos_prob(estimator, X):
        try:
            return float(np.mean(estimator.predict_proba(X)[:, 1]))
        except Exception:
            return None

    return {
        "classes": {
            "logreg": pred_log.tolist(),
            "knn":    pred_knn.tolist(),
            "rf":     pred_rf.tolist(),
            "kmeans": pred_km_mapped.tolist(),
        },
        "avg_positive_probability": {
            "logreg": avg_pos_prob(log_model, X_scaled),
            "knn":    avg_pos_prob(knn_model, X_scaled),
            "rf":     avg_pos_prob(rf_model, df_num.values),
        }
    }

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    """
    Petit point d'entrée lisible :
    - Renvoie un résumé de l'API et les endpoints utiles
    - Fournit les colonnes attendues
    - Lien vers /docs (Swagger)
    """
    return {
        "message": "API Billets prête ✅",
        "version": app.version,
        "endpoints": {
            "health": "/health",
            "version": "/version",
            "predict_one": "/predict_one",
            "predict_csv": "/predict_csv",
            "docs": "/docs"
        },
        "expected_cols": ["length", "height_left", "height_right", "margin_low", "margin_up", "diagonal"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODEL_PATHS.keys()), "expected_cols": EXPECTED_COLS}

@app.get("/version")
def version():
    return {"api": app.version}

@app.post("/predict_one")
def predict_one(billet: Billet):
    """Prédit pour un billet (JSON) et renvoie labels Vrai/Faux + majority vote."""
    df = pd.DataFrame([billet.dict()])[EXPECTED_COLS]
    df_num = _coerce_and_align(df)
    Xs = _scale(df_num)
    res = _predict_all(df_num, Xs)

    # 0/1 -> Vrai/Faux
    mapped = {m: LABELS[str(int(v[0]))] for m, v in res["classes"].items()}

    # Majority vote (sur les modèles supervisés)
    supervised = [int(res["classes"]["logreg"][0]),
                  int(res["classes"]["knn"][0]),
                  int(res["classes"]["rf"][0])]
    majority = int(round(sum(supervised) / len(supervised)))

    return {
        "input": billet.dict(),
        "prediction": {
            **mapped,
            "majority_vote": LABELS[str(majority)]
        },
        "avg_positive_probability": res["avg_positive_probability"],
    }

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), sample_rows: int = 10):
    """
    Upload d'un CSV (sans étiquette) -> prédictions Vrai/Faux pour chaque modèle,
    probabilités moyennes, comptages, et un aperçu des n premières lignes.
    """
    try:
        # Lecture + normalisation
        df_raw = _read_csv_upload(file)
        rows_received = int(df_raw.shape[0])

        df_num = _coerce_and_align(df_raw)
        before = df_num.shape[0]
        df_num = df_num.dropna(axis=0)  # sécurité finale si quelques NaN subsistent
        dropped = before - df_num.shape[0]
        if df_num.empty:
            return JSONResponse({"error": "Aucune ligne exploitable après nettoyage."}, status_code=400)

        # Prédictions
        Xs = _scale(df_num)
        res = _predict_all(df_num, Xs)

        # 0/1 -> Vrai/Faux
        mapped_preds = {model: [LABELS[str(int(v))] for v in preds]
                        for model, preds in res["classes"].items()}

        # Comptages Vrai/Faux par modèle
        def counts(arr):
            return {"Faux": arr.count("Faux"), "Vrai": arr.count("Vrai")}
        counts_dict = {k: counts(v) for k, v in mapped_preds.items()}

        # Échantillon de sortie
        sample = pd.DataFrame(mapped_preds).head(sample_rows)

        return {
            "rows_received": rows_received,
            "rows_used_for_prediction": int(df_num.shape[0]),
            "rows_dropped_after_cleaning": int(dropped),
            "counts": counts_dict,
            "avg_positive_probability": res["avg_positive_probability"],
            "sample_predictions_head": sample.to_dict(orient="records"),
            "all_predictions": mapped_preds
        }

    except Exception as e:
        return JSONResponse({"error": f"Erreur lors du traitement du fichier : {e}"}, status_code=400)
