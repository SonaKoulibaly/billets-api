# streamlit_app.py — S1 (base + thème + health check + footer)
import streamlit as st
import requests
import pandas as pd
import numpy as np
import io

# ---------- Page config ----------
st.set_page_config(
    page_title="Billets — Vrai/Faux",
    page_icon="💶",
    layout="wide"
)

# ---------- Styles personnalisés (header, cartes, footer) ----------
st.markdown("""
<style>
/* Hero gradient */
.hero {
  background: linear-gradient(135deg, #6C63FF 0%, #1E88E5 100%);
  padding: 28px 28px;
  border-radius: 18px;
  color: white;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
  margin-bottom: 14px;
}

/* Petit badge "online/offline" */
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  margin-left: 8px;
}

/* Cartes info */
.card {
  background: #131A35;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

/* Boutons plus denses */
button[kind="primary"] {
  border-radius: 999px !important;
}

/* Footer fixé */
.footer {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  padding: 8px 16px;
  background: rgba(11,16,33,0.9);
  color: #AAB3D1;
  font-size: 13px;
  border-top: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(6px);
  text-align: center;
  z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
  <h1 style="margin:0;">💶 Détection de Vrais/Faux Billets</h1>
  <p style="opacity:.95;margin:6px 0 0;">
    Dévéloppé par SONA KOULIBALY | Interface Système Machine L | pour l'authentification des billets.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar : config API ----------
st.sidebar.header("⚙️ Paramètres API")
api_url_default = "http://127.0.0.1:8000"
api_url = st.sidebar.text_input("URL de l’API FastAPI", value=api_url_default, help="Ex: http://127.0.0.1:8000")
test_btn = st.sidebar.button("Tester la connexion")

# ---------- Health check ----------
api_ok = False
health_info = {}
if test_btn:
    try:
        r = requests.get(f"{api_url}/health", timeout=8)
        if r.ok:
            api_ok = True
            health_info = r.json()
            st.sidebar.success("API en ligne ✅")
        else:
            st.sidebar.error(f"API non joignable: {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion : {e}")

# Affichage d’un résumé santé si dispo
col1, col2 = st.columns([1,1])
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📡 État de l’API")
    if api_ok:
        st.markdown(f"- **URL**: `{api_url}`")
        st.markdown(f"- **Endpoints**: `/health`, `/predict_one`, `/predict_csv`")
        st.markdown(f"- **Modèles**: `{', '.join(health_info.get('models', []))}`")
        st.markdown(f"- **Colonnes attendues**: `{', '.join(health_info.get('expected_cols', []))}`")
        st.success("Tout est prêt pour la suite !")
    else:
        st.info("Clique à gauche sur **Tester la connexion** pour vérifier l’API.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧭 Prochaines étapes")
    st.markdown("""
    - **S2** : Formulaire *1 billet* → appel `/predict_one` (affichage clair + vote majoritaire)  
    - **S3** : Upload **CSV** → appel `/predict_csv` (table, stats, graphes)  
    - **S4** : Export des résultats (CSV) et petites améliorations UX  
    - **S5** : Déploiement **Streamlit Cloud**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- S2 — Formulaire “1 billet” ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Prédire sur un seul billet")

# Astuce UI : petit helper pour afficher "Vrai/Faux" en badge
def verdict_badge(label: str) -> str:
    # label attendu: "Vrai" ou "Faux"
    color = "#22c55e" if label.lower() == "vrai" else "#ef4444"
    return f'<span class="badge" style="background:{color};color:white;">{label}</span>'

with st.form("predict_one_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        length = st.number_input("length (mm)", value=135.6, step=0.1, format="%.2f")
        margin_low = st.number_input("margin_low (mm)", value=4.2, step=0.1, format="%.2f")
    with c2:
        height_left = st.number_input("height_left (mm)", value=74.2, step=0.1, format="%.2f")
        margin_up = st.number_input("margin_up (mm)", value=4.0, step=0.1, format="%.2f")
    with c3:
        height_right = st.number_input("height_right (mm)", value=74.0, step=0.1, format="%.2f")
        diagonal = st.number_input("diagonal (mm)", value=145.5, step=0.1, format="%.2f")

    submitted = st.form_submit_button("🚀 Lancer la prédiction")
    if submitted:
        payload = {
            "length": float(length),
            "height_left": float(height_left),
            "height_right": float(height_right),
            "margin_low": float(margin_low),
            "margin_up": float(margin_up),
            "diagonal": float(diagonal),
        }

        if not api_ok:
            st.warning("ℹ️ Pense à **tester la connexion** dans la barre latérale avant d’appeler l’API.")
        try:
            r = requests.post(f"{api_url}/predict_one", json=payload, timeout=25)
            r.raise_for_status()
            data = r.json()

            # --------- Présentation des résultats ---------
            st.success("✅ Prédiction reçue")
            st.markdown("**Entrée envoyée**")
            st.write(pd.DataFrame([data.get("input", payload)]))

            pred = data.get("prediction", {})
            probs = data.get("avg_positive_probability", {})

            # Ligne de badges Vrai/Faux + vote majoritaire
            st.markdown("**Résultat (par modèle)**")
            cols = st.columns(5)
            model_labels = [
                ("Régression Logistique", pred.get("logreg", "—")),
                ("KNN", pred.get("knn", "—")),
                ("Random Forest", pred.get("rf", "—")),
                ("K-Means (mappé)", pred.get("kmeans", "—")),
                ("🗳️ Vote majoritaire", pred.get("majority_vote", "—")),
            ]
            for i, (title, lbl) in enumerate(model_labels):
                with cols[i]:
                    html = f"""
                    <div style="background:#0b1021;border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:12px;">
                      <div style="font-size:13px;opacity:.85;margin-bottom:6px;">{title}</div>
                      <div style="font-size:18px;font-weight:700;">{verdict_badge(str(lbl))}</div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)

            # Tableau résumé (probas)
            st.markdown("**Probabilités moyennes positives (si disponibles)**")
            df_probs = pd.DataFrame([{
                "Modèle": "Régression Logistique", "Proba Classe 1": probs.get("logreg", None)
            }, {
                "Modèle": "KNN", "Proba Classe 1": probs.get("knn", None)
            }, {
                "Modèle": "Random Forest", "Proba Classe 1": probs.get("rf", None)
            }])
            # Formattage sympa
            df_probs["Proba Classe 1"] = df_probs["Proba Classe 1"].apply(
                lambda x: f"{x:.2%}" if isinstance(x, (float, int)) else "—"
            )
            st.dataframe(df_probs, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erreur d’appel API : {e}")
        except Exception as e:
            st.error(f"❌ Erreur interne : {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- S3 — Upload CSV -> /predict_csv ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📥 Tester un fichier CSV (prédictions en lot)")
st.caption("Le CSV doit contenir : length, height_left, height_right, margin_low, margin_up, diagonal")

with st.form("csv_form", clear_on_submit=False):
    uploaded_file = st.file_uploader("Dépose ton fichier CSV ici", type=["csv"])
    do_predict = st.form_submit_button("🚀 Envoyer au modèle")

if do_predict and uploaded_file:
    try:
        # Envoi du fichier à l'API (multipart/form-data)
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        r = requests.post(f"{api_url}/predict_csv", files=files, timeout=120)
        r.raise_for_status()
        data = r.json()

        # ---- KPIs / métriques
        rows_received = int(data.get("rows_received", 0))
        used         = int(data.get("rows_used_for_prediction", 0))
        dropped      = int(data.get("rows_dropped_after_cleaning", 0))

        k1, k2, k3 = st.columns(3)
        with k1: st.metric("📄 Lignes reçues", rows_received)
        with k2: st.metric("✅ Lignes utilisées", used)
        with k3: st.metric("🧹 Lignes écartées", dropped)

        st.divider()

        # ---- Répartition des prédictions par modèle (Vrai/Faux)
        st.markdown("### 📊 Répartition des prédictions par modèle")

        # L’API peut renvoyer 'counts_labels' (Vrai/Faux) OU 'counts' (0/1).
        counts = data.get("counts_labels") or data.get("counts")
        if counts:
            df_counts = pd.DataFrame(counts).T  # index = modèles
            # Harmonise les colonnes en Vrai/Faux
            if "Vrai" in df_counts.columns or "Faux" in df_counts.columns:
                if "Vrai" not in df_counts: df_counts["Vrai"] = 0
                if "Faux" not in df_counts: df_counts["Faux"] = 0
                df_counts = df_counts[["Vrai", "Faux"]]
            else:
                df_counts = df_counts.rename(columns={"1": "Vrai", "0": "Faux"})
                for c in ["Vrai", "Faux"]:
                    if c not in df_counts: df_counts[c] = 0
                df_counts = df_counts[["Vrai", "Faux"]]

            st.dataframe(df_counts, use_container_width=True, height=240)
            st.bar_chart(df_counts, use_container_width=True)

        else:
            st.info("Pas de 'counts' dans la réponse. Vérifie que l’API renvoie bien les répartitions.")

        st.divider()

        # ---- Probabilités moyennes positives
        probs = data.get("avg_positive_probability")
        if probs:
            st.markdown("### 📈 Probabilités moyennes positives (classe « Vrai »)")
            df_probs = (
                pd.Series(probs)
                .to_frame("Proba moyenne")
                .applymap(lambda x: round(float(x) * 100, 2) if x is not None else None)
            )
            st.dataframe(df_probs, use_container_width=True)

        # ---- Aperçu des premières prédictions (head)
        sample = data.get("sample_predictions_head_labels") or data.get("sample_predictions_head")
        if sample:
            st.markdown("### 🔎 Aperçu des premières prédictions")
            df_sample = pd.DataFrame(sample)
            # Map 0/1 -> Vrai/Faux si besoin
            for col in df_sample.columns:
                if col.startswith("pred_"):
                    try:
                        df_sample[col] = df_sample[col].map({1: "Vrai", 0: "Faux"}).fillna(df_sample[col])
                    except Exception:
                        pass
            st.dataframe(df_sample, use_container_width=True, height=260)
        else:
            st.info("Pas d’aperçu renvoyé par l’API (clé 'sample_predictions_head*').")

    except Exception as e:
        st.error(f"❌ Erreur lors de l'appel à /predict_csv : {e}")

elif do_predict and not uploaded_file:
    st.warning("Ajoute d'abord un fichier CSV avant de lancer la prédiction.")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# =========================
# S4 — Upload CSV (pro), stats, graphes, export (auto-séparateur)
# =========================
import base64

st.markdown("## 📥Upload CSV : tableau enrichi, graphes et export...")
st.caption("Charge un fichier (.csv / .tsv / .txt) avec les colonnes : diagonal, height_left, height_right, margin_low, margin_up, length")

# --- helper badges HTML ---
def badge(val_int: int) -> str:
    return (
        '<span style="background:#16a34a;color:white;padding:4px 8px;border-radius:999px;font-size:12px;">Vrai</span>'
        if int(val_int) == 1 else
        '<span style="background:#dc2626;color:white;padding:4px 8px;border-radius:999px;font-size:12px;">Faux</span>'
    )

def to_badge_df(sample_preds: pd.DataFrame) -> pd.DataFrame:
    df = sample_preds.copy()
    for col in df.columns:
        if col.lower().startswith("pred_"):
            df[col] = df[col].apply(lambda x: badge(x))
    return df

with st.expander("➕ Importer un fichier et lancer la prédiction", expanded=True):
    csv_file = st.file_uploader(
        "Sélectionne ton fichier (.csv / .tsv / .txt)",
        type=["csv", "tsv", "txt"]
    )
    run_btn = st.button("🚀 Lancer la prédiction", use_container_width=True)

# (Optionnel) petit aperçu local avec détection automatique du séparateur
if csv_file is not None and not run_btn:
    try:
        preview_df = pd.read_csv(csv_file, sep=None, engine="python")
        st.markdown("**Aperçu (auto-détection du séparateur)**")
        st.dataframe(preview_df.head(20), use_container_width=True)
    except Exception:
        st.info("Aperçu indisponible (format/encodage atypique). Tu peux quand même lancer la prédiction.")

if csv_file is not None and run_btn:
    try:
        # On envoie tel quel à l’API (qui auto-détecte aussi le séparateur côté serveur)
        mime_guess = "text/plain"
        if csv_file.name.lower().endswith(".csv"):
            mime_guess = "text/csv"
        files = {"file": (csv_file.name, csv_file.getvalue(), mime_guess)}

        r = requests.post(f"{api_url}/predict_csv", files=files, timeout=120)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        st.error(f"❌ Erreur d’appel API /predict_csv : {e}")
    else:
        if "error" in resp:
            st.error(f"❌ API a répondu une erreur : {resp['error']}")
        else:
            # ---- résumé lignes ----
            rows_received = int(resp.get("rows_received", 0))
            rows_used     = int(resp.get("rows_used_for_prediction", 0))
            rows_dropped  = int(resp.get("rows_dropped_after_cleaning", 0))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("📥 Lignes reçues", f"{rows_received}")
            with c2:
                st.metric("✅ Lignes utilisées", f"{rows_used}")
            with c3:
                st.metric("🧹 Lignes écartées", f"{rows_dropped}")

            st.markdown("---")

            # ---- répartition par modèle (counts) ----
            counts = resp.get("counts", None)
            if counts:
                st.subheader("📊 Répartition des prédictions par modèle")
                rows = []
                for model, dct in counts.items():
                    rows.append({"modèle": model, "classe": "Faux (0)", "count": int(dct.get("0", 0))})
                    rows.append({"modèle": model, "classe": "Vrai (1)", "count": int(dct.get("1", 0))})
                df_counts = pd.DataFrame(rows)

                g1, g2 = st.columns([2, 1])
                with g1:
                    pivot = df_counts.pivot(index="modèle", columns="classe", values="count").fillna(0)
                    st.bar_chart(pivot)
                with g2:
                    st.dataframe(df_counts, use_container_width=True)

            # ---- probabilités moyennes (si dispo) ----
            avg_probs = resp.get("avg_positive_probability", {})
            if avg_probs:
                st.subheader("📈 Probabilités positives moyennes (si disponibles)")
                df_probs = pd.DataFrame([
                    {"modèle": k, "proba_moyenne_classe_1": float(v) if v is not None else None}
                    for k, v in avg_probs.items()
                ])
                st.dataframe(df_probs, use_container_width=True)

            st.markdown("---")

            # ---- échantillon de prédictions (badges jolis) ----
            sample = resp.get("sample_predictions_head", [])
            if sample:
                st.subheader("🔎 Aperçu des premières prédictions")
                df_sample = pd.DataFrame(sample)
                df_badges = to_badge_df(df_sample)
                st.markdown(df_badges.to_html(escape=False, index=False), unsafe_allow_html=True)

            # ---- export CSV (si l’API renvoie les classes complètes) ----
            st.markdown("---")
            st.subheader("📤 Export des résultats (si prédictions complètes disponibles)")

            full_classes = None
            if isinstance(resp.get("classes"), dict):
                full_classes = resp["classes"]
            elif isinstance(resp.get("predictions"), dict):
                full_classes = resp["predictions"]
            elif isinstance(resp.get("predictions", {}).get("classes"), dict):
                full_classes = resp["predictions"]["classes"]

            if full_classes:
                df_export = pd.DataFrame({f"pred_{k}": v for k, v in full_classes.items()})
                labels = {0: "Faux", 1: "Vrai"}
                for col in df_export.columns:
                    df_export[col] = pd.Series(df_export[col]).map(labels)

                st.dataframe(df_export.head(20), use_container_width=True)

                csv_bytes = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Télécharger toutes les prédictions (CSV)",
                    data=csv_bytes,
                    file_name="predictions_completes.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info(
                    "Les prédictions complètes par ligne ne sont pas renvoyées par l’API actuelle. "
                    "Tu peux garder ce S4 (résumés + graphes) ou faire évoluer `/predict_csv` pour renvoyer toutes les classes."
                )



# ---------- Footer (branding) ----------
st.markdown(
    '<div class="footer">Projet Data Machine L Done by <strong>Sona KOULIBALY</strong> — Streamlit × FastAPI × ML</div>',
    unsafe_allow_html=True
)

