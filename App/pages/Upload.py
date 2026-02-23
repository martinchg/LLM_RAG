import streamlit as st
import sys
import time
from pathlib import Path
import subprocess

from utils.auth_local import require_login

# ============================================================
# Config page + Auth globale
# ============================================================

st.set_page_config(page_title="Documents (Admin)", page_icon="📂")
require_login()

st.title("📂 Gestion des documents – Admin")

# ============================================================
# 🔐 Restriction ADMIN UNIQUEMENT
# ============================================================

username = st.session_state.get("username")

if username != "admin":
    st.error("⛔ Accès réservé à l'administrateur.")
    st.stop()

st.success("✅ Accès administrateur autorisé")

# ============================================================
# Détection robuste de la racine du projet
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE

while not (ROOT_DIR / "chunking").exists():
    ROOT_DIR = ROOT_DIR.parent

sys.path.insert(0, str(ROOT_DIR))

# ============================================================
# Import du pipeline de chunking
# ============================================================

from Chunking.main_chunking import main as chunking_main
# adapter en chunking.main_chunking si besoin

# ============================================================
# Dossiers
# ============================================================

NEW_DOCS_DIR = ROOT_DIR / "nouveaux_documents"
BASE_DOCS_DIR = ROOT_DIR / "base_documents"

NEW_DOCS_DIR.mkdir(exist_ok=True)
BASE_DOCS_DIR.mkdir(exist_ok=True)

# ============================================================
# Upload de documents
# ============================================================

st.subheader("📥 Ajouter un document à la base")

uploaded_file = st.file_uploader(
    "Formats supportés : PDF, DOCX, TXT",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    save_path = NEW_DOCS_DIR / uploaded_file.name

    if save_path.exists():
        st.warning("⚠️ Ce fichier est déjà présent.")
    else:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ `{uploaded_file.name}` ajouté (en attente de chunking)")

st.divider()

# ============================================================
# 🔄 Chunking incrémental
# ============================================================

st.subheader("🔄 Chunking incrémental")

if st.button("Lancer le chunking des nouveaux documents"):
    with st.spinner("🧠 Chunking en cours..."):
        start = time.time()
        chunking_main()
        elapsed = time.time() - start

    st.success("✅ Chunking terminé")
    st.info(f"⏱️ Temps de traitement : **{elapsed:.2f} secondes**")

st.divider()

# ============================================================
# 🧠 Re-vectorisation (Qdrant)
# ============================================================

st.subheader("🧠 Re-vectorisation des chunks (Qdrant)")

if st.button("Re-vectoriser tous les chunks"):
    with st.spinner("📡 Vectorisation + insertion Qdrant en cours..."):
        start = time.time()

        # Appel du script d'indexation
        result = subprocess.run(
            [sys.executable, str(ROOT_DIR / "Embedding" / "indexation_database.py")],
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start

    if result.returncode == 0:
        st.success("✅ Re-vectorisation terminée avec succès")
        st.info(f"⏱️ Temps de traitement : **{elapsed:.2f} secondes**")
        st.text_area("📄 Logs", result.stdout, height=200)
    else:
        st.error("❌ Erreur lors de la re-vectorisation")
        st.text_area("📄 Logs d'erreur", result.stderr, height=200)
