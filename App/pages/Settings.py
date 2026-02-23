import streamlit as st
from utils.config_loader import load_config
from utils.auth_local import require_login

# ✅ Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Paramètres",
    page_icon="⚙️",
    layout="wide"
)

# ✅ Require login AFTER set_page_config
require_login()

config = load_config()

st.title("⚙️ Paramètres")
st.subheader("Modèle LLM")

st.write(f"Modèle actuel : **{config['llm']['model']}**")
st.write(f"Host : {config['llm']['host']}")
