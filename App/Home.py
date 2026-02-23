import streamlit as st
from utils.auth_local import login_form_inside_page

st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")

# -------------------------------------------------
# Centered Welcome Banner
# -------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:30px;'>
        <h1 style='font-size:40px;'>🏠Bienvenue</h1>
        <h2 style='color:#4A90E2;'>
            INEO Defense × IMT Atlantique – RAG Chatbot
        </h2>
        <p style='font-size:18px; margin-top:10px;'>
            Veuillez vous connecter ou créer un compte pour continuer.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Login form inside the page
# -------------------------------------------------
if login_form_inside_page():
    st.switch_page("pages/Chatbot.py")
