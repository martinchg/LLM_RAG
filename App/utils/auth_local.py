import streamlit as st
import json
import os
import bcrypt

# Path to your users.json
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# ----------------------------------------------------------------------
# Ensure the users.json file exists
# ----------------------------------------------------------------------
def _ensure_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)

# Load users from JSON
def load_users():
    _ensure_file()
    with open(USERS_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

# Save users to JSON
def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# Password hashing
def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()

# Password verification
def check_password(plain_password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed.encode())
    except Exception:
        return False

# ----------------------------------------------------------------------
# Authentication widget (Login + Sign-Up)
# ----------------------------------------------------------------------
def login_widget():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    # If already logged in
    if st.session_state.authenticated:
        st.sidebar.success(f"Connecté · {st.session_state.username}")

        if st.sidebar.button("Se déconnecter"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        return True

    # ---------------------------
    # 🔐 LOGIN SECTION
    # ---------------------------
    st.sidebar.header("🔐 Connexion")

    username = st.sidebar.text_input("Utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")

    if st.sidebar.button("Se connecter"):
        users = load_users()
        user = users.get(username)

        if user and check_password(password, user.get("password", "")):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success(f"Bienvenue {username}")
            st.rerun()
        else:
            st.sidebar.error("Identifiants incorrects")

    st.sidebar.markdown("---")

    # ---------------------------
    # ➕ SIGN-UP SECTION
    # ---------------------------
    st.sidebar.header("➕ Créer un compte")

    new_user = st.sidebar.text_input("Nouvel utilisateur")
    new_pwd = st.sidebar.text_input("Nouveau mot de passe", type="password")

    if st.sidebar.button("Créer un compte"):
        if not new_user or not new_pwd:
            st.sidebar.error("Veuillez remplir tous les champs.")
        else:
            users = load_users()

            if new_user in users:
                st.sidebar.warning("❗ Cet utilisateur existe déjà.")
            else:
                hashed = hash_password(new_pwd)
                users[new_user] = {"password": hashed}
                save_users(users)
                st.sidebar.success(f"Utilisateur **{new_user}** créé avec succès !")

    return False
def login_form_inside_page():
    """
    Login UI that appears INSIDE a page (not in sidebar).
    Returns True if authenticated.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    # Already logged in
    if st.session_state.authenticated:
        return True

    st.markdown("### 🔐 Connexion")

    username = st.text_input("Utilisateur")
    password = st.text_input("Mot de passe", type="password")
    login_btn = st.button("Se connecter")

    if login_btn:
        users = load_users()
        user = users.get(username)

        if user and check_password(password, user["password"]):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Identifiants incorrects")

    st.markdown("---")

    # SIGN-UP SECTION
    st.markdown("### ➕ Créer un compte")

    new_user = st.text_input("Nouvel utilisateur")
    new_pwd = st.text_input("Nouveau mot de passe", type="password")
    signup_btn = st.button("Créer un compte")

    if signup_btn:
        if not new_user or not new_pwd:
            st.error("Veuillez remplir tous les champs.")
        else:
            users = load_users()
            if new_user in users:
                st.warning("❗ Cet utilisateur existe déjà.")
            else:
                users[new_user] = {"password": hash_password(new_pwd)}
                save_users(users)
                st.success(f"Utilisateur **{new_user}** créé avec succès !")

    return False


# ----------------------------------------------------------------------
# Require login for a page
# ----------------------------------------------------------------------
def require_login():
    if not login_widget():
        st.stop()
