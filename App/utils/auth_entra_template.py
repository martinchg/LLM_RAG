# App/utils/auth_entra_template.py
import streamlit as st
from urllib.parse import urlencode
import requests
import os

# NOTE: This is a simple template for OpenID Connect flow for Entra ID.
# In production prefer a library (authlib, streamlit-oauth) and proper token validation.

CONFIG = {
    "client_id": "",        # fill from config.toml or env
    "client_secret": "",    # fill from config or env
    "tenant_id": "",        # fill from config or env
    "redirect_uri": "http://localhost:8501",
    "scope": "openid profile email"
}

OAUTH_AUTHORIZE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
OAUTH_TOKEN = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
OAUTH_USERINFO = "https://graph.microsoft.com/oidc/userinfo"

def build_authorize_url(client_id, tenant, redirect_uri, scope, state="state123"):
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": scope,
        "state": state
    }
    return OAUTH_AUTHORIZE.format(tenant=tenant) + "?" + urlencode(params)

def exchange_code_for_token(code, client_id, client_secret, tenant, redirect_uri):
    token_url = OAUTH_TOKEN.format(tenant=tenant)
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    resp = requests.post(token_url, data=data)
    resp.raise_for_status()
    return resp.json()

def require_entra_login(client_id, client_secret, tenant, redirect_uri="http://localhost:8501"):
    """
    Basic template:
    - Redirect user to Microsoft login using authorize URL.
    - Microsoft redirects back with ?code=...
    - Exchange code for token, fetch userinfo.
    NOTE: Use libraries and validate tokens properly in prod.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    query_params = st.experimental_get_query_params()
    code = query_params.get("code", [None])[0]

    if st.session_state.authenticated:
        st.sidebar.success(f"Connecté · {st.session_state.username}")
        if st.sidebar.button("Se déconnecter"):
            st.session_state.clear()
            st.experimental_rerun()
        return True

    if not code:
        auth_url = build_authorize_url(client_id, tenant, redirect_uri, CONFIG["scope"])
        st.sidebar.markdown(f"[Se connecter avec Microsoft]({auth_url})")
        st.stop()
    else:
        # exchange code
        token_resp = exchange_code_for_token(code, client_id, client_secret, tenant, redirect_uri)
        access_token = token_resp.get("access_token")
        # optional: fetch userinfo using Microsoft Graph
        headers = {"Authorization": f"Bearer {access_token}"}
        userinfo = requests.get(OAUTH_USERINFO, headers=headers).json()
        st.session_state.authenticated = True
        st.session_state.username = userinfo.get("email") or userinfo.get("name")
        st.experimental_set_query_params()  # clear code from URL
        st.experimental_rerun()
