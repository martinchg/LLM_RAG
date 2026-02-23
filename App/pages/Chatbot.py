import sys
import os
import base64
from datetime import datetime

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)

from utils.llm_client import query_llm
from utils.auth_local import require_login

from utils.config_loader import (
    load_config,
    list_history_files,
    load_history_for,
    save_history_for,
    user_folder,
    get_history_title,   # 👈 nouveau
)
from utils.history_utils import new_history_filename
from Embedding.search_and_rerank import search_and_rerank


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
config = load_config()

st.set_page_config(
    page_title=f"{config['app']['title']} - Discussion",
    page_icon="💬",
    layout="wide"
)

# Path to logo
logo_path = ROOT_DIR + "/App/utils/ineo.jpg"

# Build title + logo inline
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="display:flex; align-items:center;">
            <img src="data:image/png;base64,{logo_base64}"
                 style="width:100px; height:70px; margin-right:12px;"/>
            <h1 style="margin:0; padding:0;">{config['app']['title']} - Discussion</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title(f"{config['app']['title']} - Discussion")
    st.write("⚠️ Logo introuvable.")

# Subtitle
st.markdown("<h3>💬 Posez-moi une question...</h3>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------
require_login()
username = st.session_state["username"]
folder = user_folder(username)


# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Fichier JSON courant (None tant qu’aucun message sauvegardé)
if "current_history_file" not in st.session_state:
    st.session_state["current_history_file"] = None

# Titre de la discussion actuelle (affiché / sauvegardé)
if "chat_title" not in st.session_state:
    st.session_state["chat_title"] = "Nouveau chat"

# Titre éventuellement saisi par l'utilisateur dans "Nouveau chat"
if "pending_title" not in st.session_state:
    st.session_state["pending_title"] = None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Options")

    st.subheader("🧠 Mode RAG (Recherche + Re-ranking)")
    use_rag = st.checkbox("Activer la recherche documentaire", value=True)
    top_k = st.slider("Top-K Qdrant", 5, 40, 20)
    final_k = st.slider("Nombre final de passages", 1, 10, 3)
    threshold = st.slider("Seuil de pertinence", -15.0, 5.0, -7.0)

    # New chat
    st.subheader("🆕 Nouveau Chat")
    custom_title = st.text_input("Nom du chat (optionnel)")
    if st.button("Créer un nouveau chat"):
        # On ne crée PAS de fichier ici.
        # On prépare juste un état propre en mémoire.
        st.session_state["current_history_file"] = None
        st.session_state["messages"] = []
        st.session_state["pending_title"] = custom_title.strip() or None
        st.session_state["chat_title"] = custom_title.strip() or "Nouveau chat"
        st.rerun()

    # History
    st.subheader("📚 Historique")
    files = list_history_files(username)

    # Selectbox avec titres lisibles au lieu des noms de fichiers
    def _format_option(option: str) -> str:
        if option == "--":
            return "--"
        return get_history_title(username, option)

    selection = st.selectbox(
        "Sélectionnez une discussion",
        ["--"] + files,
        format_func=_format_option
    )

    if selection != "--" and selection != st.session_state["current_history_file"]:
        # Changement de discussion
        st.session_state["current_history_file"] = selection
        st.session_state["messages"] = load_history_for(username, selection)
        st.session_state["chat_title"] = get_history_title(username, selection)
        st.session_state["pending_title"] = None
        st.rerun()


# ---------------------------------------------------------
# Affichage du titre de la discussion en cours
# ---------------------------------------------------------
st.caption(f"📄 Discussion : {st.session_state.get('chat_title', 'Nouveau chat')}")


# ---------------------------------------------------------
# Show messages in chat
# ---------------------------------------------------------
for role, content, timestamp in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(content)
        st.caption(f"⏰ {timestamp}")


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------
prompt = st.chat_input("Posez-moi une question...")

if prompt:
    now = datetime.now().isoformat(timespec="seconds")

    # Enregistrer le message utilisateur en mémoire
    st.session_state["messages"].append((
        "user",
        prompt,
        now
    ))

    # -----------------------------------------------------
    # RAG PIPELINE
    # -----------------------------------------------------
    if use_rag:
        with st.spinner("Recherche dans la base documentaire..."):
            results, parents = search_and_rerank(
                query=prompt,
                top_k=top_k,
                final_k=final_k,
                threshold=threshold
            )

        # Format results for prompting
        response_final = "\n\n".join(
            f"- Chunk (rank={r['rank']}, score={r['rerank_score']:.2f}) : {r['chunk']}"
            for r in results
        )

        parents_text = "\n\n".join(
            f"- ParentID {pid} : {txt}"
            for pid, txt in parents.items()
        )

        # ---------------------------------------------------------
        # ANTI-HALLUCINATION RAG PROMPT ENGINEERING
        # ---------------------------------------------------------
        rag_prompt = f"""
Tu es un assistant IA chargé de répondre à une question uniquement à partir d’extraits documentaires fournis.

Voici la question à laquelle on souhaite répondre :
{prompt}

Voici les chunks possibles avec les identifiants des parents qui leur sont associés :
{response_final}

Voici les parents avec leurs identifiants :
{parents_text}

Consigne (procédure obligatoire) :
1. Sélectionne jusqu'à 2 extraits parmi ceux ayant les reranker_scores les plus élevés (ou moins s'il n'en existe pas 2).
2. Récupère les textes complets associés à ces extraits.
3. Analyse uniquement ces textes et identifie celui qui contient l'information permettant de répondre.
4. Réponds à la question uniquement à partir du contenu de ce texte.

Contraintes strictes :
• Réponds uniquement à partir des extraits fournis, sans rien inventer, extrapoler ou déduire d’informations externes.
• Si l’information exacte n’apparaît pas dans les textes sélectionnés, indique-le explicitement.
• Reformule au plus près des formulations du texte, sans ajouter de détails absents.
• Ne mentionne jamais dans la réponse les mots “chunk”, “parent”, “score”, “reranker_score”, ni la méthode de sélection.

FORMAT DE SORTIE OBLIGATOIRE (ne rien ajouter avant ou après) :

<réponse synthétisée, en une ou deux phrases maximum>

Source :
"<nom exact du document source>", page <numéro de page>
""".strip()

        final_prompt = rag_prompt

    else:
        final_prompt = prompt

    # -----------------------------------------------------
    # LLM RESPONSE
    # -----------------------------------------------------
    with st.spinner("Génération de la réponse..."):
        response = query_llm(final_prompt)

    now_resp = datetime.now().isoformat(timespec="seconds")
    st.session_state["messages"].append((
        "assistant",
        response,
        now_resp
    ))

    # -----------------------------------------------------
    # Création du fichier & titre si c’est le premier message
    # -----------------------------------------------------
    if st.session_state["current_history_file"] is None:
        # On crée un nom de fichier interne (l'utilisateur ne le verra pas)
        filename = new_history_filename(folder)
        st.session_state["current_history_file"] = filename

        # Déterminer le titre :
        # 1. Si l'utilisateur a saisi un titre dans "Nouveau Chat" → on le garde
        if st.session_state.get("pending_title"):
            chat_title = st.session_state["pending_title"]
        else:
            # 2. Sinon, on demande au LLM de proposer un titre court
            title_prompt = f"""
Tu es un assistant qui doit générer un titre très court (maximum 6 mots) pour résumer une discussion.
La discussion commence par cette question de l'utilisateur :

\"{prompt}\"

Donne uniquement le titre, sans guillemets, sans phrase supplémentaire, en français.
""".strip()
            chat_title = query_llm(title_prompt).strip()

        st.session_state["chat_title"] = chat_title

    # -----------------------------------------------------
    # Sauvegarde de l'historique dans le JSON
    # (jamais vide, puisqu'on arrive ici seulement s'il y a un prompt)
    # -----------------------------------------------------
    save_history_for(
        username,
        st.session_state["messages"],
        st.session_state["current_history_file"],
        title=st.session_state.get("chat_title")
    )

    st.rerun()
