import streamlit as st
from utils.auth_local import require_login
from utils.config_loader import list_history_files, load_history_for, user_folder

st.set_page_config(page_title="Logs", page_icon="📜")

st.title("📜 Logs des conversations")

require_login()
username = st.session_state["username"]

# List all history files for this user
files = list_history_files(username)

if not files:
    st.info("Aucune conversation enregistrée.")
    st.stop()

# Select a history file
selected = st.selectbox("Sélectionner une conversation :", files)

# Load messages from selected file
messages = load_history_for(username, selected)

st.write(f"### 💬 Conversation : {selected}")
st.markdown("---")

# Display messages
for role, content, timestamp in messages:
    with st.chat_message(role):
        st.write(content)
        st.caption(f"⏰ {timestamp}")
