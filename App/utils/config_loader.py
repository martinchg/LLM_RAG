import toml
from functools import lru_cache
import json
import os
import hashlib

# Base directory: project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


@lru_cache(maxsize=1)
def load_config():
    config_path = os.path.join(BASE_DIR, "config.toml")
    return toml.load(os.path.abspath(config_path))


# 🔐 Build a stable hashed folder per user
def user_folder(username: str) -> str:
    uname = username.strip() or "anonymous"
    uid_hash = hashlib.sha256(uname.encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(BASE_DIR, "utils", "history", uid_hash)
    os.makedirs(folder, exist_ok=True)
    return folder


# List user history files
def list_history_files(username: str):
    folder = user_folder(username)
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.endswith(".json")])


# 🧠 Charger les messages d’une discussion
# Compatibilité :
#  - anciens fichiers : une simple liste de messages
#  - nouveaux fichiers : {"title": ..., "messages": [...]}
def load_history_for(username: str, filename: str):
    folder = user_folder(username)
    path = os.path.join(folder, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ancien format : liste de messages
            if isinstance(data, list):
                return data

            # Nouveau format : dict avec "messages"
            if isinstance(data, dict):
                return data.get("messages", [])
        except json.JSONDecodeError:
            return []
    return []


# 🔎 Récupérer uniquement le titre d’une discussion
def get_history_title(username: str, filename: str) -> str:
    folder = user_folder(username)
    path = os.path.join(folder, filename)

    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return "(Discussion)"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ancien format : pas de titre
        if isinstance(data, list):
            return "(Ancienne discussion)"

        if isinstance(data, dict):
            title = data.get("title")
            if title and isinstance(title, str) and title.strip():
                return title.strip()
            return "(Discussion sans titre)"
    except json.JSONDecodeError:
        return "(Discussion corrompue)"

    return "(Discussion)"


# 💾 Sauvegarder une discussion
# - messages : liste de (role, content, timestamp)
# - title : optionnel, None pour garder l’ancien comportement
def save_history_for(username: str, messages, filename: str, title: str | None = None):
    folder = user_folder(username)
    path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)

    # Si on nous passe déjà un dict complet, on le respecte
    if isinstance(messages, dict):
        data = messages
    else:
        data = {
            "title": title,
            "messages": messages,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
