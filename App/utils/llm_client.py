# App/utils/llm_client.py
import os
from utils.config_loader import load_config

# ------------------------------------------------------------------
# Detect Windows host IP when inside WSL so we can reach Ollama
# ------------------------------------------------------------------
def get_windows_ip():
    """
    In WSL, the Windows host IP is the default gateway.
    On native Linux/macOS/Windows (no WSL), fallback to localhost.
    """
    try:
        ip = os.popen("ip route | grep default | awk '{print $3}'").read().strip()
        if ip:
            return ip
    except Exception:
        pass
    return "localhost"  # Fallback for non-WSL or error cases

WINDOWS_IP = get_windows_ip()
OLLAMA_URL = f"http://{WINDOWS_IP}:11434"

# Make sure the ollama client uses the correct host
os.environ["OLLAMA_HOST"] = OLLAMA_URL

# 👉 import ollama AFTER setting OLLAMA_HOST
import ollama

# Load configuration
config = load_config()

# System message to avoid hallucinations and protect privacy
SYSTEM_PROMPT = (
    "Tu es un assistant sécurisé. "
    "Tu n'as accès qu'aux informations fournies dans ce message. "
    "Tu ne peux pas voir, récupérer ou évoquer les conversations passées "
    "des utilisateurs ou des conversations d'autres utilisateurs. "
    "Ne fais jamais de suppositions ou d'inventions à propos d'autres utilisateurs. "
    "Si on te demande des informations sur d'autres utilisateurs, ou leur historique "
    "de discussion, réponds simplement : "
    "'⛔ Désolé, je ne peux afficher que votre propre historique.' "
    "Sinon, réponds normalement."
)

# Create a client explicitly bound to the right host
client = ollama.Client(host=OLLAMA_URL)

def query_llm(prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    model_name = config["llm"]["model"]

    try:
        response = client.chat(
            model=model_name,
            messages=messages,
            stream=False,
        )
    except Exception as e:
        return (
            f"⚠️ Erreur de connexion au modèle ({model_name}).\n"
            f"Assurez-vous que Ollama est en cours d'exécution sur Windows.\n\n"
            f"WSL essaie de se connecter à : {OLLAMA_URL}\n\n"
            f"Détails : {e}"
        )

    # -----------------------------
    # FIXED: extract the assistant answer properly
    # -----------------------------
    # New Python Ollama client format
    try:
        return response.message.content
    except:
        pass

    # Old format (dict)
    try:
        return response["message"]["content"]
    except:
        pass

    # Fallback (debug)
    return str(response)
