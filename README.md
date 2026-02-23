# Structure du projet

```bash
LLM_RAG_2/
├── App/                          # Application Streamlit (interface utilisateur)
│   ├── Home.py                   # Point d’entrée principal de l’application
│   ├── config.toml               # Fichier de configuration de l’application
│   ├── pages/                    # Application Streamlit multi-pages
│   │   ├── Chatbot.py            # Interface de chat (requêtes RAG)
│   │   ├── Logs.py               # Consultation des logs
│   │   ├── Settings.py           # Paramètres de l’application
│   │   └── Upload.py             # Upload et ingestion de documents
│   └── utils/                    # Fonctions utilitaires
│       ├── auth_local.py         # Authentification locale
│       ├── auth_entra_template.py# Modèle d’authentification Microsoft Entra ID
│       ├── config_loader.py      # Chargement de la configuration
│       ├── create_users.py       # Script de création des utilisateurs
│       ├── history_utils.py      # Gestion de l’historique des conversations
│       ├── llm_client.py         # Client LLM / Ollama
│       ├── users.json            # Base utilisateurs locale
│       ├── history/              # Historique des conversations
│       ├── ineo.jpg              # logo
│       └──ollama_logo.png
│
├── Chunking/
│   ├── __init__.py                     # Prétraitement et découpage des documents
│   ├── agentic_chunker_ollama.py # Chunking sémantique via Ollama
│   ├── main_chunking.py            # Lancement du pipeline de chunking
│   ├── registry.py
│   ├── process_paragraph.py
│   └── file_readers.py
│
│
│
│
├── Embedding/                    # Embeddings, indexation et recherche
│   ├── indexation_database.py    # Indexation des embeddings dans Qdrant
│   ├── search_and_rerank.py      # Recherche sémantique + re-ranking
│   ├── re_ranker_simple.py       # Re-ranking simple
│   ├── Deepeval1.py              # Évaluation du Pipeline (DeepEval)
│   ├── benchmark.py              # Benchmark des modèles d’embedding
│   ├── benchmark.ipynb           # Notebook de benchmark
│   ├── embedding.ipynb           # Expérimentations embeddings
│   ├── Info_Qdrant_et_transformers.py
│   ├── docker-compose.yml        # Services liés aux embeddings / Qdrant
│   ├── paragraphs.json           # Données par paragraphes
│   └── README.md                 # Documentation spécifique Embedding
│
├── .gitignore
└── README.md                     # Documentation principale du projet
```

# Architecture générale

Le projet est structuré autour de trois briques principales :

- __Chunking__ : préparation et découpage sémantique des documents sources
- __Embedding__ : vectorisation, indexation dans Qdrant, recherche et re-ranking
- __App__ : interface utilisateur Streamlit permettant l’upload de documents,
  l’interrogation du système RAG et la consultation de l’historique

L’ensemble du projet est conteneurisé avec Docker afin de garantir une exécution
identique sur macOS, Windows et Linux.

# Librairies

```bash
Dans le fichier requirements.txt
```

# Pour lancer le projet en local (sans docker)

- Se placer sur LLM_RAG_2 si ce n'est pas déjà fait
- Lancer docker et qdrant avec "docker run -p 6333:6333 qdrant/qdrant" sur un terminal tout seul
- 
- Taper ensuite les commandes :

  ```bash
  > ollama pull mistral
  > streamlit run App/Home.py
  ```

- Dans la page upload vous pouvez uploader vos documents et les chunker puis faire l'indexation dans Qdrant
- Dans la page Chatbot vous pouvez poser vos questions au chatbot
