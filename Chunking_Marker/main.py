import json
import os
import logging
from pathlib import Path
from tqdm import tqdm

# Import de vos modules
from Files_ingestion import collecte_fichiers_locaux, copy_local_files
from Text_extraction import extract
from Chunking import create_parent_child_chunks

# 1. On récupère le dossier exact où se trouve main.py

try:
    BASE_DIR = Path(__file__).parent.resolve()
except NameError:
    # Sécurité : si tu lances le code directement dans une cellule Colab/Jupyter
    # au lieu d'exécuter le fichier .py, __file__ n'existe pas. On prend alors le dossier courant.
    BASE_DIR = Path(os.getcwd()).resolve()

# 2. On définit les chemins de manière relative par rapport à ce dossier
PATHS = {
    "files_to_collect": str(BASE_DIR / "files_to_collect"),
    "collected_files": str(BASE_DIR / "collected_files"),
    "output_dir": str(BASE_DIR)  # Sauvegarde les logs et les chunks dans le même dossier
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".mp3", ".wav", ".xlsx", ".xls"}

# --- VARIABLES DE CHUNKING ---
CHILD_SIZE = 200      # Taille des "Enfants" (utilisés pour la recherche/embedding)
PARENT_SIZE = 2000    # Taille max des "Parents" (contexte envoyé au LLM)

# --- Setup Logging ---
log_filename = os.path.join(PATHS["output_dir"], 'pipeline.log')
os.makedirs(PATHS["output_dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Utilitaires ---
def load_processed_files(output_file):
    """Lit le fichier final existant pour éviter de refaire le travail."""
    processed = set()
    if not os.path.exists(output_file):
        return processed
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # On suppose que c'est une liste de dicts
            for item in data:
                fname = item.get('filename')
                if fname:
                    processed.add(fname)
    except Exception:
        # Si le fichier est corrompu ou vide, on ignore
        pass
    return processed

def main():
    # 1. Préparation
    os.makedirs(PATHS["output_dir"], exist_ok=True)
    os.makedirs(PATHS["collected_files"], exist_ok=True)

    # 2. Ingestion
    logger.info("--- Démarrage de l'ingestion ---")
    fichiers = collecte_fichiers_locaux(PATHS["files_to_collect"])
    if fichiers:
        copy_local_files(fichiers, PATHS["collected_files"])
    
    # 3. Définition des fichiers de sortie FINAUX
    final_children_path = os.path.join(PATHS["output_dir"], "mes_chunks.json")
    final_parents_path = os.path.join(PATHS["output_dir"], "paragraphs.json")

    # Checkpoint simple : on regarde ce qui est déjà dans le fichier final enfants
    files_already_processed = load_processed_files(final_children_path)
    logger.info(f"Fichiers déjà traités : {len(files_already_processed)}")

    # 4. Pipeline
    all_files = [p for p in Path(PATHS["collected_files"]).rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files_to_process = [f for f in all_files if f.name not in files_already_processed]

    logger.info(f"Reste à traiter : {len(files_to_process)} fichiers")

    # On charge l'existant si on veut faire de l'"append" sur le JSON final
    new_parents = []
    new_children = []
    
    if files_to_process:
        progress_bar = tqdm(files_to_process, desc="Processing", unit="file")
        
        for file_path in progress_bar:
            try:
                progress_bar.set_postfix(current=file_path.name[:20])

                # A. Extraction
                elements, _ = extract(file_path)
                if not elements:
                    continue

                # B. Chunking (Retourne Parents ET Enfants séparés)
                # MODIFICATION ICI : On passe PARENT_SIZE
                file_parents, file_children = create_parent_child_chunks(
                    elements, 
                    parent_size_tokens=PARENT_SIZE, 
                    child_size_tokens=CHILD_SIZE
                )                
                
                if not file_children:
                    continue

                # C. Accumulation
                new_parents.extend(file_parents)
                new_children.extend(file_children)

            except Exception as e:
                logger.error(f"❌ Erreur sur {file_path.name}: {e}", exc_info=True)

        # 5. Fusion et Sauvegarde
        existing_children = []
        existing_parents = []

        if os.path.exists(final_children_path):
            with open(final_children_path, 'r', encoding='utf-8') as f:
                try: existing_children = json.load(f)
                except: pass
        
        if os.path.exists(final_parents_path):
            with open(final_parents_path, 'r', encoding='utf-8') as f:
                try: existing_parents = json.load(f)
                except: pass

        # Fusion
        final_children_data = existing_children + new_children
        final_parents_data = existing_parents + new_parents

        # Ecriture fichier ENFANTS (mes_chunks.json)
        logger.info(f"💾 Sauvegarde Enfants ({len(final_children_data)} chunks)...")
        with open(final_children_path, 'w', encoding='utf-8') as f:
            json.dump(final_children_data, f, indent=2, ensure_ascii=False)

        # Ecriture fichier PARENTS (paragraphs.json)
        logger.info(f"💾 Sauvegarde Parents ({len(final_parents_data)} parents)...")
        with open(final_parents_path, 'w', encoding='utf-8') as f:
            json.dump(final_parents_data, f, indent=2, ensure_ascii=False)

        logger.info("✅ Terminé avec succès !")
    else:
        logger.info("Rien de nouveau à traiter.")

if __name__ == "__main__":
    main()