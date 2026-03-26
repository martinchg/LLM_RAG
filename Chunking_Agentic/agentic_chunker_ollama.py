import subprocess
import time

AGENTIC_PROMPT = """Tu es un assistant spécialisé dans la segmentation sémantique de texte.
Ta tâche est de découper le paragraphe fourni en plusieurs sous-parties cohérentes appelées "chunks".

Règles :
- NE RÉÉCRIS PAS le texte. Garde exactement les mêmes mots, la même ponctuation et les mêmes espaces.
- Découpe uniquement là où il est logique de séparer des idées (phrases, sous-idées, transitions...).
- Chaque chunk doit pouvoir être compris indépendamment mais sans perte de sens globale.
- Évite les chunks trop courts (< 50 caractères) ou trop longs (> 800 caractères) sauf si nécessaire.
- Le texte final doit rester identique après recomposition de tous les chunks.

Format de sortie :
Écris les chunks EXACTEMENT sous la forme suivante, séparés par des barres obliques "/".
Ne mets ni texte explicatif ni phrase avant ou après.

Exemple :
Input: "Marie aime les pommes. Elle vit à Paris. Elle travaille dans une librairie."
Output: "Marie aime les pommes./Elle vit à Paris./Elle travaille dans une librairie."

Voici le texte à découper :
{paragraph}
"""


class AgenticChunker:
    """
    Le LLM décide du découpage des chunks.
    Chaque chunk hérite :
    - d’un parent_id unique (identifiant du paragraphe)
    - du numéro de page
    - du nom du document
    """

    def __init__(self, model="mistral", delay=1.0):
        self.model = model
        self.delay = delay
        self.chunks = {}

    # -----------------------------------------------------------
    # 1. Appel au modèle Ollama
    # -----------------------------------------------------------
    def query_llm(self, prompt):
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur Ollama : {e.stderr}")
            return ""

    # -----------------------------------------------------------
    # 2. Création des chunks
    # -----------------------------------------------------------
    def chunk_paragraph(self, paragraph_text, document_name, page_number, parent_id):
        """
        Découpe le texte via LLM et ajoute les métadonnées :
        - ID parent unique
        - nom du document
        - numéro de page
        """
        
        prompt = AGENTIC_PROMPT.format(paragraph=paragraph_text)
        response = self.query_llm(prompt)
        raw_chunks = [c.strip() for c in response.split("/") if c.strip()]

        chunks_list = []
        for chunk_text in raw_chunks:
            chunks_list.append({
                "parent_paragraph_id": parent_id,   # ✅ ID unique du paragraphe
                "page_number": page_number,         # ✅ numéro de page
                "document_name": document_name,
                "text": chunk_text
            })

        self.chunks = {i: c for i, c in enumerate(chunks_list)}
        time.sleep(self.delay)
        return chunks_list

    # -----------------------------------------------------------
    # 3. Affichage
    # -----------------------------------------------------------
    def pretty_print_chunks(self):
        print("\n----- Chunks créés -----\n")
        for c in self.chunks.values():
            print(f"📄 Parent {c['parent_paragraph_id']} | Page {c['page_number']} | {c['document_name']}")
            print(c["text"])
            print("-" * 60)