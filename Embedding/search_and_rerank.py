from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from pathlib import Path

print("Initialisation des modèles et connexion à Qdrant...")


ROOT_DIR = Path(__file__).resolve().parents[1]
PAR_PATH = ROOT_DIR / "paragraphs.json"

# ============================================================
# 1. Initialisation
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Utilisation du périphérique : {device.upper()}")

# Connexion Qdrant (local)
client = QdrantClient("localhost", port=6333)

# Modèle d’embedding (même que celui utilisé à l’indexation)
embedder = SentenceTransformer("BAAI/bge-m3", device=device)

# Modèle de re-ranking
reranker_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(reranker_name)
reranker = AutoModelForSequenceClassification.from_pretrained(reranker_name).to(device).eval()

print("✅ Modèles chargés et connexion à Qdrant établie.")


# ============================================================
# 2. Fonction de recherche et de re-ranking
# ============================================================

def search_and_rerank(query: str, top_k: int = 20, final_k: int = 3, threshold: float = -7.0):
    """
    Recherche les chunks les plus pertinents pour une requête donnée,
    puis applique un re-ranking via cross-encoder.
    """
    if not query.strip():
        return []

    # --- Étape 1 : embedding du texte de la requête ---
    query_emb = embedder.encode(query, normalize_embeddings=True).tolist()

    # --- Étape 2 : recherche sémantique initiale ---
    hits = client.search(
        collection_name="documents_chunks",   # ✅ nouvelle collection
        query_vector=query_emb,
        limit=top_k
    )
    if not hits:
        return []

    # --- Étape 3 : re-ranking cross-encoder ---
    docs = [hit.payload["Chunk"] for hit in hits]
    inputs = tokenizer(
        [query] * len(docs),
        docs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = reranker(**inputs).logits.squeeze(dim=1)

    scores = logits.detach().cpu().numpy().tolist()

    # --- Étape 4 : tri et filtrage ---
    pairs = list(zip(hits, scores))
    kept = [(h, s) for (h, s) in pairs if s >= threshold]
    kept.sort(key=lambda x: x[1], reverse=True)

    print(f"[DEBUG] threshold={threshold} | candidats={len(pairs)} | gardés={len(kept)}")

    # --- Étape 5 : formatage du résultat ---
    df_parents = pd.read_json(PAR_PATH)

    results = [] 
    parents = {}
    for i, (h, s) in enumerate(kept[:final_k]):
        payload = h.payload
        dict_chunk = {
            "rank": i + 1,
            "rerank_score": float(s),
            "doc": payload.get("Doc"),
            "page": payload.get("Page"),
            "parent_id": payload.get("ParentID"),
            "chunk": payload.get("Chunk"),
            "similarity_score": float(h.score)
        }
        results.append(dict_chunk)
        if dict_chunk["parent_id"] not in parents.keys():
            # print((df_parents[df_parents["paragraph_id"] == dict_chunk["parent_id"]].iloc[0]))
            parents[dict_chunk["parent_id"]] = (df_parents[df_parents["paragraph_id"] == dict_chunk["parent_id"]].iloc[0])["text"]
    
    
    return results,parents


# ============================================================
# 3. Interface utilisateur (CLI)
# ============================================================

if __name__ == "__main__":
    print("\nTape une requête (ou 'exit' pour quitter).")
    try:
        while True:
            query = input("\n> Requête: ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("👋 Bye!")
                break

            # Recherche + re-ranking
            results = search_and_rerank(query=query, top_k=20, final_k=7, threshold=-8.0)

            if not results:
                print("Aucun résultat trouvé.")
                continue

            print("\n Résultats les plus pertinents :\n")
            for r in results:
                print(f"[{r['rank']}] rerank={r['rerank_score']:.3f}")
                print(f"Paragraphe: {r['parent_id']}")
                print(f" Chunk: {r['chunk'][:300]}{'...' if len(r['chunk']) > 300 else ''}")
                print("-" * 90)

    except (KeyboardInterrupt, EOFError):
        print("\n👋 Bye!")