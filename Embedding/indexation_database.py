import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# ============================================================
# 0. Chemins
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT_DIR / "chunks.json"

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(f"chunks.json introuvable : {CHUNKS_PATH}")

# ============================================================
# 1. Lecture des chunks
# ============================================================

df = pd.read_json(CHUNKS_PATH)

if "text" not in df.columns:
    raise ValueError("Colonne obligatoire 'text' manquante dans chunks.json")

def get_col(candidates, label):
    found = candidates.intersection(df.columns)
    if not found:
        raise ValueError(f"Manque colonne {label} (attendu: {' / '.join(candidates)})")
    return list(found)[0]

# Récupération dynamique des colonnes
page_col = get_col({"page", "page_number"}, "pagination")
doc_col = get_col({"document_name", "filename"}, "document")
parent_col = get_col({"parent_paragraph_id", "parent_id"}, "ID parent")

print(f"{len(df)} chunks chargés depuis {CHUNKS_PATH.name}")
print(df.head(3))

# ============================================================
# 2. Préparation
# ============================================================

df["Chunk"] = df["text"].astype(str)
df["Doc"] = df[doc_col]
df["Page"] = df[page_col].astype(int)
df["Parent"] = df[parent_col]

# ============================================================
# 3. Modèle d'embedding
# ============================================================

print("Chargement du modèle BAAI/bge-m3...")
model = SentenceTransformer("BAAI/bge-m3")

# ============================================================
# 4. Connexion Qdrant
# ============================================================

client = QdrantClient("localhost", port=6333)
collection_name = "documents_chunks"

print(f"(Re)création de la collection '{collection_name}'...")
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE
    )
)

# ============================================================
# 5. Encodage + insertion
# ============================================================

points = []

for idx, row in df.iterrows():
    emb = model.encode(row["Chunk"]).tolist()

    point = models.PointStruct(
        id=idx,
        vector=emb,
        payload={
            "Chunk": row["Chunk"],
            "Doc": row["Doc"],
            "Page": row["Page"],
            "ParentID": row["Parent"]
        }
    )
    points.append(point)

    if (idx + 1) % 50 == 0 or idx == len(df) - 1:
        print(f"Encodé {idx + 1}/{len(df)} chunks")

print("Insertion dans Qdrant...")
client.upsert(collection_name=collection_name, points=points)

print("Tous les chunks ont été insérés avec succès dans Qdrant")
