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

expected_cols = {"text", "document_name", "page_number", "parent_paragraph_id"}
missing = expected_cols - set(df.columns)

if missing:
    raise ValueError(f"Colonnes manquantes dans chunks.json : {missing}")

print(f"{len(df)} chunks chargés depuis {CHUNKS_PATH.name}")
print(df.head(3))

# ============================================================
# 2. Préparation
# ============================================================

df["Chunk"] = df["text"].astype(str)
df["Doc"] = df["document_name"]
df["Page"] = df["page_number"].astype(int)
df["Parent"] = df["parent_paragraph_id"]

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
