

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
model = SentenceTransformer("intfloat/multilingual-e5-large")


from sentence_transformers import CrossEncoder
#reranker = CrossEncoder("BAAI/bge-reranker-large")
reranker = CrossEncoder("BAAI/bge-reranker-large", local_files_only=True)


### pour activer la base vectorielle   docker run -p 6333:6333 qdrant/qdrant

## pour avoir de la persistance :
## docker run -d --name qdrant_local -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

## apres redémarrage 
## docker start qdrant_local

## vérifier la liste des conteneurs actifs
# docker ps

# stopper Qdrant
# docker stop qdrant_local
  