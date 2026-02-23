"""
### Évaluation du modèle d'embeddings

Pour évaluer la qualité d'un modèle d'embeddings dans une tâche de **retrieval** (recherche de documents pertinents à partir d'une requête),
on utilise plusieurs métriques standards : **MRR (Mean Reciprocal Rank)**, **Recall@k**, et **nDCG (Normalized Discounted Cumulative Gain)**.

---

#### 1. Mean Reciprocal Rank (MRR)

Le **MRR** mesure la capacité du modèle à placer le premier document pertinent le plus haut possible dans le classement.

- Pour chaque requête, on calcule le **rang** (*rank*) du premier document pertinent dans la liste triée des résultats.
- On prend ensuite son **inverse** (1/rank).
- La moyenne de ces valeurs sur toutes les requêtes donne le **MRR**.

MRR = (1/N)*somme(1/rank_i) tel que i de 1 jusqu'à N

> **Interprétation :**
> - Un MRR proche de **1.0** signifie que les documents pertinents apparaissent presque toujours tout en haut du classement.
> - Plus le MRR est élevé, meilleure est la précision du modèle sur les premiers résultats.

---

#### 2. Recall@k

Le **Recall@k** mesure la proportion de documents pertinents retrouvés **parmi les k premiers résultats**.

Recall@k = (documents pertinents retrouvés dans le top-k) / (documents pertinents totaux)

> **Interprétation :**
> - Si `Recall@10 = 0.8`, cela signifie que 80 % des documents pertinents se trouvent parmi les 10 premiers résultats.
> - Cette métrique évalue la **capacité de couverture** du modèle (retrouver tous les bons documents).

---

#### 3. Normalized Discounted Cumulative Gain (nDCG)

Le **nDCG** mesure la qualité **du classement** des documents pertinents.  
Il prend en compte la position des documents pertinents et attribue une **pondération logarithmique** : les documents retrouvés plus haut valent plus.


nDCG@k = DCG@k/IDCG@k

avec  

DCG@k = somme( rel_i/log_2(i+1) ) tel que i de 1 jusqu'à k

et rel_i est la pertinence du document à la position *i*.

> **Interprétation :**
> - nDCG varie entre **0 et 1**.
> - Plus il est proche de 1, plus les documents pertinents sont bien classés dans les premières positions.
> - Cette métrique reflète la **qualité du tri des résultats**.

---

**En pratique :**
- On utilise **MRR** pour évaluer la **précision des premiers résultats**,  
- **Recall@k** pour mesurer la **capacité à tout retrouver**,  
- et **nDCG** pour juger la **qualité du classement global**.

---

### Benchmark entre BGE-M3 et Multilingual-E5-Large

Deux modèles d'embeddings ont été évalués sur un corpus technique lié au traitement du signal.  
Les métriques précédentes ont été calculées sur un jeu de 40 requêtes afin de comparer leurs performances globales.

#### Résultats :

**Model BGE-M3 :**
- MRR          : 0.8481  
- Recall@1     : 0.2500  
- Recall@3     : 0.5083  
- Recall@5     : 0.6083  
- Recall@10    : 0.8167  
- nDCG@10      : 0.7172  

**Model Multilingual-E5-Large :**
- MRR          : 0.8205  
- Recall@1     : 0.2333  
- Recall@3     : 0.5000  
- Recall@5     : 0.6333  
- Recall@10    : 0.7750  
- nDCG@10      : 0.6815  

---

### Analyse et interprétation

→ Le modèle **BGE-M3** présente de meilleures performances globales :

- **MRR plus élevé**, indiquant qu'il retrouve les documents pertinents plus haut dans le classement.  
- **nDCG@10 supérieur**, reflétant un meilleur ordonnancement des documents.  
- **Recall@10 légèrement meilleur**, traduisant une meilleure couverture des documents pertinents.  

En résumé :
- **BGE-M3** est plus précis, mieux classant et plus robuste que **E5** sur ce jeu de données.  
- Son architecture hybride (dense + lexical + multi-vector) lui confère un avantage pour la recherche sémantique multilingue.

---

### Conclusion

Le modèle **BGE-M3** est le plus performant pour cette tâche de recherche sémantique,
et il est donc recommandé pour l'intégration dans le pipeline **RAG (Retrieval-Augmented Generation)** du projet.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json

bge_model = SentenceTransformer("BAAI/bge-m3")
multilingual_model = SentenceTransformer("intfloat/multilingual-e5-large")

def charger_json_simple(chemin_fichier):
    """Charge simplement un fichier JSON"""
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

df_test = charger_json_simple("./test_data.json")
queries = df_test['queries']
corpus = df_test['corpus']
ground_truth = df_test['ground_truth']


def get_embeddings(model, texts):
    """Retourne les embeddings normalisés (numpy array)"""
    emb = model.encode(texts)
    return np.array(emb)

def evaluate_model(model_name, model, queries, corpus, ground_truth, k_values=[1, 3, 5, 10]):
    print(f"\n Évaluation du modèle : {model_name}")
    
    # Embeddings
    query_texts = [q["text"] for q in queries]
    doc_texts   = [d["text"] for d in corpus]
    doc_ids     = [d["id"] for d in corpus]

    query_emb = get_embeddings(model, query_texts)
    doc_emb   = get_embeddings(model, doc_texts)

    # Similarités (matrice)
    sims = cosine_similarity(query_emb, doc_emb)

    # Résultats pour toutes les queries
    mrr_scores, recalls, ndcgs = [], {k: [] for k in k_values}, {k: [] for k in k_values}

    for i, q in enumerate(tqdm(queries)):
        qid = q["id"]
        relevant_docs = set(ground_truth.get(qid, []))
        if not relevant_docs:
            continue

        # Classement décroissant
        sorted_idx = np.argsort(-sims[i])
        ranked_doc_ids = [doc_ids[j] for j in sorted_idx]

        # Positions des docs pertinents
        ranks = [ranked_doc_ids.index(doc_id) + 1 for doc_id in relevant_docs if doc_id in ranked_doc_ids]
        if not ranks:
            continue

        # MRR
        mrr_scores.append(1 / min(ranks))

        # Recall@k et nDCG@k
        for k in k_values:
            top_k = ranked_doc_ids[:k]
            hits = [1 if d in relevant_docs else 0 for d in top_k]
            recall_k = sum(hits) / len(relevant_docs)
            recalls[k].append(recall_k)

            # nDCG@k
            dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
            ideal_hits = [1] * min(len(relevant_docs), k)
            idcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(ideal_hits))
            ndcg_k = dcg / idcg if idcg > 0 else 0
            ndcgs[k].append(ndcg_k)

    # Moyennes
    print(f"MRR : {np.mean(mrr_scores):.4f}")
    for k in k_values:
        print(f"Recall@{k} : {np.mean(recalls[k]):.4f}")
        print(f"nDCG@{k}   : {np.mean(ndcgs[k]):.4f}")

    return {
        "MRR": np.mean(mrr_scores),
        "Recall": {k: np.mean(recalls[k]) for k in k_values},
        "nDCG": {k: np.mean(ndcgs[k]) for k in k_values},
    }

results_1 = evaluate_model("model bge", bge_model, queries, corpus, ground_truth)
results_2 = evaluate_model("model multilingual", multilingual_model, queries, corpus, ground_truth)

if __name__ == "__main__":
    print(results_1, results_2)