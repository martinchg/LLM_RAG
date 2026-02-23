import json
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric
)
from deepeval.evaluate import AsyncConfig
from deepeval.models import OllamaModel
import sys
import pandas as pd

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "36000" 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import query_llm

"""📊 Définition des Métriques d'Évaluation RAG
🎯 Métriques utilisées
Ce benchmark évalue un système RAG avec 4 métriques DeepEval :

1️⃣ Faithfulness (Fidélité) - Score: 0 à 1
Mesure : La réponse générée est-elle fidèle au contexte récupéré, sans hallucinations?

Extrait les vérités du contexte et les affirmations de la réponse
Vérifie que chaque affirmation peut être justifiée par le contexte
Détecte les informations inventées ou inexactes
Interprétation :

0.8+ : Excellente fidélité
0.5-0.8 : Bonne fidélité avec quelques imprécisions
<0.5 : Hallucinations détectées

2️⃣ Contextual Precision (Précision Contextuelle) - Score: 0 à 1
Mesure : Quel pourcentage des documents récupérés sont vraiment pertinents?

Évalue si chaque document retourné contient des informations utiles
Calcule: documents pertinents / documents totaux récupérés
Ignore l'ordre de classement
Interprétation :

1.0 : Tous les documents sont pertinents
0.7-0.9 : Bonne précision, peu de bruit
0.5-0.7 : Acceptable, du bruit présent
<0.5 : Mauvaise précision

3️⃣ Contextual Recall (Rappel Contextuel) - Score: 0 à 1
Mesure : TOUTES les informations nécessaires sont-elles présentes dans les documents récupérés?

Extrait les phrases clés de la réponse attendue
Vérifie si chaque phrase clé existe dans les documents récupérés
Calcule: phrases trouvées / phrases totales
Interprétation :

1.0 : Toutes les infos nécessaires sont présentes
0.7-0.9 : La plupart des infos sont présentes
0.5-0.7 : Environ la moitié des infos manquent
<0.5 : Trop d'infos manquantes
4️⃣ Answer Relevancy (Pertinence de la Réponse) - Score: 0 à 1
Mesure : La réponse générée répond-elle vraiment à la question posée?

Analyse la cohérence sémantique entre la question et la réponse
Détecte les informations hors-sujet
Calcule: infos pertinentes / infos totales
Interprétation :

0.8+ : Très pertinent
0.6-0.8 : Pertinent avec quelques hors-sujets
0.4-0.6 : Partiellement pertinent
<0.4 : Peu pertinent
📈 Score Global
Score moyen = (Faithfulness + Precision + Recall + Relevancy) / 4

Excellent:  0.8-1.0 ✅✅✅
Bon:        0.6-0.8 ✅✅
Acceptable: 0.4-0.6 ✅
Faible:     <0.4    ❌
Pass rate = Pourcentage de métriques qui dépassent le seuil de 0.5

"""

TOP_K = 5        # nombre de documents à récupérer

# Chargement des modèles
retriever_model = SentenceTransformer("BAAI/bge-m3")

# -------------------------------------------------
# 🔹 CHARGEMENT DES DONNÉES
# -------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

dataset = load_json("./test_data.json")  # contient queries, corpus, ground_truth
queries = dataset["queries"]
corpus = dataset["corpus"]
ground_truth = dataset["ground_truth"]

corpus_ids = [d["id"] for d in corpus]
corpus_texts = [d["text"] for d in corpus]

# 🔹 ÉTAPE 2 — GÉNÉRATION DE LA RÉPONSE
# -------------------------------------------------

from search_and_rerank import search_and_rerank

def generate_answer(question):
    query = question
    results, parents = search_and_rerank(query)

    
    print(f"\n🔍 Résultats pour : {question}\n")

    response_final = ""
    chunks_list = []

    for r in results:
        phrase = f"{r['rank']}.  {r['doc']} (Page {r['page']})\n"
        phrase += f"    {r['chunk'][:400]}...\n"
        phrase += f"    Pertinence : {r['rerank_score']:.4f} | Score Qdrant : {r['similarity_score']:.4f}\n"
        phrase += "-" * 100 + "\n"

        response_final += phrase
        chunks_list.append(r["chunk"])
    prompt = f"""
Voici la question à laquelle on souhaite répondre :
{query}

•⁠  ⁠Voici les chunks possibles avec les identifiants des parents qui leur sont associés :
{response_final}

•⁠  ⁠Voici les parents avec leurs identifiants :
{parents}

Consigne (procédure obligatoire) :
1.⁠ ⁠Sélectionne que le 1 chunk parmi ceux ayant les reranker_scores les plus élevés.
2.⁠ ⁠Récupère le parent associé à ce chunk sélectionné.
3.⁠ ⁠Analyse uniquement ce parent et identifie celui qui contient l'information permettant de répondre.
4.⁠ ⁠Répond à la question uniquement à partir du contenu de ce parent.

Contraintes strictes :
•⁠  ⁠Réponds uniquement à partir des extraits fournis, sans rien inventer, extrapoler ou déduire d’informations externes.
•⁠  ⁠Si l’information exacte n’apparaît pas dans le parent sélectionné, dis-le clairement.
•⁠  ⁠Reformule au plus près des formulations du texte, sans ajouter de détails absents.
•⁠  ⁠Ne mentionne jamais dans la réponse les mots “chunk”, “parent”, “score”, “reranker_score”, ni la méthode de sélection.
"""





    reponse_final = query_llm(prompt)
    return reponse_final.strip(), chunks_list

# -------------------------------------------------
# 🔹 ÉTAPE 3 — ÉVALUATION GÉNÉRATION (DeepEval + Ollama)
# -------------------------------------------------

# Configuration Ollama
eval_llm = OllamaModel(model="mistral-nemo:12b")

def evaluate_generation(queries, ground_truth):
    print("\n🔹 Évaluation avec DeepEval + Ollama...")
    print(f"📌 Total de questions à évaluer: {len(queries)}\n")

    test_cases = []
    
    # 🔹 Évalue TOUTES les questions
    for idx, q in enumerate(queries[:]):
        print(f"⏳ Traitement question {idx + 1}/{len(queries)}: {q['text'][:150]}...")
        
        qid = q["id"]
        relevant_docs = [c for c in corpus if c["id"] in ground_truth.get(qid, [])]

        # Génération
        response, chunks_list = generate_answer(q["text"])  
        expected_answer = "\n".join([d["text"] for d in relevant_docs])

        # Crée LLMTestCase
        test_case = LLMTestCase(
            input=q["text"],
            actual_output=response,
            expected_output=expected_answer,
            retrieval_context=chunks_list  
        )
        test_cases.append(test_case)

    # Utilise Ollama pour les métriques
    metrics = [
        FaithfulnessMetric(model=eval_llm),
        ContextualPrecisionMetric(model=eval_llm),
        ContextualRecallMetric(model=eval_llm),
        AnswerRelevancyMetric(model=eval_llm),
    ]

    print(f"\n🔍 Évaluation des {len(test_cases)} test cases...\n")
    results = evaluate(
        test_cases, 
        metrics=metrics,
        async_config=AsyncConfig(run_async=False)
    )
    
    # ✅ AJOUT: Extraction et calcul des statistiques
    print("\n" + "="*80)
    print("EXTRACTION DES RESULTATS")
    print("="*80 + "\n")
    
    # Charge le fichier généré par DeepEval
    deepeval_file = ".deepeval/.latest_test_run.json"
    
    if os.path.exists(deepeval_file):
        with open(deepeval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extraction des métriques globales
        metrics_summary = data["testRunData"]["metricsScores"]
        
        results_dict = {
            "total_questions": len(test_cases),
            "evaluation_time_seconds": data["testRunData"]["runDuration"],
            "timestamp": pd.Timestamp.now().isoformat(),
            "metrics": {}
        }
        
        # Calcul pour chaque métrique
        for metric_data in metrics_summary:
            metric_name = metric_data["metric"]
            scores = metric_data["scores"]
            passed = metric_data["passes"]
            total = passed + metric_data["fails"]
            
            avg_score = sum(scores) / len(scores) if scores else 0
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            results_dict["metrics"][metric_name] = {
                "pass_rate_percent": round(pass_rate, 2),
                "passed": passed,
                "total": total,
                "average_score": round(avg_score, 4),
                "scores": scores
            }
            
            # Affichage avec précision exacte
            print(f"{metric_name}: {pass_rate:.2f}% pass rate")
        
        # Calcul du score global (moyenne de toutes les métriques)
        all_avg = [results_dict["metrics"][m]["average_score"] for m in results_dict["metrics"]]
        global_score = sum(all_avg) / len(all_avg) if all_avg else 0
        results_dict["global_average_score"] = round(global_score, 4)
        
        print("\n" + "="*80)
        print(f"Score Global Moyen: {global_score:.4f}")
        print("="*80 + "\n")
        
        # Sauvegarde dans un fichier JSON dédié
        output_path = "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Résultats sauvegardés dans: {output_path}")
        print(f"✅ Résultats DeepEval complets dans: {deepeval_file}\n")
        
        return results_dict
    else:
        print(f"❌ ERREUR: Fichier {deepeval_file} introuvable!")
        return None

# -------------------------------------------------
# 🚀 MAIN
# -------------------------------------------------
if __name__ == "__main__":
    evaluate_generation(queries, ground_truth)