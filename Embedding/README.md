Ce projet a pour objectif de tester et de vérifier le pipeline de création d'embeddings et d'évaluation de modèles pour le traitement du signal.  
**Attention** : la version actuelle est une version de test. Le modèle entraîné et les datasets sont **uniquement pour vérifier que le code fonctionne**, et ne sont pas destinés à un vrai entraînement sur le domaine du signal.

---

## Structure du projet

| Fichier / Dossier | Rôle |
|------------------|------|
| `embedding.ipynb` | Notebook principal : création des embeddings, insertion dans Qdrant, recherche vectorielle et calcul des métriques (MRR, Recall@k, nDCG). Contient également le fine-tuning léger pour tester le pipeline. |
| `checkpoints/` | Dossier contenant les modèles sauvegardés lors du fine-tuning. |
| `fine_tuned_all_MiniLM_L6_v2/` | Modèle All-MiniLM-L6-v2 entraîné sur un petit dataset de test. Sert uniquement à vérifier le pipeline, pas à produire un modèle final performant. |
| `training_data.csv` | Dataset utilisé pour le fine-tuning léger. Contient des exemples limités et génériques. |
| `original_data.csv` | Dataset original servant à créer des embeddings de base et comparer les performances avant fine-tuning. |
| `test_data.json` | Dataset de test pour évaluer les modèles (contient queries, corpus et ground truth). |
| `docker-compose.yml` | Fichier de configuration Docker pour lancer le service Qdrant et les dépendances nécessaires. |

---

## Notes importantes

- Les **datasets actuels sont uniquement des exemples de vérification**, ils ne sont pas adaptés à un véritable entraînement sur le domaine du traitement du signal.
- Le **modèle fine-tuné** (`all-MiniLM-L6-v2`) est beaucoup moins performant que le modèle bge-m3 ou multilingual-e5-large. Il sert juste à valider que le pipeline fonctionne correctement.
- Ce projet permet de vérifier que :
  - La création d'embeddings fonctionne
  - La recherche vectorielle dans Qdrant fonctionne
  - Le calcul des métriques (MRR, Recall@k, nDCG) est correct
- L’étape suivante consistera à **construire un dataset complet et adapté** au domaine du traitement du signal pour entraîner réellement un modèle performant.