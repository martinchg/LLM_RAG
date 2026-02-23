from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

## modele de reranking 
model_name = "BAAI/bge-reranker-v2-m3"

## chargement du modele de Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

#Device (GPU si dispo, sinon CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

query =" Ou se trouve la Tour effiel "
documents =[
    "La tour Eiffel est à Paris.",
    "Le Colisée se trouve à Rome.",
    "La Statue de la Liberté est à New York.",
    "La tour Montparnasse est un gratte-ciel parisien.",
    "anh tam est à la tour effiel ",
    "le concorde est au Bourget"]

# Prépare les paires (requête, document)
paires= [(query, doc) for doc in documents]

# Tokenization

inputs = tokenizer(
    [p[0] for p in paires], # query
    [p[1] for p in paires],# docs
     padding =True,
     truncation = True,
     return_tensors = "pt").to(device)

# inference
with torch.no_grad():
    scores = model(**inputs).logits.squeeze(dim=1)

# tri par score décroissant
ranked = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)

# résultats
print("\nClassement par pertinence :\n")
for i, (doc, score) in enumerate(ranked, 1):
    print(f"{i}. {doc}  (score={score:.4f})")


##fixer un seuil de score  : si le meilleur est en dessous du seuil on répond qu'on n'a pas de réponse 