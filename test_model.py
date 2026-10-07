import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("products.csv")

df["text"] = (
    df["product_name"] + " " +
    df["category"] + " " +
    df["ingredients"] + " " +
    df["description"] + " " +
    df["skin_type"] + " " +
    df["concern"]
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
matrix = vectorizer.fit_transform(df["text"])
sim = cosine_similarity(matrix)

query = "Glow Vitamin C Serum"
idx = df.index[df["product_name"] == query][0]

results = sorted(
    [(i, sim[idx][i]) for i in range(len(df)) if i != idx],
    key=lambda x: x[1],
    reverse=True
)[:5]

print(f"Recommendations for: {query}\n")
for i, score in results:
    print(f"{df.loc[i, 'product_name']}: {score:.3f}")