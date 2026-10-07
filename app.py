import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="GlowMatch AI",
    page_icon="✨",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("products.csv")

@st.cache_resource
def build_model(data):
    # Combine product attributes into one searchable text representation.
    data = data.copy()
    data["text"] = (
        data["product_name"].fillna("") + " " +
        data["category"].fillna("") + " " +
        data["ingredients"].fillna("") + " " +
        data["description"].fillna("") + " " +
        data["skin_type"].fillna("") + " " +
        data["concern"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    matrix = vectorizer.fit_transform(data["text"])
    similarity = cosine_similarity(matrix)
    return data, vectorizer, similarity

data = load_data()
data, vectorizer, similarity = build_model(data)

# ---------------------------
# Header
# ---------------------------
st.title("✨ GlowMatch AI")
st.subheader("Personalized Skincare Product Recommendations")
st.write(
    "Select a skincare product and discover similar products using "
    "NLP-based content similarity."
)

st.info(
    "How it works: product descriptions and attributes are converted into "
    "TF-IDF vectors, and cosine similarity is used to identify the most "
    "similar products."
)

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Recommendation Settings")
top_n = st.sidebar.slider("Number of recommendations", 3, 8, 5)

categories = ["All"] + sorted(data["category"].unique().tolist())
category_filter = st.sidebar.selectbox("Category filter", categories)

if category_filter != "All":
    filtered_indices = data.index[data["category"] == category_filter].tolist()
else:
    filtered_indices = data.index.tolist()

# ---------------------------
# Product selector
# ---------------------------
product_name = st.selectbox(
    "Choose a product you are interested in:",
    data["product_name"].tolist()
)

selected_idx = data.index[data["product_name"] == product_name][0]
selected = data.loc[selected_idx]

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Selected Product")
    st.markdown(f"**{selected['product_name']}**")
    st.write(f"**Category:** {selected['category']}")
    st.write(f"**Concern:** {selected['concern']}")
    st.write(f"**Skin type:** {selected['skin_type']}")
    st.write(f"**Key ingredients:** {selected['ingredients']}")

with col2:
    st.markdown("### Product Description")
    st.write(selected["description"])

# ---------------------------
# Recommendations
# ---------------------------
if st.button("✨ Get Recommendations", use_container_width=True):
    scores = list(enumerate(similarity[selected_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in scores:
        if idx == selected_idx:
            continue

        if idx not in filtered_indices:
            continue

        recommendations.append((idx, score))

        if len(recommendations) >= top_n:
            break

    st.markdown("---")
    st.markdown("## Recommended For You")

    for rank, (idx, score) in enumerate(recommendations, 1):
        product = data.loc[idx]

        with st.container(border=True):
            c1, c2, c3 = st.columns([0.6, 4, 1.2])

            with c1:
                st.markdown(f"### {rank}")

            with c2:
                st.markdown(f"**{product['product_name']}**")
                st.caption(
                    f"{product['category']} • {product['concern']} • "
                    f"{product['skin_type']}"
                )
                st.write(product["description"])
                st.write(f"**Key ingredients:** {product['ingredients']}")

            with c3:
                st.metric("Similarity", f"{score*100:.0f}%")

    st.success(
        "Recommendations are generated from product-content similarity, "
        "not from medical or clinical suitability."
    )

# ---------------------------
# Model explanation
# ---------------------------
with st.expander("🔍 How the AI/ML model works"):
    st.markdown("""
    **1. Feature construction**  
    Product name, category, ingredients, description, skin type and concern
    are combined into a single text representation.

    **2. TF-IDF vectorization**  
    Text is converted into numerical vectors based on the importance of
    words and phrases across the product catalog.

    **3. Cosine similarity**  
    The system compares the selected product's vector with every other
    product vector.

    **4. Ranking**  
    Products with the highest similarity scores are returned as
    recommendations.
    """)

st.caption(
    "Academic/demo project. Recommendations are based on catalog similarity "
    "and should not be treated as dermatological advice."
)