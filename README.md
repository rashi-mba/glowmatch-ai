# GlowMatch AI — Product Recommendation System

An NLP/ML-based skincare product recommendation engine built using Python,
TF-IDF vectorization, cosine similarity and Streamlit.

## Business Problem

E-commerce platforms often have large product catalogs, making it difficult
for customers to discover relevant alternatives. This project demonstrates
a content-based recommendation approach that recommends products similar to
a customer's selected product.

## Solution

The system combines:
- Product name
- Category
- Ingredients
- Product description
- Target skin type
- Primary concern

These attributes are converted into TF-IDF vectors. Cosine similarity is then
used to compare products and rank the most similar recommendations.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Streamlit

## Project Structure

```text
product_recommendation_system/
├── app.py
├── products.csv
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the application

```bash
streamlit run app.py
```

The app will open in your browser.

## ML Methodology

### TF-IDF

Term Frequency-Inverse Document Frequency converts product text into
numerical features while giving more importance to informative words.

### Cosine Similarity

Cosine similarity measures the similarity between the selected product vector
and other product vectors.

The recommendation score is:

```text
cosine_similarity(selected_product, other_product)
```

Products are ranked from highest to lowest similarity.

## Example

If a user selects:

**Glow Vitamin C Serum**

the system may recommend products such as:
- Licorice Brightening Serum
- Vitamin C Day Cream
- Green Tea Antioxidant Serum
- SPF 50 Brightening Sunscreen

because their catalog attributes contain similar concepts such as
brightening, antioxidants, vitamin C, niacinamide and uneven skin tone.

## Limitations

This is a content-based prototype. It does not currently use:
- User purchase history
- Ratings/reviews
- Collaborative filtering
- Real-time inventory
- Price optimization
- Clinical/dermatological suitability

A production system could combine content-based filtering with collaborative
filtering and user behavior data to create a hybrid recommendation engine.

## Future Scope

- Add user profiles and purchase history
- Add ratings and reviews
- Build hybrid recommendations
- Add price and availability filters
- Deploy with a cloud database
- A/B test recommendation quality
- Track CTR and conversion rate

## Interview Explanation

"I built a content-based product recommendation system for a skincare
catalog. I used TF-IDF to convert product attributes and descriptions into
numerical vectors and cosine similarity to identify products with similar
content. I selected this approach because it works well for a small catalog
where user-level interaction data is limited. I then built a Streamlit
interface so users can select a product and receive ranked recommendations."

## Note

This is an academic/demo recommendation system. It is not a medical or
dermatological recommendation tool.