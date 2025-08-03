import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load saved models and data
with open('rfm.pkl', 'rb') as f:
    rfm = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('product_similarity.pkl', 'rb') as f:
    product_similarity_df = pickle.load(f)

# Optional: Load product mapping (StockCode to Description)
try:
    product_map = pd.read_csv("product_map.csv")  # Should have 'StockCode' and 'Description'
    product_dict = product_map.set_index('Description')['StockCode'].to_dict()
    reverse_product_dict = product_map.set_index('StockCode')['Description'].to_dict()
except:
    product_dict = {}
    reverse_product_dict = {}

# Streamlit UI Setup
st.set_page_config(page_title="Retail Customer Analysis", layout="wide")
st.title("🛒 Customer Segmentation & Product Recommendation App")

# Sidebar navigation
page = st.sidebar.radio("Go to", ["📊 Customer Segmentation", "🎁 Product Recommendation"])

# -----------------------------------------------
# 📊 1️⃣ Customer Segmentation
# -----------------------------------------------
if page == "📊 Customer Segmentation":
    st.header("📊 Customer Segmentation")

    recency = st.number_input("Recency (days ago)", min_value=0, max_value=1000, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, max_value=1000, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=100.0)

    if st.button("Predict Customer Segment"):
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        
        # Normalize using same scale as original training
        scaler = StandardScaler()
        scaler.fit(rfm[['Recency', 'Frequency', 'Monetary']])
        input_scaled = scaler.transform(input_data)

        cluster = kmeans.predict(input_scaled)[0]

        label_map = {
            0: "At-Risk",
            1: "High-Value",
            2: "Regular"
        }
        label = label_map.get(cluster, f"Cluster {cluster}")
        st.success(f"🧠 Predicted Segment: **{label}**")

# -----------------------------------------------
# 🎁 2️⃣ Product Recommendation
# -----------------------------------------------
elif page == "🎁 Product Recommendation":
    st.header("🎁 Product Recommendation System")

    # Dropdown instead of text input for safer selection
    product_name = st.selectbox(
        "Select a Product Name:",
        sorted(product_dict.keys())
    )

    if st.button("Get Recommendations"):
        stock_code = product_dict.get(product_name)

        if stock_code and stock_code in product_similarity_df.columns:
            similarity_scores = product_similarity_df[stock_code].sort_values(ascending=False)[1:6]

            st.subheader("🔁 Top 5 Similar Products:")
            for i, code in enumerate(similarity_scores.index, 1):
                name = reverse_product_dict.get(code, f"StockCode: {code}")
                st.write(f"**{i}. {name}**")
        else:
            st.error("❌ Similarity data not found for this product.")
