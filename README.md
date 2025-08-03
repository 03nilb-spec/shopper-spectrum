#  Shopper Spectrum

A retail data science app that segments customers using RFM clustering and provides product recommendations using collaborative filtering.

---

##  Live Demo

Check out the deployed app here:  
 [Shopper Spectrum Streamlit App](https://shopper-spectrum-onuy6np3nhnlgcb9oyzgf7.streamlit.app/)


## Features

### 1️. Customer Segmentation
- Based on **Recency**, **Frequency**, and **Monetary** value (RFM)
- Uses **KMeans clustering** to classify customers into:
  - **High-Value**
  - **Regular**
  - **At-Risk**

### 2️. Product Recommendation System
- **Item-based Collaborative Filtering**
- Computes **cosine similarity** between products
- Recommends 5 similar products for any input product

---

##  Tech Stack

- Python 
- Pandas, NumPy, scikit-learn
- Matplotlib, Seaborn
- Streamlit 
- Git + Git LFS (for large files)

---

##  How to Run Locally

1. **Clone the repository**
   ```
   git clone https://github.com/your-username/shopper-spectrum.git
   cd shopper-spectrum
   ```
2. **Install Dependies**
   ```
   Mentioned in the requirements
   ```

---

3. **Run the Streamlit app**
```
   streamlit run app.py
```

# Notes 
```
product_similarity.pkl is tracked via Git LFS.
```
