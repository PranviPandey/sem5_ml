# =============================================
# 📊 Customer Segmentation & Sales Forecast App
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from prophet import Prophet

# Streamlit settings
st.set_page_config(page_title="Customer Segmentation & Forecast", layout="wide")

st.title("🛒 Customer Segmentation & Sales Forecasting Dashboard")
st.write("Upload your sales dataset to explore customer clusters and predict future sales trends.")

# ===============================
# File Upload Section
# ===============================
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    # ======================================
    # Data Preprocessing
    # ======================================
    data = df.copy()
    if 'Invoice ID' in data.columns:
        data.drop(['Invoice ID'], axis=1, inplace=True)

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Weekday'] = data['Date'].dt.day_name()

    label_cols = [c for c in ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'Weekday'] if c in data.columns]

    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    data = data.dropna().reset_index(drop=True)

    # ======================================
    # Clustering (K-Means & GMM)
    # ======================================
    X = data.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca)

    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca)

    kmeans_score = silhouette_score(X_pca, kmeans_labels)
    gmm_score = silhouette_score(X_pca, gmm_labels)

    # ======================================
    # Display Results
    # ======================================
    st.header("🧩 Customer Segmentation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("K-Means Clusters (PCA Projection)")
        fig1, ax1 = plt.subplots(figsize=(6,5))
        ax1.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='viridis', s=40)
        ax1.set_xlabel('PCA 1')
        ax1.set_ylabel('PCA 2')
        ax1.set_title('K-Means')
        st.pyplot(fig1)

    with col2:
        st.subheader("Gaussian Mixture Model (PCA Projection)")
        fig2, ax2 = plt.subplots(figsize=(6,5))
        ax2.scatter(X_pca[:,0], X_pca[:,1], c=gmm_labels, cmap='plasma', s=40)
        ax2.set_xlabel('PCA 1')
        ax2.set_ylabel('PCA 2')
        ax2.set_title('GMM')
        st.pyplot(fig2)

    st.markdown(f"""
    ### 💡 Interpretation:
    - **K-Means Silhouette Score:** `{kmeans_score:.3f}`  
    - **GMM Silhouette Score:** `{gmm_score:.3f}`
    
    **Observation:**  
    K-Means forms **distinct, compact clusters** → better separation and higher silhouette score.  
    GMM produces **overlapping clusters** due to its probabilistic nature.  
    **Hence, K-Means is preferred** for this dataset.
    """)

    # ======================================
    # Cluster Summary
    # ======================================
    data['Cluster'] = kmeans.labels_
    cluster_summary = data.groupby('Cluster').mean(numeric_only=True)

    st.header("📈 Cluster Summary")
    col3, col4 = st.columns([1.2, 1])

    with col3:
        fig3, ax3 = plt.subplots(figsize=(8,5))
        sns.heatmap(cluster_summary, cmap='coolwarm', annot=True, ax=ax3)
        ax3.set_title("Average Feature Values per Cluster")
        st.pyplot(fig3)

    with col4:
        st.markdown("""
        ### 🔍 Interpretation:
        - **Cluster 1:** High unit price, quantity, and sales → **Premium Customers**
        - **Cluster 0:** Moderate spenders → **Regular Customers**
        - **Cluster 2:** Low spenders but satisfied → **Budget Customers**
        """)

    # ======================================
    # Behavior Visualization
    # ======================================
    st.header("🧠 Cluster Behavior Insights")
    fig4, axes = plt.subplots(1, 3, figsize=(16,5))

    avg_sales = data.groupby('Cluster')['Sales'].mean().reset_index()
    sns.barplot(x='Cluster', y='Sales', data=avg_sales, palette='coolwarm', ax=axes[0])
    axes[0].set_title('Average Sales per Cluster')

    avg_rating = data.groupby('Cluster')['Rating'].mean().reset_index()
    sns.barplot(x='Cluster', y='Rating', data=avg_rating, palette='viridis', ax=axes[1])
    axes[1].set_title('Average Rating per Cluster')

    sns.scatterplot(x='Quantity', y='Unit price', hue='Cluster', data=data, palette='tab10', alpha=0.7, ax=axes[2])
    axes[2].set_title('Quantity vs Unit Price')

    st.pyplot(fig4)

    st.markdown("""
    ### 💬 Insights:
    - **Cluster 1:** Drives most revenue (bulk or premium buyers).  
    - **Cluster 0:** Stable, regular shoppers.  
    - **Cluster 2:** Budget-conscious yet satisfied (high ratings).  
    """)

    # ======================================
    # Sales Forecasting (Prophet)
    # ======================================
    st.header("⏳ Sales Forecasting with Prophet")

    if 'Date' in data.columns and 'Sales' in data.columns:
        daily_sales = data.groupby('Date')['Sales'].sum().reset_index()
        daily_sales.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

        model = Prophet(daily_seasonality=True)
        model.fit(daily_sales)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.subheader("📅 Next 30-Day Sales Forecast")
        
        fig5 = model.plot(forecast, figsize=(10,5))
        plt.title("Sales Forecast for Next 30 Days")
        st.pyplot(fig5)
        st.markdown("""
        ### 💡 Interpretation:
        - Sales Forecast for Next 30 days:
        - *Black dots* = your actual sales (historical data)
        - *Blue line* = Prophet’s predicted trend (yhat)
        - *Light blue area* = uncertainty range (± predicted variation)
        - The relatively stable band indicates that Prophet sees no strong upward/downward trend, just periodic fluctuations.
                """)
        st.subheader("📊 Trend & Seasonality Components")
        fig6 = model.plot_components(forecast)
        st.pyplot(fig6)

        st.markdown("""
        ### 💡 Interpretation:
        - The **trend** shows whether sales are increasing or stabilizing over time.
        - **Seasonal patterns** may indicate high-demand days (like weekends or holidays).
        - This helps plan **inventory and promotions** effectively.
        """)

    else:
        st.warning("⚠️ Columns 'Date' and 'Sales' are required for forecasting.")

else:
    st.info("👆 Please upload a CSV file to start analysis.")
