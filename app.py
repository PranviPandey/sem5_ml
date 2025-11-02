import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Retail Customer Segmentation & Sales Prediction", layout="wide")
st.title("🛍️ Customer Segmentation & Sales Prediction for Retail Store")
st.write("Upload your dataset to analyze customer behavior, identify segments, and predict future sales.")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file (e.g., supermarket_sales.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    st.write("**Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    # -------------------- DATA PREPROCESSING --------------------
    st.subheader("🧹 Data Preprocessing")

    data = df.copy()
    if 'Invoice ID' in data.columns:
        data.drop(['Invoice ID'], axis=1, inplace=True)

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['Weekday'] = data['Date'].dt.day_name()

    label_cols = [c for c in ['Branch','City','Customer type','Gender','Product line','Payment','Weekday'] if c in data.columns]
    encoders = {}

    for col in label_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    data = data.dropna().reset_index(drop=True)
    st.success("✅ Preprocessing complete. Dataset is ready for clustering and prediction.")
    st.dataframe(data.head())

    # -------------------- CLUSTERING --------------------
    st.subheader("🎯 Customer Segmentation (K-Means + PCA)")

    X = data.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    n_clusters = st.slider("Select number of clusters (k)", 3, 10, 7)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca)
    silhouette = silhouette_score(X_pca, kmeans_labels)

    st.write(f"**Silhouette Score:** {silhouette:.3f}")

    # Plot clusters
    fig, ax = plt.subplots(figsize=(7,5))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='viridis', s=40)
    plt.title('K-Means Clusters (PCA projection)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    st.pyplot(fig)

    data['Cluster'] = kmeans.labels_
    cluster_summary = data.groupby('Cluster').mean(numeric_only=True)

    st.write("### Cluster Summary (Average Feature Values)")
    st.dataframe(cluster_summary)

    # Heatmap
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(cluster_summary, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

    # -------------------- VISUAL INSIGHTS --------------------
    st.subheader("📊 Cluster Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_sales = data.groupby('Cluster')['Sales'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='Cluster', y='Sales', data=avg_sales, palette='coolwarm', ax=ax)
        ax.set_title('Average Sales per Cluster')
        st.pyplot(fig)

    with col2:
        avg_rating = data.groupby('Cluster')['Rating'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='Cluster', y='Rating', data=avg_rating, palette='viridis', ax=ax)
        ax.set_title('Average Rating per Cluster')
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots()
        sns.scatterplot(x='Quantity', y='Unit price', hue='Cluster', data=data, palette='tab10', alpha=0.7, ax=ax)
        ax.set_title('Quantity vs Unit Price (by Cluster)')
        st.pyplot(fig)

    # -------------------- SALES PREDICTION --------------------
    st.subheader("💰 Sales Prediction")

    features = ['Unit price', 'Quantity', 'Rating', 'Cluster']
    if all(f in data.columns for f in features):
        X = data[features]
        y = data['Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_choice = st.radio("Choose a regression model:", ["Multiple Linear Regression", "Random Forest Regressor"])

        if model_choice == "Multiple Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.write("### Model Performance")
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("MAE", f"{mae:.3f}")
        st.metric("RMSE", f"{rmse:.3f}")

        # Plot actual vs predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6, color='green')
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title(f"{model_choice}: Actual vs Predicted Sales")
        st.pyplot(fig)

        # Feature importance (for RF)
        if model_choice == "Random Forest Regressor":
            importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            importance = importance.sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=importance, palette='Blues_r', ax=ax)
            ax.set_title('Feature Importance (Random Forest)')
            st.pyplot(fig)

    else:
        st.warning("Required columns missing for Sales Prediction (need Unit price, Quantity, Rating, Cluster).")

    # -------------------- DOWNLOAD PROCESSED DATA --------------------
    st.subheader("📥 Download Processed Data with Clusters")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "segmented_data.csv", "text/csv")
else:
    st.info("👆 Upload a CSV file to begin analysis.")
