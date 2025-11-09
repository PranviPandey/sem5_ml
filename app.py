import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Segmentation & Sales Prediction", layout="wide")

st.title("🛍️ Customer Segmentation & Sales Prediction Dashboard")

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file (e.g., supermarket_sales.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(df.describe())

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.header("🔧 Data Preprocessing")

    data = df.copy()
    if 'Invoice ID' in data.columns:
        data.drop(['Invoice ID'], axis=1, inplace=True)

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
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
    st.success("✅ Data Preprocessing Completed!")

    # -------------------------------
    # Customer Segmentation
    # -------------------------------
    st.header("👥 Customer Segmentation")

    X = data.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca)

    gmm = GaussianMixture(n_components=7, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca)

    kmeans_score = silhouette_score(X_pca, kmeans_labels)
    gmm_score = silhouette_score(X_pca, gmm_labels)

    st.write("### Silhouette Score Comparison")
    st.write(f"K-Means: {kmeans_score:.3f} | GMM: {gmm_score:.3f}")

    better_cluster = "K-Means" if kmeans_score > gmm_score else "GMM"
    st.success(f"🏆 Better Clustering Model: **{better_cluster}** (Higher Silhouette Score)")

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='viridis', s=40)
    ax[0].set_title("K-Means Clusters (PCA projection)")
    ax[1].scatter(X_pca[:,0], X_pca[:,1], c=gmm_labels, cmap='plasma', s=40)
    ax[1].set_title("GMM Clusters (PCA projection)")
    st.pyplot(fig)

    # -------------------------------
    # Cluster Summary
    # -------------------------------
    st.header("📈 Cluster Summary")

    data['Cluster'] = kmeans.labels_
    cluster_summary = data.groupby('Cluster').mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(cluster_summary, cmap='coolwarm', annot=True, ax=ax)
    st.pyplot(fig)

    st.dataframe(cluster_summary)

    # -------------------------------
    # Interpretation
    # -------------------------------
    st.header("📊 Cluster Insights")

    fig, axes = plt.subplots(1,3, figsize=(15,4))
    sns.barplot(x='Cluster', y='Sales', data=data, palette='coolwarm', ax=axes[0])
    axes[0].set_title("Average Sales per Cluster")
    sns.barplot(x='Cluster', y='Rating', data=data, palette='viridis', ax=axes[1])
    axes[1].set_title("Average Rating per Cluster")
    sns.scatterplot(x='Quantity', y='Unit price', hue='Cluster', data=data, palette='tab10', alpha=0.7, ax=axes[2])
    axes[2].set_title("Quantity vs Unit Price (by Cluster)")
    st.pyplot(fig)

    # -------------------------------
    # Sales Prediction
    # -------------------------------
    st.header("💰 Sales Prediction Models")

    X = data[['Unit price', 'Quantity', 'Rating', 'Cluster']]
    y = data['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Model Performance Metrics
    results = {
        "Model": ["Linear Regression", "Random Forest"],
        "R2 Score": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf)],
        "MAE": [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf)],
        "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_rf))]
    }
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Determine better model
    best_model = results_df.loc[results_df['R2 Score'].idxmax(), 'Model']
    st.success(f"🏆 The **better performing model** is: **{best_model}** (higher R² score)")

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].scatter(y_test, y_pred_lr, color='blue', alpha=0.6)
    ax[0].set_title("Linear Regression: Actual vs Predicted Sales")
    ax[1].scatter(y_test, y_pred_rf, color='green', alpha=0.6)
    ax[1].set_title("Random Forest: Actual vs Predicted Sales")
    st.pyplot(fig)

    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
    st.subheader("🔹 Feature Importance (Random Forest)")
    st.dataframe(feature_importance)

    st.write(f"Train R²: {rf.score(X_train, y_train):.3f}")
    st.write(f"Test R²: {rf.score(X_test, y_test):.3f}")

else:
    st.info("👆 Please upload your dataset to get started.")
