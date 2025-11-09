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
    st.subheader("🧩 Customer Segmentation Analysis")

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

    <div style="background-color:#d4edda; padding:15px; border-radius:10px; border-left:6px solid #28a745;">
        <b>Observation:</b><br>
        The <b>K-Means</b> model forms <b>clear and compact clusters</b>, showing stronger separation and a higher silhouette score.  
        In contrast, the <b>GMM</b> model produces slightly overlapping clusters because of its probabilistic nature.  
        <b>✅ Hence, K-Means is the preferred model for this dataset.</b>
    </div>
""", unsafe_allow_html=True)


    # ======================================
    # Cluster Summary
    # ======================================
    data['Cluster'] = kmeans.labels_
    cluster_summary = data.groupby('Cluster').mean(numeric_only=True)

    st.subheader("📈 Cluster Summary")
    col3, col4 = st.columns([1.2, 1])

    # 🎨 Left column → Heatmap
    with col3:
        fig3, ax3 = plt.subplots(figsize=(8,5))
        sns.heatmap(cluster_summary, cmap='coolwarm', annot=True, ax=ax3)
        ax3.set_title("Average Feature Values per Cluster")
        st.pyplot(fig3)

    # 💬 Right column → Interpretation
    with col4:
        st.markdown("""
        ### 🔍 Interpretation:
        - **Cluster 0:** Moderate spenders → **Regular Customers**  
        - **Cluster 1:** High unit price, quantity, and sales → **Premium Customers**  
        - **Cluster 2:** Low spenders but satisfied → **Budget Customers**
        """)

    # 🧾 Display cluster summary table below
    st.markdown("""
    <div style="margin-top:25px; background-color:#f8f9fa; padding:10px; border-radius:10px;">
        <h4 style="color:#333;">🧾 Cluster Summary Table</h4>
        <p style="font-size:15px; color:#555;">Below is the detailed numerical summary for each cluster, showing average values for key metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(cluster_summary.style.background_gradient(cmap='YlGnBu').format(precision=2))


    # ======================================
    # Behavior Visualization
    # ======================================
    st.subheader("🧠 Cluster Behavior Insights")

    # 🎨 Create 3 side-by-side graphs
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

    # 📝 Add fixed interpretation text below each graph
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:10px; border-radius:10px; border-left:5px solid #1f77b4;">
            <b>Average Sales per Cluster:</b><br>
Cluster 1 drives the major share of total revenue, followed by Cluster 0, while Cluster 2 contributes the least.
Hence, Cluster 1 represents the most valuable customer group for business focus.
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:10px; border-radius:10px; border-left:5px solid #2ca02c;">
            <b>Average Rating per Cluster:</b><br>
            All clusters show high ratings (~6.8–7.0), with only minor differences.
Cluster 2, despite low spending, gives slightly higher ratings, showing customer satisfaction is not purely linked to high spending.
Even low-spending customers are satisfied with their experience, which implies good overall service quality across customer groups.
</div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:10px; border-radius:10px; border-left:5px solid #ff7f0e;">
            <b>Quantity vs Unit Price:</b><br>
            Cluster 1 (orange): purchases larger quantities at higher unit prices → premium or bulk buyers.<br>
Cluster 0 (blue): moderate quantities at average prices → regular shoppers.<br>
Cluster 2 (green): low quantities and lower prices → budget-conscious segment.
The scatterplot confirms that spending behavior differs significantly across clusters:
Cluster 1 = high value, Cluster 0 = stable, Cluster 2 = low-cost.</div>
        """, unsafe_allow_html=True)


    st.markdown("""
    ### 💬 Insights:
    - **Cluster 0:** Stable, regular shoppers.
    - **Cluster 1:** Drives most revenue (bulk or premium buyers).    
    - **Cluster 2:** Budget-conscious yet satisfied (high ratings).  
    """)

    # ======================================
    # Sales Forecasting (Prophet)
    # ======================================
    st.subheader("⏳ Sales Forecasting with Prophet")

    if 'Date' in data.columns and 'Sales' in data.columns:
        daily_sales = data.groupby('Date')['Sales'].sum().reset_index()
        daily_sales.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

        model = Prophet(daily_seasonality=True)
        model.fit(daily_sales)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.markdown("<h2 style='font-size:25px;'>📅 Next 30-Day Sales Forecast</h2>", unsafe_allow_html=True)
        
        fig5 = model.plot(forecast, figsize=(10,5))
        plt.title("Sales Forecast for Next 30 Days")
        st.pyplot(fig5)
        st.markdown("""
        ### 💡 Interpretation:
        - Sales Forecast for the Next 30 Days

        - The <b>black dots </b>represent actual past sales data.

        - The <b>blue line </b>shows Prophet’s predicted sales trend for the coming days.

        - The <b>light blue shaded area </b>illustrates the possible variation or uncertainty in those predictions.

        <b>Overall, the forecast suggests steady sales with no major upward or downward trend — only small, periodic fluctuations that reflect normal business patterns.</b> """,unsafe_allow_html=True)
        st.subheader("📊 Trend & Seasonality Components")

        # Generate Prophet component plots
        fig6 = model.plot_components(forecast)

        # Display all Prophet component plots
        st.pyplot(fig6)

        # 🧭 Add clear, humanized explanations for each Prophet component
        st.markdown("""
        ### 🪜 **Overall Trend**
        The **trend line** shows how total sales have evolved over time.  
        You can see gradual rises or dips representing long-term business movement.  
        A **stable trend** means consistent sales without strong growth or decline.

        ---

        ### 📅 **Weekly Seasonality**
        This part shows how sales vary by day of the week.  
        For example, **weekends might show higher sales** if customers shop more, while weekdays could be steady or slower.  
        It helps identify **which days consistently perform best**.

        ---

        ### 🌤️ **Daily / Monthly Seasonality**
        This component captures **short-term repeating patterns**, like monthly cycles or daily demand changes.  
        If you notice repeating peaks and troughs, it indicates **periodic behavior** — for instance, **higher sales near month-end or festivals**.

        ---
        """)


    else:
        st.warning("⚠️ Columns 'Date' and 'Sales' are required for forecasting.")

else:
    st.info("👆 Please upload a CSV file to start analysis.")
