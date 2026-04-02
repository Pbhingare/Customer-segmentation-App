import streamlit as st
import pandas as pd
from src.preprocessing import get_features
from src.model import train_kmeans, elbow_method
from src.visualization import plot_clusters, plot_elbow

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation Dashboard")

# Sidebar
st.sidebar.header("⚙️ Controls")
k = st.sidebar.slider("Select number of clusters", 2, 10, 5)
show_elbow = st.sidebar.checkbox("Show Elbow Method")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    dataset = pd.read_csv(uploaded_file)
else:
    dataset = pd.read_csv(r"C:\Users\prach\Downloads\Mall_Customers.csv")

st.write("Dataset Preview:", dataset.head())

# Preprocessing
x = get_features(dataset)

# Elbow method
if show_elbow:
    wcss = elbow_method(x)
    fig1 = plot_elbow(wcss)
    st.pyplot(fig1)

# Model training
model, y_kmeans = train_kmeans(x, k)

# Add cluster column
dataset['cluster'] = y_kmeans

# Plot clusters
fig2 = plot_clusters(x, y_kmeans, model, k)
st.pyplot(fig2)

# Show summary
st.subheader("📊 Cluster Summary")
st.write(dataset.groupby('cluster').mean(numeric_only=True))

# Metrics
st.metric("Total Customers", len(dataset))
st.metric("Clusters", k)