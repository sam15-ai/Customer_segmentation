import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Add CSS styling and emojis
st.markdown(
    """
    <style>
    .stApp {
        background-color:rgb(247, 243, 221); 
    }
    
    .app-title {
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding-bottom: 65px;
    }
    .upload-section {
        //color:rgb(60, 55, 61);
        font-size: 24px;
        font-weight: bold;
        text-align: left;
    }
    .success-message {
        color: #4CAF50;
        font-size: 20px;
        font-weight: bold;
    }
    .error-message {
        color: #ff4d4d;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title with emoji
st.markdown('<div class="app-title">📊 Customer Segmentation with CSV Upload</div>', unsafe_allow_html=True)

# Upload CSV file section
st.markdown('<div class="upload-section">📂 Upload Customer Data</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📥 Choose a CSV file", type="csv")

if uploaded_file:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("#### 🗂️ Uploaded Data Preview")
    st.write(data.head())

    # Check if required columns exist
    required_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
    if all(col in data.columns for col in required_columns):
        # Preprocess the data
        st.markdown('<div class="success-message">✅ Processing Data...</div>', unsafe_allow_html=True)
        features = data[required_columns]
        scaled_features = scaler.transform(features)

        # Predict clusters
        data['Cluster'] = kmeans.predict(scaled_features)

        # Display the clustered data
        st.write("#### 🏷️ Clustered Data")
        st.write(data.head())

        # Visualize the clusters
        st.write("#### 📊 Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=data['Annual Income (k$)'],
            y=data['Spending Score (1-100)'],
            hue=data['Cluster'],
            palette='Set1',
            ax=ax
        )
        plt.title("Customer Clusters")
        st.pyplot(fig)

        # Option to download the results
        st.write("#### 📤 Download Clustered Data")
        csv = data.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name='clustered_customers.csv',
            mime='text/csv',
        )
    else:
        st.markdown(f'<div class="error-message">❌ The uploaded file must contain the following columns: {required_columns}</div>', unsafe_allow_html=True)
else:
    st.info("ℹ️ Please upload a CSV file to proceed.")
