import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Prediction", layout="wide")
st.title(" Wine Quality Prediction App")

#Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model_bundle = load_model()
model = model_bundle['model']
feature_names = model_bundle['feature_names']
target_names = model_bundle['target_names']

#this is for side bar
st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader("Upload Wine CSV", type=["csv"])

#Load dataset
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("data/winequality-red.csv")

df = load_data(uploaded)

# For Data Overview 
st.subheader("Dataset Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# Interactive Filter 
st.subheader("Filter by Feature")
col = st.selectbox("Choose a column to filter", feature_names)
min_val, max_val = float(df[col].min()), float(df[col].max())
rng = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
filtered = df[df[col].between(rng[0], rng[1])]
st.write(f"Filtered rows: {filtered.shape[0]}")

# Visualisations 
st.subheader("Data Visualisations")

fig1 = px.histogram(df, x=col, nbins=20, title=f"Distribution of {col}")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.scatter(df, x=feature_names[0], y=feature_names[1],
                  color=df['quality'] if 'quality' in df.columns else None,
                  title="Scatter Plot")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.box(df, y=col, title=f"Boxplot of {col}")
st.plotly_chart(fig3, use_container_width=True)

# Prediction 
st.subheader("Make a Prediction")
vals = []
cols = st.columns(len(feature_names))
for i, fname in enumerate(feature_names):
    with cols[i]:
        vals.append(st.number_input(fname, float(df[fname].min()),
                                    float(df[fname].max()),
                                    float(df[fname].mean())))

input_arr = np.array(vals).reshape(1, -1)

if st.button("Predict"):
    pred = model.predict(input_arr)[0]
    st.success(f"Predicted Quality: {target_names[int(pred)]}")

