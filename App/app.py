import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Luna",
    layout="centered"
)

# App title
st.title("Luna - Interactive Model Tuning App")
st.write("Luna is an app that allows you to upload your datasets, choose machine learning algorithms, and interactively tune hyperparameters and see the effects on model performance.")
st.header("Get Started!")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])