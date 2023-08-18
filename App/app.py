import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Interactive Model Tuning App",
    layout="centered"
)

# App title
st.title("Luna - Interactive Model Tuning App")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])