import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Reading the dataset
    df = pd.read_csv(uploaded_file)

    # Sidebar options
    st.sidebar.header("Model Configuration")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    algorithms = st.sidebar.multiselect("Select Algorithms", ["Random Forest", "Gradient Boosting", "SVM"])

    # Splits the dataset
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Displays model performance for selected algorithms
    st.subheader("Model Performance")

    for algorithm in algorithms:
        st.write(f"**{algorithm}**")

        # Setting up and configure the classifier based on user choice
        if algorithm == "Random Forest":
            n_estimators = st.slider("Number of Estimators", 10, 100, 50)
            max_depth = st.slider("Max Depth", 1, 20, 10)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif algorithm == "Gradient Boosting":
            n_estimators = st.slider("Number of Estimators", 10, 100, 50)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        elif algorithm == "SVM":
            C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
            clf = SVC(C=C, random_state=42)
        elif algorithm == "KNN":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif algorithm == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 5)
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            
        # Training and evaluating the classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display accuracy
        st.write(f"Accuracy: {accuracy:.2f}")

    
    
