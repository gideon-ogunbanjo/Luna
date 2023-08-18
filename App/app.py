import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Luna",
    layout="centered"
)

# App title
st.title("Luna - Interactive Model Tuning App")
st.write("Luna is an app that allows you to upload your datasets, choose machine learning algorithms, and interactively tune hyperparameters and see the effects on model performance.")
st.header("Get Started!")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)

    # Sidebar options
    st.sidebar.header("Model Configuration")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
    algorithms = st.sidebar.multiselect("Select Algorithms", ["Random Forest", "Gradient Boosting", "SVM", "KNN", "Decision Tree", "Linear Regression", "Random Forest Regression"])

    # Split the dataset
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if problem_type == "Classification":
        y = pd.factorize(y)[0]  # Convert categorical target to numerical labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display model performance for selected algorithms
    st.subheader("Model Performance")

    for algorithm in algorithms:
        st.write(f"**{algorithm}**")

        # Model Configuration
        if algorithm in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            if problem_type == "Classification":
                if algorithm == "Random Forest":
                    n_estimators = st.slider("Number of Estimators", 10, 100, 50)
                    max_depth = st.slider("Max Depth", 1, 20, 10)
                    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif algorithm == "Gradient Boosting":
                    n_estimators = st.slider("Number of Estimators", 10, 100, 50)
                    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                elif algorithm == "Decision Tree":
                    max_depth = st.slider("Max Depth", 1, 20, 5)
                    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            else:
                if algorithm == "Random Forest":
                    n_estimators = st.slider("Number of Estimators", 10, 100, 50)
                    max_depth = st.slider("Max Depth", 1, 20, 10)
                    clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif algorithm == "Gradient Boosting":
                    n_estimators = st.slider("Number of Estimators", 10, 100, 50)
                    learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                    clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                elif algorithm == "Decision Tree":
                    max_depth = st.slider("Max Depth", 1, 20, 5)
                    clf = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        elif algorithm == "SVM":
            C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
            clf = SVC(C=C, random_state=42)
        elif algorithm == "KNN":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif algorithm == "Linear Regression":
            clf = LinearRegression()
        elif algorithm == "Random Forest Regression":
            n_estimators = st.slider("Number of Estimators", 10, 100, 50)
            max_depth = st.slider("Max Depth", 1, 20, 10)
            clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if problem_type == "Classification":
            score = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {score:.2f}")
        else:
            score = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {score:.2f}")
