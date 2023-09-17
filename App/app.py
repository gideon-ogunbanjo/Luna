# Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
import xgboost
import shap  

st.set_option('deprecation.showPyplotGlobalUse', False)

# Page configuration
st.set_page_config(
    page_title="Luna",
    layout="centered"
)

# App title
st.title("Luna - Interactive Model Tuning, Algorithm Recommendation and Evaluation App")
st.write("Luna is an interactive model tuning and evaluation app. Upload a CSV dataset, select various machine learning algorithms, fine-tune hyperparameters, and get code snippets on the selected algorithm.")
st.header("Get Started Now!")

# File Uploader Widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])  # Allows both CSV and Excel files

if uploaded_file is not None:
    # Read the dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')  # 'openpyxl' engine for Excel files

    # Sidebar options
    st.sidebar.header("Model Configuration")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
    algorithms = st.sidebar.multiselect("Select Algorithms", ["Random Forest", "Gradient Boosting", "SVM", "KNN", "Decision Tree", "Linear Regression"])

    # Split the dataset
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if problem_type == "Classification":
        y = pd.factorize(y)[0]  # Convert categorical target to numerical labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Algorithm Recommendation and selection based on their strengths and weaknesses
    def recommend_algorithm(X, y, problem_type):
        num_samples, num_features = X.shape
        if problem_type == "Classification":
            if num_samples > 1000 and num_features > 10:
                if num_samples > 10000:  # For very large datasets
                    return "Random Forest"
                else:
                    return "Gradient Boosting"
            else:
                if num_samples < 500:  # For small datasets
                    return "K-Nearest Neighbors"
                else:
                    return "Support Vector Machine"
        elif problem_type == "Regression":
            if num_samples > 1000 and num_features > 10:
                if num_samples > 10000:  # For very large datasets
                    return "Random Forest Regression"
                else:
                    return "Gradient Boosting Regression"
            else:
                if num_samples < 500:  # For small datasets
                    return "Linear Regression"
                else:
                    return "Support Vector Machine Regression"
        return None

    # Algorithm recommendation
    recommended_algorithm = recommend_algorithm(X_train, y_train, problem_type)
    st.sidebar.subheader("Recommended Algorithm")
    st.sidebar.write(f"The recommended algorithm for this dataset is: **{recommended_algorithm}**")

    # Model performance for selected algorithms
    st.subheader("Model Performance")

    # Algorithm code snippets
    algorithm_code_snippets = {
        "Random Forest": {
            "import": "from sklearn.ensemble import RandomForestClassifier",
            "init": "modelmodel = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)",
            "train": "model.fit(X_train, y_train)",
            "evaluate": "score = accuracy_score(y_test, clf.predict(X_test))",
        },
        "Gradient Boosting": {
            "import": "from sklearn.ensemble import GradientBoostingClassifier",
            "init": "model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)",
            "train": "model.fit(X_train, y_train)",
            "evaluate": "score = accuracy_score(y_test, clf.predict(X_test))",
        },
        "Random Forest Regression": {
            "import": "from sklearn.ensemble import RandomForestRegressor",
            "init": "model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)",
            "train": "model.fit(X_train, y_train)",
            "evaluate": "score = mean_squared_error(y_test, clf.predict(X_test))",
        },
        "Linear Regression": {
            "import": "from sklearn.linear_model import LinearRegression",
            "init": "model = LinearRegression()",
            "train": "model.fit(X_train, y_train)",
            "evaluate": "score = mean_squared_error(y_test, clf.predict(X_test))",
        },
    }
    # Conditions for Model Initialization
    for algorithm in algorithms:
        st.write(f"Recommended Algorithm: **{algorithm}**")

        # Initializing the model based on the selected algorithm
        if algorithm == "Random Forest":
            model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
        elif algorithm == "Gradient Boosting":
            model = GradientBoostingClassifier() if problem_type == "Classification" else GradientBoostingRegressor()
        elif algorithm == "SVM":
            model = SVC()
        elif algorithm == "KNN":
            model = KNeighborsClassifier()
        elif algorithm == "Decision Tree":
            model = DecisionTreeClassifier() if problem_type == "Classification" else DecisionTreeRegressor()
        elif algorithm == "Linear Regression":
            model = LinearRegression()

        # Model training and predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Conditions for Model Evaluation
        if problem_type == "Classification":
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy Score: {accuracy:.4f}")
        elif problem_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.4f}")

        # Explainability using SHAP (added for both classification and regression)
        st.subheader("Model Explanation")

        if algorithm in ["Random Forest", "Gradient Boosting", "Random Forest Regression", "Linear Regression"]:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer.shap_values(X_test)

            # Visualize SHAP summary plot
            st.write("SHAP Summary Plot:")
            shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
            st.pyplot()

        # Code Snippet Initialization
        st.subheader("Code Snippet:")

        algorithm_info = algorithm_code_snippets.get(algorithm)
        if algorithm_info:

            complete_code = f"""
            {algorithm_info['import']}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            {algorithm_info['init']}

            {algorithm_info['train']}

            {algorithm_info['evaluate']}

            # Explainability using SHAP
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer.shap_values(X_test)

            # Visualize SHAP summary plot
            shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)

            print("Evaluation Result:", score)
                        """
            st.code(complete_code, language="python")
        else:
            st.write("No code snippets available for this algorithm.")

    link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
    st.markdown(link, unsafe_allow_html=True)
