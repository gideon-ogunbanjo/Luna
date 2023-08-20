import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)

# Page configuration
st.set_page_config(
    page_title="Luna",
    layout="centered"
)

# App title
st.title("Luna - Interactive Model Tuning, Algorithm Recommendation and Evaluation App")
st.write("Luna is an interactive model tuning and evaluation app. Upload a CSV dataset, select various machine learning algorithms, fine-tune hyperparameters, and visualize model performance metrics.")
st.header("Get Started Now!")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)

    # Sidebar options
    st.sidebar.header("Model Configuration")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression", "Clustering", "Dimensionality Reduction"])
    algorithms = st.sidebar.multiselect("Select Algorithms", ["Random Forest", "Gradient Boosting", "SVM", "KNN", "Decision Tree", "Linear Regression", "Random Forest Regression", "K-Means Clustering", "Hierarchical Clustering", "DBSCAN", "PCA", "t-SNE"])

    # Split the dataset
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if problem_type == "Classification":
        y = pd.factorize(y)[0]  # Convert categorical target to numerical labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Algorithm Recommendation
    def recommend_algorithm(X, y, problem_type, unsupervised_algorithms):
        num_samples, num_features = X.shape
        if problem_type == "Classification":
            if num_samples > 1000 and num_features > 10:
                return "Random Forest"
            else:
                return "Gradient Boosting"
        elif problem_type == "Regression":
            if num_samples > 1000 and num_features > 10:
                return "Random Forest Regression"
            else:
                return "Linear Regression"
        elif problem_type == "Clustering":
            if "K-Means Clustering" in unsupervised_algorithms:
                return "K-Means Clustering"
            elif "Hierarchical Clustering" in unsupervised_algorithms:
                return "Hierarchical Clustering"
            elif "DBSCAN" in unsupervised_algorithms:
                return "DBSCAN"
        elif problem_type == "Dimensionality Reduction":
            if "PCA" in unsupervised_algorithms:
                return "PCA"
            elif "t-SNE" in unsupervised_algorithms:
                return "t-SNE"
        return None

    # Display algorithm recommendation
    recommended_algorithm = recommend_algorithm(X_train, y_train, problem_type, algorithms)
    st.sidebar.subheader("Recommended Algorithm")
    st.sidebar.write(f"The recommended algorithm for this dataset is: **{recommended_algorithm}**")

    # Display model performance for selected algorithms
    st.subheader("Model Performance")

    for algorithm in algorithms:
        st.write(f"Recommended Algorithm: {algorithm}")

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
            elif algorithm == "K-Means Clustering":
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                unsupervised_clf = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = unsupervised_clf.fit_predict(X)
                st.write(f"Cluster Labels: {cluster_labels}")
            elif algorithm == "Hierarchical Clustering":
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                linkage = st.selectbox("Linkage Method", ["ward", "complete", "average"])
                unsupervised_clf = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                cluster_labels = unsupervised_clf.fit_predict(X)
                st.write(f"Cluster Labels: {cluster_labels}")
            elif algorithm == "DBSCAN":
                eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
                min_samples = st.slider("Min Samples", 2, 20, 5)
                unsupervised_clf = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = unsupervised_clf.fit_predict(X)
                st.write(f"Cluster Labels: {cluster_labels}")
            elif algorithm == "PCA":
                num_components = st.slider("Number of Components", 1, min(X.shape), 2)
                unsupervised_clf = PCA(n_components=num_components, random_state=42)
                reduced_X = unsupervised_clf.fit_transform(X)
                st.write(f"Explained Variance Ratios: {unsupervised_clf.explained_variance_ratio_}")
            elif algorithm == "t-SNE":
                perplexity = st.slider("Perplexity", 5, 50, 30)
                n_iter = st.slider("Number of Iterations", 250, 1000, 500)
                unsupervised_clf = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
                reduced_X = unsupervised_clf.fit_transform(X)
                st.subheader("t-SNE Visualization")
                plt.scatter(reduced_X[:, 0], reduced_X[:, 1], marker="o")
                plt.xlabel("t-SNE Component 1")
                plt.ylabel("t-SNE Component 2")
                st.pyplot(plt)

        if algorithm not in ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN", "PCA", "t-SNE"]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            if problem_type == "Classification":
                score = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {score:.2f}")
            else:
                score = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {score:.2f}")
    
    # Data Visualization
    st.subheader("Data Visualization")
    st.write("Explore and visualize your data:")
    # Plotly Scatter Plot
    st.sidebar.subheader("Data Visualization")
    x_column = st.sidebar.selectbox("X Axis", df.columns)
    y_column = st.sidebar.selectbox("Y Axis", df.columns)

    st.subheader("Plotly Scatter Plot")

    if st.sidebar.button("Generate Plotly Scatter Plot"):
        fig = px.scatter(df, x=x_column, y=y_column, title='Scatter Plot')
        st.plotly_chart(fig, use_container_width=True)

    link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
    st.markdown(link, unsafe_allow_html=True)
