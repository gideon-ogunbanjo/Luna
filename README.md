# Luna - Interactive Model Tuning App

Luna is an app that allows users to upload a CSV dataset, select different machine learning algorithms, tune hyperparameters, and view the model performance metrics
## Features
- File Upload: Luna allows users upload CSV files to the app.
- Target Column and Problem Type Selection: Luna allows users select the target column and problem type (Classification or Regression) from the sidebar.
- Algorithm Selection: Luna has a wide variety of supervised learning algorithms which users can select from a list (Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, Linear Regression, Random Forest Regression).
- Algorithm Selection: Luna has an in-built algorithm recommendation system that recommends the most suitable algorithm based on the dataset characteristics and problem type.
- Hyper Parameter Configuration: Luna allows users configure hyperparameters of selected algorithms using sliders.
- Interactive User Interface: Luna has an interactive UI that allows users view the model's performance metrics, such as accuracy for classification and mean squared error for regression.

### Limitations
1. Limited Dataset Size: Luna is designed to handle relatively small to medium-sized datasets. For large datasets with thousands of samples and numerous features, the app's performance might be compromised, leading to slower execution times and potential memory limitations.

2. Algorithm Selection: While Luna provides a variety of machine learning algorithms to choose from, the selection is not exhaustive. Luna is limited to **Supervised Learning Algorithms** only. Users might require algorithms that are not included in the app. Additionally, the app's algorithm recommendations are based on basic dataset characteristics and might not account for domain-specific considerations.

3. Hyperparameter Tuning: While the app enables users to adjust hyperparameters, it offers only a limited set of hyperparameters for tuning. Advanced users might need access to a broader range of hyperparameters to fine-tune algorithms more precisely.

4. No Feature Engineering: Luna focuses primarily on algorithm selection, hyperparameter tuning, and performance evaluation. It does not include tools for feature engineering, which is a crucial aspect of the machine learning pipeline.

5. Simplified Problem Types: The app categorizes problems into classification and regression types. Complex problems, such as time series analysis, anomaly detection, and clustering, are not addressed by Luna's current version.

6. No Data Preprocessing: Data preprocessing, such as handling missing values, outliers, and encoding categorical variables, is not covered by the app. Users need to preprocess the data externally before uploading it to Luna.

7. Dataset Type: Luna is limited to **CSV Files** only for now. More updates will be released with diversity across all types of datasets.

### Dependencies
The app is built using Python and the following libraries:

Streamlit
Pandas
Scikit-learn
You can find these dependencies listed in the `requirements.txt` file.

> Luna is an evolving project, and its limitations might change as new features are added and the app is updated.
### Creator
Gideon Ogunbanjo