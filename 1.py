'''
Mock_Scikit

1 problem 1 : https://tinyurl.com/scikitMockInterviewQuestion
'''

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.svm import SVC # type: ignore

# Load the dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert 'TotalCharges' to numeric, handling errors as NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing 'TotalCharges'
df_clean = df.dropna(subset=["TotalCharges"]).copy()

# Drop 'customerID' as it is not useful for prediction
df_clean.drop(columns=["customerID"], inplace=True)

# Separate features (X) and target (y)
X = df_clean.drop("Churn", axis=1)
y = df_clean["Churn"]

# Encode target variable: No -> 0, Yes -> 1
y = LabelEncoder().fit_transform(y)

# Identify numerical and categorical columns
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
categorical_features = [col for col in categorical_features if col not in numeric_features]

# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True)
}

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train and evaluate models
model_results = {}
for model_name, model in models.items():
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_results[model_name] = {
        "model": clf,
        "accuracy": acc,
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

# Select the best model based on accuracy
best_model_name = max(model_results, key=lambda name: model_results[name]["accuracy"])
best_model = model_results[best_model_name]["model"]
best_accuracy = model_results[best_model_name]["accuracy"]

# Print best model
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.2f}")
