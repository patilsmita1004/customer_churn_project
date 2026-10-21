from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the saved Machine Learning pipeline
MODEL_PATH = os.path.join("model", "churn_model.pkl")
model = joblib.load(MODEL_PATH)

# Analytics dataset
DATA_PATH = os.path.join("dataset", "Telco-customer-churn.csv")


@lru_cache(maxsize=1)
def get_analytics():
    df = pd.read_csv(DATA_PATH)

    if "Churn" not in df.columns:
        raise ValueError("Dataset must contain a Churn column.")

    # Clean the Telco dataset
    df = df.drop_duplicates().copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna(subset=["Churn"])

    churn = df["Churn"].astype(str).str.strip()
    df = df[churn.isin(["Yes", "No"])].copy()
    df["Churn"] = churn[churn.isin(["Yes", "No"])]

    total = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    stayed = int((df["Churn"] == "No").sum())

    churn_rate = churned / total * 100 if total else 0
    retention_rate = stayed / total * 100 if total else 0

    # Exclude customer identifiers and target from features
    drop_cols = [
        col for col in ["customerID", "Churn"] if col in df.columns
    ]
    X = df.drop(columns=drop_cols)
    y = (df["Churn"] == "Yes").astype(int)

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    model_names = []
    model_accuracies = []

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        model_names.append(name)
        model_accuracies.append(round(
            accuracy_score(y_test, predictions) * 100, 2
        ))

    return {
        "total_customers": total,
        "churned_customers": churned,
        "retained_customers": stayed,
        "churn_rate": round(churn_rate, 2),
        "retention_rate": round(retention_rate, 2),
        "churn_labels": ["Stayed", "Churned"],
        "churn_values": [stayed, churned],
        "model_names": model_names,
        "model_accuracies": model_accuracies
    }


@app.context_processor
def inject_analytics():
    try:
        return {"analytics": get_analytics(), "analytics_error": None}
    except (FileNotFoundError, ValueError, KeyError) as error:
        return {"analytics": None, "analytics_error": str(error)}



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect customer information from the HTML form
        customer = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": float(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        # Convert the input into a DataFrame
        customer_df = pd.DataFrame([customer])

        # Predict churn
        prediction = model.predict(customer_df)[0]

        # Calculate churn probability when supported
        churn_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer_df)[0]
            classes = list(model.classes_)

            if "Yes" in classes:
                churn_probability = float(
                    probabilities[classes.index("Yes")] * 100
                )

        # Determine risk level
        if churn_probability is None:
            risk_level = "Probability unavailable"
            risk_color = "#64748b"
            risk_action = "Review the model's prediction."
            risk_width = 0

        elif churn_probability < 30:
            risk_level = "Low Risk"
            risk_color = "#16a34a"
            risk_action = (
                "Maintain good service and continue regular customer support."
            )
            risk_width = churn_probability

        elif churn_probability < 60:
            risk_level = "Medium Risk"
            risk_color = "#f59e0b"
            risk_action = (
                "Review customer feedback and consider a suitable retention offer."
            )
            risk_width = churn_probability

        else:
            risk_level = "High Risk"
            risk_color = "#dc2626"
            risk_action = (
                "Contact the customer and investigate possible service concerns."
            )
            risk_width = churn_probability

        # Prepare the prediction message
        if prediction == "Yes":
            message = "The customer is predicted to leave the company."
        else:
            message = "The customer is predicted to stay with the company."

        return render_template(
            "index.html",
            prediction=message,
            churn_probability=(
                round(churn_probability, 2)
                if churn_probability is not None else None
            ),
            risk_level=risk_level,
            risk_color=risk_color,
            risk_width=round(risk_width, 2),
            risk_action=risk_action,
            customer=customer
        )

    except (ValueError, KeyError) as error:
        return render_template(
            "index.html",
            prediction=f"Invalid customer input. Please check the values: {error}"
        )


if __name__ == "__main__":
    app.run(debug=True)