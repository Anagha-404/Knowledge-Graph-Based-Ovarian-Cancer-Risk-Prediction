import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


def shap_explain(df):

    # -------------------------------
    # STEP 1: Convert to matrix
    # -------------------------------
    pivot = df.pivot(index="patient", columns="gene", values="expression").fillna(0)
    X = pivot

    # -------------------------------
    # STEP 2: Create labels (balanced)
    # -------------------------------
    threshold = X.mean().mean()  # global average

    y = (X.mean(axis=1) > threshold).astype(int)

    #IMPORTANT: Ensure both classes exist
    if len(set(y)) < 2:
        print(" Only one class detected, adjusting labels...")
        y.iloc[:len(y)//2] = 0
        y.iloc[len(y)//2:] = 1

    print("\nSHAP label distribution:")
    print(y.value_counts())

    # STEP 3: Train simple model

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y) 

    # STEP 4: SHAP explanation
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    print("\n--- SHAP EXPLANATION ---")
    print("Showing explanation for first patient")

    vals = shap_values[0].values
    features = X.columns

    importance = pd.DataFrame({
        "gene": features,
        "importance": vals
    })

    importance = importance.sort_values(by="importance", ascending=False).head(5)

    print("\nTop SHAP important genes:")
    print(importance)