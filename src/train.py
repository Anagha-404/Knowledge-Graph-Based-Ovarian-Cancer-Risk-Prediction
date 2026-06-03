# src/train.py — replace entire file with this
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import numpy as np


def train_model(data, labels, train_mask=None, class_weights=None):

    # Extract patient features from graph
    X = data.x.numpy()
    y = labels.numpy()

    X_train = X[train_mask]
    y_train = y[train_mask]

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,        # ADD THIS — prevents deep overfitting trees
        min_samples_leaf=3, # ADD THIS — requires at least 3 samples per leaf
        class_weight="balanced",
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    from sklearn.model_selection import cross_val_score, StratifiedKFold

# Cross-validation on train patients only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_patients = data.x.numpy()[train_mask]
    y_patients = labels.numpy()[train_mask]

    cv_accuracy = cross_val_score(clf, X_patients, y_patients, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(clf, X_patients, y_patients, cv=cv, scoring='f1')
    cv_recall = cross_val_score(clf, X_patients, y_patients, cv=cv, scoring='recall')

    print("\n" + "="*50)
    print("5-FOLD CROSS VALIDATION RESULTS")
    print("="*50)
    print(f"Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"Recall:   {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")

    # Predict on all nodes (gene nodes will get label 0, doesn't matter)
    all_preds = clf.predict(X)
    all_probs = clf.predict_proba(X)[:, 1]

    # Evaluate on train patients
    y_pred_train = all_preds[train_mask]
    print("\nTRAIN SET EVALUATION METRICS (patients only):")
    print("Accuracy: ", accuracy_score(y_train, y_pred_train))
    print("Precision:", precision_score(y_train, y_pred_train, zero_division=0))
    print("Recall:   ", recall_score(y_train, y_pred_train, zero_division=0))
    print("F1 Score: ", f1_score(y_train, y_pred_train, zero_division=0))
    print("\nTrain Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred_train))

    preds = torch.tensor(all_preds, dtype=torch.long)
    prob = torch.tensor(all_probs, dtype=torch.float)

    return clf, preds, prob