from src.extract import get_expression_data
from src.build_graph import build_graph
from src.train import train_model
from src.predict_new import add_patient_to_neo4j, predict_new_patient
from src.connect import run_query
from src.shap_explain import shap_explain

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from mapping import get_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -------------------------------
# LOAD DATA
# -------------------------------
print("\nLoading data...")
df = get_expression_data()
label_df = get_labels()

label_df = label_df.drop_duplicates(subset="patient")  # fix cartesian product
df = df.merge(label_df, on="patient", how="inner")

print("Patients in expression data:", df["patient"].nunique())
print("Patients in label data:", label_df["patient"].nunique())
print("Patients after merge:", df.shape[0])
print("Sample patient IDs from expression data:", df["patient"].unique()[:5])
print("Sample patient IDs from label data:", label_df["patient"].unique()[:5])
# -------------------------------
# BALANCE DATA (PATIENT LEVEL)
# -------------------------------
patients_0 = df[df["label"] == 0]["patient"].unique()
patients_1 = df[df["label"] == 1]["patient"].unique()


min_count = min(len(patients_0), len(patients_1))
p0 = patients_0[:min_count]
p1 = patients_1[:min_count]

# p1 = patients_1[:200]

df = df[df["patient"].isin(list(p0) + list(p1))]

print(f"\nBalanced dataset: {len(p0)} low-risk patients, {len(p1)} high-risk patients")


# -------------------------------
# BUILD GRAPH
# -------------------------------
print("\nBuilding graph...")
graph_data, node_map = build_graph(df)
print("Graph built!")
print("\nGRAPH INFO:")
print(graph_data)


# -------------------------------
# CREATE LABEL TENSOR
# Only patient nodes get real labels (0 or 1).
# Gene nodes stay 0 and are excluded from training/evaluation via masks.
# -------------------------------
labels = torch.zeros(len(node_map), dtype=torch.long)

patient_indices = []

for _, row in label_df.iterrows():
    if row["patient"] in node_map:
        idx = node_map[row["patient"]]
        labels[idx] = row["label"]
        patient_indices.append(idx)


patient_indices = torch.tensor(patient_indices)

print("\nPatient label distribution (before split):")
patient_labels = labels[patient_indices]
print(pd.Series(patient_labels.numpy()).value_counts())

# ADD HERE:
patient_feats = graph_data.x[patient_indices]
print("\nPatient feature variance:", patient_feats.var(dim=0))
print("Sample patient features:")
print(patient_feats[:5])
# -------------------------------
# TRAIN-TEST SPLIT — ON PATIENTS ONLY
# -------------------------------
p_idx_list = patient_indices.tolist()
p_labels_list = patient_labels.tolist()

train_p_idx, test_p_idx, _, _ = train_test_split(
    p_idx_list,
    p_labels_list,
    test_size=0.3,
    random_state=42,
    stratify=p_labels_list
)

train_idx = torch.tensor(train_p_idx)
test_idx = torch.tensor(test_p_idx)

print(f"\nTrain patients: {len(train_idx)}, Test patients: {len(test_idx)}")
print("Train label distribution:", pd.Series(labels[train_idx].numpy()).value_counts().to_dict())
print("Test label distribution: ", pd.Series(labels[test_idx].numpy()).value_counts().to_dict())


# -------------------------------
# COMPUTE CLASS WEIGHTS
# -------------------------------
train_labels_np = labels[train_idx].numpy()
num_class_0 = (train_labels_np == 0).sum()
num_class_1 = (train_labels_np == 1).sum()

total = len(train_labels_np)
weight_0 = total / (2 * num_class_0) if num_class_0 > 0 else 1.0
weight_1 = total / (2 * num_class_1) if num_class_1 > 0 else 1.0

class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)
print(f"\nClass weights → Low Risk: {weight_0:.2f}, High Risk: {weight_1:.2f}")


# -------------------------------
# TRAIN MODEL
# -------------------------------
print("\nTraining model...")
clf, _, _ = train_model(
    graph_data,
    labels,
    train_mask=train_idx,
    class_weights=class_weights
)


# -------------------------------
# INFERENCE — run once, reuse for all evaluations
# -------------------------------
# clf returned from train_model, preds and prob already computed
X_all = graph_data.x.numpy()
all_preds = torch.tensor(clf.predict(X_all), dtype=torch.long)
prob = torch.tensor(clf.predict_proba(X_all)[:, 1], dtype=torch.float)

# =====================================================
# 1. TRAIN SET EVALUATION
# =====================================================
y_true_train = labels[train_idx].cpu().numpy()
y_pred_train = all_preds[train_idx].cpu().numpy()

print("\n" + "="*50)
print("EVALUATION METRICS (TRAIN SET)")
print("="*50)
print("Accuracy: ", accuracy_score(y_true_train, y_pred_train))
print("Precision:", precision_score(y_true_train, y_pred_train, zero_division=0))
print("Recall:   ", recall_score(y_true_train, y_pred_train, zero_division=0))
print("F1 Score: ", f1_score(y_true_train, y_pred_train, zero_division=0))

cm_train = confusion_matrix(y_true_train, y_pred_train)
print("\nTrain Confusion Matrix:")
print(cm_train)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Train Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_train.png")
plt.show()


# =====================================================
# 2. TEST SET EVALUATION
# =====================================================
y_true_test = labels[test_idx].cpu().numpy()
y_pred_test = all_preds[test_idx].cpu().numpy()

print("\n" + "="*50)
print("EVALUATION METRICS (TEST SET)")
print("="*50)
print("Accuracy: ", accuracy_score(y_true_test, y_pred_test))
print("Precision:", precision_score(y_true_test, y_pred_test, zero_division=0))
print("Recall:   ", recall_score(y_true_test, y_pred_test, zero_division=0))
print("F1 Score: ", f1_score(y_true_test, y_pred_test, zero_division=0))

cm_test = confusion_matrix(y_true_test, y_pred_test)
print("\nTest Confusion Matrix:")
print(cm_test)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")
plt.show()


# =====================================================
# 3. OVERALL (ALL PATIENTS) EVALUATION
# =====================================================
y_true_all = labels[patient_indices].cpu().numpy()
y_pred_all = all_preds[patient_indices].cpu().numpy()

print("\n" + "="*50)
print("EVALUATION METRICS (ALL PATIENTS)")
print("="*50)
print("Accuracy: ", accuracy_score(y_true_all, y_pred_all))
print("Precision:", precision_score(y_true_all, y_pred_all, zero_division=0))
print("Recall:   ", recall_score(y_true_all, y_pred_all, zero_division=0))
print("F1 Score: ", f1_score(y_true_all, y_pred_all, zero_division=0))

cm_all = confusion_matrix(y_true_all, y_pred_all)
print("\nOverall Confusion Matrix:")
print(cm_all)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_all, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Overall Confusion Matrix (All Patients)")
plt.tight_layout()
plt.savefig("confusion_matrix_overall.png")
plt.show()


# =====================================================
# USER INPUT — NEW PATIENT PREDICTION
# =====================================================
print("\n--- ENTER NEW PATIENT DATA ---")

new_patient = input("Enter patient name: ")

gene_data = []
num_genes = int(input("Number of genes: "))

for i in range(num_genes):
    gene = input(f"Gene {i+1}: ")
    value = float(input(f"Expression value: "))
    gene_data.append((gene, value))

print("\nPatient data collected")

# Add to Neo4j
add_patient_to_neo4j(new_patient, gene_data)

# Rebuild graph with new patient included
df_full = get_expression_data()
new_patient_df = df_full[df_full["patient"] == new_patient]
df_sample = df_full.sample(10000, random_state=42)
df_final = pd.concat([df_sample, new_patient_df]).drop_duplicates()

graph_data_new, node_map_new = build_graph(df_final)

# Predict
prediction = predict_new_patient(model, new_patient, graph_data_new, node_map_new)
print(f"\nPrediction for {new_patient}: {prediction}")

# Top genes from Neo4j
query = f"""
MATCH (p:Patient {{name: '{new_patient}'}})-[r:EXPRESSES]->(g:Gene)
RETURN p.name AS patient, g.name AS gene, r.value AS expression
"""
data = run_query(query)
df_new = pd.DataFrame(data)

top_genes = df_new.sort_values(by="expression", ascending=False).head(5)
print("\nTop genes influencing prediction:")
print(top_genes)

# SHAP Explanation
print("\nRunning SHAP explanation...")
df_shap_sample = df_final.sample(3000, random_state=42)
shap_explain(df_shap_sample)