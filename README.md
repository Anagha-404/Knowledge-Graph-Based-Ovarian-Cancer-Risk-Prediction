# Knowledge Graph-Based Ovarian Cancer Risk Prediction using Graph Neural Networks and Explainable AI

A mini project that combines **Knowledge Graphs**, **Random Forest**, and **Explainable AI (SHAP)** to predict ovarian cancer risk from patient gene expression data while identifying the key genes driving each prediction.

Submitted to the APJ Abdul Kalam Technological University in partial fulfillment of the requirements for the B.Tech degree in Computer Science and Engineering.

**Department of Computer Science and Engineering**  
Mar Baselios College of Engineering and Technology (Autonomous)  
Thiruvananthapuram — April 2026

---

## Team

- Anagha Asok Kumar (B23CS1112, MBT23CS022)
- Chaithanya P Sunil (B23CS1123, MBT23CS044)
- Gowri Ajai (B23CS1132, MBT23CS062)
- Neha Nevin (B23CS1148, MBT23CS094)

**Guide:** Ms. Diana Mathew, Assistant Professor, Dept. of CSE

---

## Overview

Predicting disease risk from gene expression data is challenging because genes interact through complex biological pathways that traditional machine learning models often fail to capture. This project addresses that gap by:

1. Representing patients and genes as a **Knowledge Graph**, with edges capturing gene expression relationships.
2. Storing and querying this graph using **Neo4j**.
3. Constructing graph-based representations of patient-gene relationships using **PyTorch Geometric**.
4. Classifying patients into **high-risk** and **low-risk** categories using a **Random Forest** classifier trained on the top 20 most variable gene expression features.
5. Explaining predictions with **SHAP (SHapley Additive exPlanations)** to identify the most influential genes driving each prediction.
6. Supporting **real-time updates**, allowing new patient data to be added and predicted without rebuilding the entire knowledge graph.

---

## Problem Statement

Build a knowledge-graph-based framework for predicting ovarian cancer risk from gene expression data by combining graph-based biomedical relationship modeling, Random Forest classification, and Explainable AI, classifying patients into **high-risk** and **low-risk** categories while providing interpretable, gene-level explanations for each prediction.

---

## System Architecture

**Pipeline:**  
Data Preprocessing → Knowledge Graph Construction → Graph Storage (Neo4j) → Graph Processing → Feature Extraction → Random Forest Classification → SHAP Explainability → Result Visualization

| Stage                        | Description                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| Data Preprocessing           | Handles missing values, normalization, class balancing, and selection of the top variable genes |
| Knowledge Graph Construction | Builds nodes (patients, genes) and edges (expression relationships)                             |
| Graph Storage                | Persists the knowledge graph in a Neo4j database                                                |
| Graph Processing             | Constructs patient-gene graph representations and feature matrices                              |
| Feature Extraction           | Generates normalized top-20 gene expression feature vectors for each patient                    |
| Random Forest Classification | Predicts patient risk using graph-derived gene expression features                              |
| Explainability Module        | Uses SHAP to rank genes by importance for each prediction                                       |
| Real-Time Update Module      | Adds new patients and generates predictions without rebuilding the graph                        |

---

## Core Classes

| Class                    | Responsibility                                              |
| ------------------------ | ----------------------------------------------------------- |
| `Patient`                | Represents a patient and their linked gene data             |
| `Gene`                   | Represents a biological gene                                |
| `Expression`             | Stores the gene expression value for a patient-gene pair    |
| `Neo4jConnector`         | Handles database read/write via Cypher queries              |
| `Extract`                | Retrieves and prepares expression data and labels           |
| `DataFrame`              | Tabular intermediate representation of processed data       |
| `BuildGraph`             | Converts processed data into a patient-gene graph structure |
| `RandomForestClassifier` | Performs patient risk classification                        |
| `Train`                  | Trains and evaluates the Random Forest model                |
| `PredictNew`             | Adds new patients and generates real-time predictions       |

---

## Tech Stack

| Component       | Technology                               |
| --------------- | ---------------------------------------- |
| OS              | Windows / Linux                          |
| Language        | Python                                   |
| Graph Database  | Neo4j                                    |
| ML/DL Libraries | PyTorch, PyTorch Geometric, scikit-learn |
| Data Processing | Pandas, NumPy                            |
| Explainability  | SHAP                                     |
| Visualization   | Streamlit, PyVis, Matplotlib, NetworkX   |

**Minimum Requirements:** Intel i5 or higher, 8 GB RAM (16 GB recommended), 10 GB free storage, internet access for dataset/library installation.

---

## Results

### Classification Performance

| Evaluation          | Accuracy  | Precision | Recall    | F1-Score  |
| ------------------- | --------- | --------- | --------- | --------- |
| Train Set           | 92.2%     | 86.5%     | 100%      | 92.8%     |
| Test Set            | 53.6%     | 53.3%     | 57.1%     | 55.2%     |
| **Overall Dataset** | **80.4%** | **76.9%** | **87.0%** | **81.6%** |

### Cross-Validation Performance

| Evaluation                       | Result        |
| -------------------------------- | ------------- |
| 5-Fold Cross-Validation Accuracy | 59.4% ± 15.7% |
| 5-Fold Cross-Validation Recall   | 66.2% ± 23.0% |
| 5-Fold Cross-Validation F1-Score | 60.6% ± 17.8% |

### Computational Performance

| Model                                    | Training Time | Prediction Speed |
| ---------------------------------------- | ------------- | ---------------- |
| Random Forest                            | Low           | Very Fast        |
| Knowledge Graph + Random Forest Pipeline | Moderate      | Fast             |

The proposed framework effectively combines knowledge graph modeling with Random Forest classification to provide interpretable ovarian cancer risk predictions. Although the dataset size limits generalization, SHAP explanations and graph-based relationship modeling improve transparency and clinical interpretability.

---

## Project Structure (Design Reference)

```text
├── data/                  # Raw and preprocessed gene expression datasets
├── graph/                 # Knowledge graph construction scripts (Neo4j integration)
├── models/                # Random Forest training and evaluation scripts
├── explainability/        # SHAP-based explanation generation
├── predict/               # Real-time prediction module
├── app/                   # Streamlit interface and graph visualization
└── docs/                  # Diagrams: Use Case, Class, DFD (Levels 0–2), Architecture
```

> **Note:** This README documents the design and implementation described in the project report. Update the structure above to match the actual repository layout if the codebase differs.

---

## Key Features

- Graph-based modeling of patient-gene relationships using a Neo4j knowledge graph.
- Neo4j-backed knowledge graph for scalable and queryable biomedical data storage.
- Feature extraction using the top 20 most variable gene expression values.
- Random Forest-based ovarian cancer risk classification.
- SHAP-based explainability for clinically interpretable predictions.
- Interactive visualization through Streamlit and PyVis.
- Dynamic support for adding new patient data and generating predictions.

---

## References

1. J. Chen et al., _Prediction of Ovarian Cancer-Related Metabolites Based on Graph Neural Network_, _Frontiers in Cell and Developmental Biology_, vol. 9, 2021.
2. G. S. P. Ghantasala et al., _Enhanced Ovarian Cancer Survival Prediction using Temporal Analysis and Graph Neural Networks_, _BMC Medical Informatics and Decision Making_, vol. 24, 2024.
3. K. Tan et al., _A Hierarchical Graph Convolution Network for Representation Learning of Gene Expression Data_, _IEEE Journal of Biomedical and Health Informatics_, vol. 25, no. 8, 2021.
4. D. R. Cox, _Regression Models and Life-Tables_, _Journal of the Royal Statistical Society: Series B_, vol. 34, no. 2, 1972.
5. S. Ji et al., _A Survey on Knowledge Graphs: Representation, Acquisition, and Applications_, _IEEE Transactions on Neural Networks and Learning Systems_, vol. 33, no. 2, 2022.

---

## Acknowledgement

This project was carried out under the guidance of **Ms. Diana Mathew, Assistant Professor**, and coordinated by **Ms. Anjali S**, in the Department of Computer Science and Engineering, Mar Baselios College of Engineering and Technology.
