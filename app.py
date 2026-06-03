import streamlit as st
import pandas as pd
import torch

# Your project imports
from src.extract import get_expression_data
from src.build_graph import build_graph
from src.train import train_model
from src.predict_new import add_patient_to_neo4j, predict_new_patient
from mapping import get_labels

# -----------------------------
# LOAD + TRAIN MODEL (ONLY ONCE)
# -----------------------------
@st.cache_resource
def load_model_pipeline():
    df = get_expression_data()
    label_df = get_labels()

    # Limit for speed
    df = df.sample(30000)

    # Build graph
    graph_data, node_map = build_graph(df)

    # Create labels
    labels = torch.zeros(len(node_map), dtype=torch.long)

    for _, row in label_df.iterrows():
        if row["patient"] in node_map:
            labels[node_map[row["patient"]]] = row["label"]

    # Train model
    model, _, _ = train_model(graph_data, labels)


    return model, graph_data, node_map


# Load once
model, graph_data, node_map = load_model_pipeline()

# -----------------------------
# UI STARTS HERE
st.title("Cancer Risk Prediction using GNN")
st.write("Enter gene expression values to predict patient risk.")

# Inputs
patient_name = st.text_input("Enter Patient Name")
num_genes = st.number_input("Number of genes", min_value=1, max_value=10, step=1)

gene_data = []

st.subheader("Enter Gene Data")

for i in range(int(num_genes)):
    col1, col2 = st.columns(2)

    with col1:
        gene = st.text_input(f"Gene {i+1}", key=f"gene_{i}")

    with col2:
        value = st.number_input(f"Expression {i+1}", key=f"value_{i}")

    if gene:
        gene_data.append((gene, value))

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict"):

    if not patient_name or len(gene_data) == 0:
        st.warning("Please enter all inputs")

    else:
        st.write("Running prediction...")

        # Add to Neo4j
        add_patient_to_neo4j(patient_name, gene_data)

        # Reload full data
        df_new_full = get_expression_data()

        # Ensure new patient is included
        new_patient_df = df_new_full[df_new_full["patient"] == patient_name]
        df_sample = df_new_full.sample(30000)

        df_combined = pd.concat([df_sample, new_patient_df]).drop_duplicates()

        # Rebuild graph
        graph_data_new, node_map_new = build_graph(df_combined)

        # Predict
        prediction = predict_new_patient(
            model,
            patient_name,
            graph_data_new,
            node_map_new
        )

        st.success(f"Prediction: {prediction}")

        # -----------------------------
        # TOP GENES
        # -----------------------------
        df_patient = df_combined[df_combined["patient"] == patient_name]

        top_genes = df_patient.sort_values(
            by="expression", ascending=False
        ).head(5)

        st.subheader("Top Influencing Genes")
        st.dataframe(top_genes)

        # -----------------------------
        # KNOWLEDGE GRAPH VISUALIZATION
        # -----------------------------
        st.subheader("Knowledge Graph Visualization")

        from pyvis.network import Network
        import networkx as nx
        import streamlit.components.v1 as components
        import tempfile

        # Create graph
        G = nx.Graph()

        for _, row in top_genes.iterrows():
            patient = row["patient"]
            gene = row["gene"]
            value = row["expression"]

            # Patient node
            G.add_node(
                patient,
                label=patient,
                color="#FF6B6B",
                size=25
            )

            # Gene node
            G.add_node(
                gene,
                label=gene,
                color="#4D96FF",
                size=18
            )

            # Edge
            G.add_edge(
                patient,
                gene,
                value=value,
                title=f"Expression: {value}"
            )

        # Create network
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#111111",
            font_color="white"
        )

        net.from_nx(G)

        # Optional styling
        net.set_options("""
        var options = {
          "nodes": {
            "font": { "size": 16 }
          },
          "edges": {
            "smooth": true
          },
          "physics": {
            "enabled": true
          }
        }
        """)

        # Save and display
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html = open(tmp_file.name, "r", encoding="utf-8").read()

        components.html(html, height=600)