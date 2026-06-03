import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

TOP_N_GENES = 20  # use top 20 most variable genes as patient features


def build_graph(df):
    # -------------------------------
    # Step 1: select top N most variable genes
    # These are the genes that differ most between patients
    # and carry the most discriminative signal
    # -------------------------------
    gene_variance = df.groupby("gene")["expression"].var().fillna(0)
    top_genes = gene_variance.nlargest(TOP_N_GENES).index.tolist()

    # Filter df to only top genes for feature construction
    df_top = df[df["gene"].isin(top_genes)]

    # -------------------------------
    # Step 2: unique nodes (use ALL genes for graph edges, top genes for features)
    # -------------------------------
    patients = df["patient"].unique()
    genes = df["gene"].unique()

    all_nodes = list(patients) + list(genes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    num_nodes = len(all_nodes)

    # -------------------------------
    # Step 3: build edges (bidirectional)
    # -------------------------------
    src, dst, edge_weights = [], [], []

    for _, row in df.iterrows():
        p = node_to_idx[row["patient"]]
        g = node_to_idx[row["gene"]]
        w = float(row["expression"])

        src.append(p);  dst.append(g);  edge_weights.append(w)
        src.append(g);  dst.append(p);  edge_weights.append(w)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    # -------------------------------
    # Step 4: build patient features from top N genes
    # Each patient gets a vector of length TOP_N_GENES
    # containing their expression value for each top gene
    # -------------------------------
    # pivot: rows = patients, cols = top genes
    pivot = df_top.pivot_table(
        index="patient",
        columns="gene",
        values="expression",
        aggfunc="mean"
    ).reindex(columns=top_genes, fill_value=0.0)

    # normalize per gene across patients to [0, 1]
    pivot_np = pivot.values.astype(np.float32)
    col_min = pivot_np.min(axis=0)
    col_max = pivot_np.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    pivot_norm = (pivot_np - col_min) / denom

    # build feature matrix for ALL nodes
    # patients get their gene expression vector
    # gene nodes get a zero vector
    x = torch.zeros((num_nodes, TOP_N_GENES), dtype=torch.float)

    for i, patient in enumerate(pivot.index):
        if patient in node_to_idx:
            idx = node_to_idx[patient]
            x[idx] = torch.tensor(pivot_norm[i], dtype=torch.float)

    # -------------------------------
    # Step 5: create graph
    # -------------------------------
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return graph_data, node_to_idx