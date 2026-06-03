from src.connect import run_query
import torch


def add_patient_to_neo4j(patient_name, gene_data):
    """
    gene_data = list of tuples
    Example: [("gene1", 0.8), ("gene2", 0.3)]
    """

    # Create patient node
    run_query(f"""
    MERGE (p:Patient {{name: '{patient_name}'}})
    """)

    # Add genes + relationships
    for gene, value in gene_data:
        run_query(f"""
        MERGE (g:Gene {{name: '{gene}'}})
        MERGE (p:Patient {{name: '{patient_name}'}})
        MERGE (p)-[:EXPRESSES {{value: {value}}}]->(g)
        """)


def predict_new_patient(model, new_patient, graph_data, node_map, threshold=0.25):

    # Check if patient exists in graph
    if new_patient not in node_map:
        return "Patient not found in graph"

    # Get node index
    patient_idx = node_map[new_patient]

    # Run model (NO rebuilding graph)
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)

    prob_high_risk = torch.softmax(out, dim=1)[:, 1]
    label = (prob_high_risk[patient_idx] > threshold).long().item()

    return "HIGH_RISK" if label == 1 else "LOW_RISK"
