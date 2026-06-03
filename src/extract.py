import pandas as pd
from src.connect import run_query


def get_expression_data():
    # Read directly from CSV — has all 285 patients
    # Neo4j only has partial data due to the 100k row limit in load_dataset_neo4j.py
    df = pd.read_csv("converted_dataset.csv")
    df.columns = ["patient", "gene", "expression"]  # normalize column names
    return df


def get_labels():
    return pd.read_csv("labels.csv")