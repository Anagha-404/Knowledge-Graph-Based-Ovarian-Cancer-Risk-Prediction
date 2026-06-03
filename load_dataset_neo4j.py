from src.connect import run_query
import pandas as pd


df = pd.read_csv("converted_dataset.csv")

# 🔥 LIMIT DATA (VERY IMPORTANT)
df = df.head(100000)   # only 1 lakh rows
count = 0

for _, row in df.iterrows():
    patient = row["Patient"]
    gene = row["Gene"]
    value = float(row["Expression"])

    query = f"""
    MERGE (p:Patient {{name: '{patient}'}})
    MERGE (g:Gene {{name: '{gene}'}})
    MERGE (p)-[:EXPRESSES {{value: {value}}}]->(g)
    """

    run_query(query)

    count += 1
    if count % 1000 == 0:
        print(f"{count} rows inserted...")

print("✅ New dataset loaded into Neo4j!")