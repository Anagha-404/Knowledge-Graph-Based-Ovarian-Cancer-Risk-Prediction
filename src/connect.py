from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"  # change this

driver = GraphDatabase.driver(uri, auth=(username, password))

def run_query(query):
    with driver.session(database="gene") as session:   
        result = session.run(query)
        return [record.data() for record in result]