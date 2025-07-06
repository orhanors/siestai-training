import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

def load_neo4j_graph(env_path: str = '.env') -> Neo4jGraph:
    # Load from environment
    load_dotenv(env_path, override=True)
    
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
    
    # Optional: OpenAI config if needed elsewhere
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings' if os.getenv('OPENAI_BASE_URL') else None

    # Initialize Neo4j graph object
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
    
    return graph, OPENAI_API_KEY, OPENAI_ENDPOINT