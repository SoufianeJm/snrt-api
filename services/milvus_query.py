from typing import List, Dict
from pymilvus import connections, Collection

# Milvus configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "content_items"
VECTOR_FIELD_NAME = "embedding"

def search_content(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    """
    Search Milvus collection for similar content using the query embedding.
    
    Args:
        query_embedding (List[float]): The embedding vector of the query
        top_k (int): Number of results to return (default: 3)
        
    Returns:
        List[Dict]: List of search results, each containing the document fields
    """
    # Connect to Milvus
    connections.connect(
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    
    # Get collection and load it
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # Define search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }
    
    # Perform search
    results = collection.search(
        data=[query_embedding],
        anns_field=VECTOR_FIELD_NAME,
        param=search_params,
        limit=top_k,
        output_fields=['id', 'type', 'title', 'description', 'date', 'time', 'extra'],
        consistency_level="Strong"
    )
    
    # Process results
    search_results = []
    for hits in results:
        for hit in hits:
            result = {
                'id': hit.id,
                'score': hit.score,
                **{field: hit.entity.get(field) for field in ['type', 'title', 'description', 'date', 'time', 'extra']}
            }
            search_results.append(result)
    
    return search_results 