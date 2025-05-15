from fastapi import APIRouter
from pydantic import BaseModel
from services.embedder import embed_query
from services.intent_classifier import classify_intent
from services.milvus_query import search_content
from services.formatter import format_response

# Create router instance
search_router = APIRouter(tags=["search"])

# Define the search query model
class SearchQuery(BaseModel):
    query: str

@search_router.post("/search")
async def search(query_model: SearchQuery):
    """
    Endpoint for semantic search queries.
    Processes the query through intent classification, embedding, and Milvus search.
    Returns formatted results based on the search intent and content matches.
    """
    # 1. Classify intent
    intent, confidence, intent_description = classify_intent(query_model.query)
    
    # 2. Embed the query
    query_embedding = embed_query(query_model.query)
    
    # 3. Search content in Milvus
    search_results = search_content(query_embedding=query_embedding, top_k=3)
    
    # 4. Format the final response
    response = format_response(
        intent=intent,
        confidence=confidence,
        intent_description=intent_description,
        search_results=search_results,
        query=query_model.query
    )
    
    return response 