import re
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from services.embedder import embed_query
from services.intent_classifier import classify_intent
from services.milvus_query import search_content
from services.formatter import format_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # 3. Build filter expression based on intent and query content
    filter_expr: Optional[str] = None
    
    # Extract year from query if present
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', query_model.query)
    extracted_year = year_match.group(0) if year_match else None
    
    # Apply filter rules based on intent
    if intent == "match_score" and extracted_year:
        filter_conditions = []
        # Always add the date condition
        filter_conditions.append(f'date LIKE "{extracted_year}%"')
        
        # Define content type keywords
        video_keywords = [
            "video", "vidéo", "revoir", "rediffusion", "youtube",
            "replay", "résumé", "résumé", "highlights", "extraits",
            "montage", "buts", "but", "buts", "but", "match complet"
        ]
        
        article_keywords = [
            "article", "lire", "nouvelle", "analyse", "commentaire",
            "reportage", "dossier", "interview", "entretien"
        ]
        
        # Check for content type keywords in the query
        query_lower = query_model.query.lower()
        
        # Check for video-related terms
        if any(keyword in query_lower for keyword in video_keywords):
            filter_conditions.append('type == "video"')
            logger.info(f"Video type detected in query: '{query_model.query}'")
        # Check for article-related terms
        elif any(keyword in query_lower for keyword in article_keywords):
            filter_conditions.append('type == "article"')
            logger.info(f"Article type detected in query: '{query_model.query}'")
        else:
            # Default to 'match' if no specific type is mentioned
            filter_conditions.append('type == "match"')
            logger.info(f"No specific content type detected, defaulting to 'match' for query: '{query_model.query}'")
        
        # Join all conditions with AND
        filter_expr = " AND ".join(filter_conditions)
        logger.info(f"Query: '{query_model.query}', Intent: '{intent}', Applying filter_expr: '{filter_expr}'")
    
    # 4. Search content in Milvus with filter
    search_results = search_content(
        query_embedding=query_embedding,
        top_k=3,
        filter_expr=filter_expr
    )
    
    # 5. Format the final response
    response = format_response(
        intent=intent,
        confidence=confidence,
        intent_description=intent_description,
        search_results=search_results,
        query=query_model.query
    )
    
    return response 