from fastapi import FastAPI
from router.search_router import search_router

app = FastAPI(
    title="SNRT Semantic Search API",
    description="A semantic search API powered by Milvus and Groq",
    version="1.0.0"
)

# Include the search router without prefix
app.include_router(search_router)

@app.get("/")
async def root():
    return {"message": "SNRT Semantic Search API"} 