from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from query import analyze_query  # Ensure this function exists and works

app = FastAPI()

# Root route (for testing if backend is live)
@app.get("/")
def read_root():
    return {"message": "InsightAgent backend is running!"}

# Request model for /run endpoint
class QueryRequest(BaseModel):
    documents: str  # Currently unused but can be extended
    questions: List[str]

# Endpoint to run the query
@app.post("/api/v1/hackrx/run")
async def run_query(request: QueryRequest):
    responses = []
    for question in request.questions:
        result = analyze_query(question)  # This should return a string or dict
        responses.append(result)
    return {"answers": responses}
