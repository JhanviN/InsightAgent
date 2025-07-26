from fastapi import FastAPI, Header, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import json
import os
from dotenv import load_dotenv
from query import analyze_query  # Your existing function

load_dotenv()
EXPECTED_TOKEN = os.getenv("AUTH_TOKEN")

# Initialize app and router
app = FastAPI()
router = APIRouter()

# Request schema
class QueryRequest(BaseModel):
    documents: str  # Currently unused
    questions: List[str]

# Root endpoint (no base prefix)
@app.get("/")
def read_root():
    return {"message": "InsightAgent backend is running!"}

# POST /api/v1/hackrx/run
@router.post("/hackrx/run")
async def run_query(
    request: QueryRequest,
    authorization: str = Header(...)
):

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid format. Expected 'Bearer <token>'.")

    token = authorization[len("Bearer "):].strip()

    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    answers = []
    for question in request.questions:
        result = analyze_query(question)

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                answers.append(parsed.get("justification", parsed))
            except json.JSONDecodeError:
                answers.append(result)
        else:
            answers.append(result.get("justification", result))

    return {"answers": answers}

app.include_router(router, prefix="/api/v1")
