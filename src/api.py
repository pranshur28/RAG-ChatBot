import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from .rag_pipeline import RAGPipeline
from .csv_analyzer import CSVAnalyzer
import tempfile

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Market Analysis RAG Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
PDF_PATH = os.getenv("PDF_PATH", "C:/Users/prans/Downloads/taylortradingtechnique_compress (1).pdf")
rag_pipeline = RAGPipeline(PDF_PATH)
csv_analyzer = CSVAnalyzer()

class MarketAnalysisRequest(BaseModel):
    symbol: str

class InsightRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "Market Analysis RAG Chatbot API"}

@app.post("/analyze")
async def analyze_market(request: MarketAnalysisRequest):
    """Analyze market data for a given symbol"""
    try:
        result = rag_pipeline.analyze_market(request.symbol)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insight")
async def get_insight(request: InsightRequest):
    """Get specific insight from the trading book"""
    try:
        result = rag_pipeline.get_trading_book_insight(request.query)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """Analyze uploaded CSV file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Load and analyze the CSV
        if not csv_analyzer.load_csv(temp_path):
            raise HTTPException(status_code=400, detail="Failed to load CSV file")

        # Generate insights
        insights = csv_analyzer.generate_insights()

        # Clean up temporary file
        os.unlink(temp_path)

        return insights

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
