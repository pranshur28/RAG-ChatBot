import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.api import app
from src.rag_pipeline import RAGPipeline
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize RAG pipeline
pdf_path = os.getenv("PDF_PATH")
rag_pipeline = RAGPipeline(pdf_path)

class Query(BaseModel):
    query: str

class MarketQuery(BaseModel):
    symbol: str

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

@app.post("/query")
async def query_endpoint(query: Query):
    try:
        response = rag_pipeline.get_trading_insight(query.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-market")
async def analyze_market(query: MarketQuery):
    try:
        analysis = rag_pipeline.analyze_market(query.symbol)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Analyze CSV
        analysis = rag_pipeline.analyze_csv(str(temp_path))
        
        # Clean up
        os.remove(temp_path)
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Run the FastAPI application
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
