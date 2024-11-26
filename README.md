# Market Analysis RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot that analyzes market data using insights from a trading book. The system combines real-time market data with knowledge extracted from the provided trading book to generate informed market analysis and trading insights.

## Features

- PDF document processing and knowledge extraction
- Real-time market data fetching using yfinance
- Vector store for efficient knowledge retrieval
- RAG-based market analysis combining book knowledge with current market data
- FastAPI-based REST API for easy integration

## Prerequisites

- Python 3.8+
- OpenAI API key
- Trading book PDF file

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd market-analysis-rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
```

5. Configure your environment variables in `.env`:
- Add your OpenAI API key
- Set the path to your trading book PDF
- Adjust other configurations as needed

## Usage

1. Start the API server:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Market Analysis
```http
POST /analyze
```

Analyze market data for a specific symbol using insights from the trading book.

Request body:
```json
{
    "symbol": "AAPL"
}
```

### 2. Trading Book Insights
```http
POST /insight
```

Get specific insights from the trading book.

Request body:
```json
{
    "query": "What are the key principles of trend following?"
}
```

## Example Usage

### Python
```python
import requests

# Market Analysis
response = requests.post(
    "http://localhost:8000/analyze",
    json={"symbol": "AAPL"}
)
print(response.json())

# Trading Book Insight
response = requests.post(
    "http://localhost:8000/insight",
    json={"query": "How to identify market trends?"}
)
print(response.json())
```

### cURL
```bash
# Market Analysis
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"symbol":"AAPL"}'

# Trading Book Insight
curl -X POST "http://localhost:8000/insight" \
     -H "Content-Type: application/json" \
     -d '{"query":"How to identify market trends?"}'
```

## Project Structure

```
market-analysis-rag-chatbot/
├── src/
│   ├── __init__.py
│   ├── api.py              # FastAPI application
│   ├── document_loader.py  # PDF processing
│   ├── market_data.py      # Market data handling
│   ├── rag_pipeline.py     # Core RAG implementation
│   └── vector_store.py     # Vector store management
├── .env.example
├── main.py
├── requirements.txt
└── README.md
```

## Configuration

The system can be configured through environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PDF_PATH`: Path to the trading book PDF
- `HOST`: API host (default: 0.0.0.0)
- `PORT`: API port (default: 8000)

## Notes

- The system uses OpenAI's embeddings for document vectorization
- Market data is fetched in real-time using yfinance
- The vector store persists embeddings to disk for faster subsequent loads
- The system combines both historical trading knowledge and current market data for analysis

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 200: Successful operation
- 400: Bad request (e.g., invalid symbol)
- 500: Internal server error

Error responses include a detail message explaining the error.
