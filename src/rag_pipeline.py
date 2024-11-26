import os
import logging
from typing import Dict, Any, List, Optional
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, document_loader, vector_store):
        """Initialize RAG pipeline with document loader and vector store"""
        try:
            self.document_loader = document_loader
            self.vector_store = vector_store
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise

    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents and add to vector store"""
        try:
            if not file_paths:
                return {"success": False, "error": "No files provided"}

            for file_path in file_paths:
                try:
                    # Load and chunk document
                    chunks = self.document_loader.load_document(file_path)
                    if not chunks:
                        return {"success": False, "error": f"No content extracted from {file_path}"}

                    # Add chunks to vector store
                    self.vector_store.add_documents(chunks)

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    return {"success": False, "error": f"Failed to process {os.path.basename(file_path)}: {str(e)}"}

            return {"success": True}

        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            return {"success": False, "error": str(e)}

    def query(self, query_text: str) -> str:
        """Process a query and return response"""
        try:
            # Get relevant documents from vector store
            relevant_docs = self.vector_store.query(query_text)
            
            if not relevant_docs:
                return "I couldn't find any relevant information in the documents to answer your question."
            
            # Combine documents into context
            context = "\n\n".join(relevant_docs)
            
            # Create chat completion with context
            messages = [
                {"role": "system", "content": "You are a helpful assistant analyzing trading data. Use the provided context to answer questions accurately and concisely."},
                {"role": "user", "content": f"Using this trading data context:\n\n{context}\n\nAnswer this question: {query_text}"}
            ]
            
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Sorry, I encountered an error while processing your query: {str(e)}"
