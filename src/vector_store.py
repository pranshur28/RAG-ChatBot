import os
import uuid
import logging
import chromadb
from chromadb.config import Settings
from typing import List, Optional
import openai
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreHandler:
    def __init__(self, api_key: str):
        """Initialize vector store handler"""
        try:
            logger.info("Initializing in-memory vector store")
            self.openai_client = openai.OpenAI(api_key=api_key)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Initialize collection as None
            self.collection = None
            logger.info("Vector store handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def create_or_load_store(self, collection_name: str = "trading_docs"):
        """Create or load a ChromaDB collection"""
        try:
            logger.info(f"Creating collection: {collection_name}")
            
            # Get or create collection
            try:
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Successfully accessed collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Error accessing collection: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to create/load collection: {str(e)}")
            raise

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add to vector store")
                return

            if not self.collection:
                logger.error("Collection not initialized")
                raise ValueError("Collection not initialized. Call create_or_load_store first.")

            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Generate embeddings in smaller batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    # Generate embeddings
                    response = self.openai_client.embeddings.create(
                        input=batch,
                        model="text-embedding-ada-002"
                    )
                    embeddings = [r.embedding for r in response.data]
                    logger.info(f"Generated embeddings for batch of {len(batch)} documents")
                    
                    # Convert embeddings to the correct format
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    
                    # Add to ChromaDB
                    ids = [str(uuid.uuid4()) for _ in batch]
                    metadatas = [{"source": f"document_{idx}"} for idx in range(len(batch))]
                    
                    self.collection.add(
                        embeddings=embeddings_array.tolist(),
                        documents=batch,
                        ids=ids,
                        metadatas=metadatas
                    )
                    logger.info(f"Successfully added batch of {len(batch)} documents to collection")
                    
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise

    def query(self, query_text: str, n_results: int = 5) -> List[str]:
        """Query the vector store for relevant documents"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                raise ValueError("Collection not initialized. Call create_or_load_store first.")

            # Generate query embedding
            response = self.openai_client.embeddings.create(
                input=query_text,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding

            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Extract and return documents
            if results and 'documents' in results and results['documents']:
                return results['documents'][0]  # First list contains matches for first query
            return []

        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []
