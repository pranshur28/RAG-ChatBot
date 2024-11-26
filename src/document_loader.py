import os
import logging
from typing import List, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document loader with chunking parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Document loader initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def load_document(self, file_path: str) -> List[str]:
        """Load and chunk a document based on file type"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return []

            file_extension = os.path.splitext(file_path)[1].lower()
            logger.info(f"Loading document: {file_path}")

            if file_extension == '.csv':
                return self._process_csv(file_path)
            elif file_extension == '.txt':
                return self._process_text(file_path)
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return []

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []

    def _process_csv(self, file_path: str) -> List[str]:
        """Process CSV file into chunks"""
        try:
            logger.info(f"Processing CSV file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Convert each row to a string representation
            chunks = []
            for i in range(0, len(df), self.chunk_size):
                chunk_df = df.iloc[i:i + self.chunk_size]
                chunk_text = chunk_df.to_string(index=False)
                chunks.append(chunk_text)
            
            logger.info(f"Created {len(chunks)} chunks from CSV")
            return chunks

        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {str(e)}")
            return []

    def _process_text(self, file_path: str) -> List[str]:
        """Process text file into chunks"""
        try:
            logger.info(f"Processing text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split into chunks
            chunks = []
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                chunks.append(chunk)

            logger.info(f"Created {len(chunks)} chunks from text file")
            return chunks

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
