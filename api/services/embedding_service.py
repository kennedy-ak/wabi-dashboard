"""OpenAI Embeddings Service for generating text embeddings."""

import logging
import openai
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating OpenAI text embeddings."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"  # More cost-effective embedding model
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return None
            
            # Clean and truncate text if too long (embedding models have token limits)
            cleaned_text = text.strip()[:8000]  # Conservative limit
            
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_text
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in a batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (or None for failed items)
        """
        try:
            if not texts:
                return []
            
            # Clean and prepare texts
            cleaned_texts = []
            for text in texts:
                if text and text.strip():
                    cleaned_texts.append(text.strip()[:8000])
                else:
                    cleaned_texts.append("")
            
            # Filter out empty texts for API call
            non_empty_indices = []
            non_empty_texts = []
            for i, text in enumerate(cleaned_texts):
                if text:
                    non_empty_indices.append(i)
                    non_empty_texts.append(text)
            
            if not non_empty_texts:
                logger.warning("No valid texts provided for batch embedding")
                return [None] * len(texts)
            
            # Generate embeddings for non-empty texts
            response = self.client.embeddings.create(
                model=self.model,
                input=non_empty_texts
            )
            
            # Map results back to original positions
            results = [None] * len(texts)
            for i, embedding_data in enumerate(response.data):
                original_index = non_empty_indices[i]
                results[original_index] = embedding_data.embedding
            
            logger.info(f"Generated {len(non_empty_texts)} embeddings in batch")
            return results
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [None] * len(texts)
