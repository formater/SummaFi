from transformers import pipeline
import torch
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer."""
        # Setup device
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Sentiment analyzer using device: {'cuda' if self.device == 0 else 'cpu'}")
        
        # Initialize sentiment pipeline
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=self.device,
            top_k=None  # Return all scores
        )
        
        # Get model's max length
        self.max_length = self.analyzer.model.config.max_position_embeddings
        logger.info(f"Maximum sequence length: {self.max_length}")
        
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        try:
            # Split text into chunks if needed
            chunks = self._split_text(text)
            
            # Initialize scores
            raw_scores = {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0
            }
            
            # Analyze each chunk
            chunk_count = 0
            for chunk in chunks:
                # Get sentiment predictions
                predictions = self.analyzer(chunk, truncation=True, max_length=self.max_length)
                
                # Each prediction is a list of all sentiment scores
                for sentiment_scores in predictions:
                    chunk_count += 1
                    # Add up scores for each sentiment
                    for score_dict in sentiment_scores:
                        label = score_dict['label'].lower()
                        score = float(score_dict['score'])
                        raw_scores[label] += score
            
            # If no chunks were processed, return default values
            if chunk_count == 0:
                return {"positive": 0.3333, "negative": 0.3333, "neutral": 0.3334}
            
            # Average the scores across chunks
            averaged_scores = {
                label: score / chunk_count 
                for label, score in raw_scores.items()
            }
            
            # Normalize to probabilities (0-1) and round
            total = sum(averaged_scores.values())
            return {
                label: round(score / total, 4)
                for label, score in averaged_scores.items()
            }
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            logger.error(f"Raw scores were: {raw_scores}")
            return {"error": 1.0}
            
    def _split_text(self, text: str, overlap: int = 50) -> list:
        """Split text into overlapping chunks."""
        words = text.split()
        if len(words) <= self.max_length:
            return [text]
            
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.max_length
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap  # Create overlap between chunks
            
        return chunks