from typing import Dict, Optional, Any
from newspaper import Article
import logging
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    logger.info("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {str(e)}")

class ArticleExtractor:
    """
    Utility class for extracting articles from URLs.
    
    This class provides functionality to extract and parse news articles from URLs
    while maintaining GDPR compliance by not storing any data locally.
    """
    
    def __init__(self):
        """Initialize the article extractor."""
        # Verify NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.warning("NLTK data not found, attempting to download...")
            nltk.download('punkt')
            nltk.download('punkt_tab')
    
    @staticmethod
    def extract_article(url: str) -> Dict[str, Optional[str]]:
        """Extract article content and metadata from URL."""
        try:
            # Validate URL format before processing
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError("Invalid URL format")
                
            # Initialize and download article
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract text and basic metadata first
            result = {
                'text': article.text,
                'title': article.title,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
            }
            
            # Try NLP features, but don't fail if they're not available
            try:
                article.nlp()
                result.update({
                    'keywords': article.keywords,
                    'summary': article.summary
                })
            except Exception as nlp_error:
                logger.warning(f"NLP features unavailable: {str(nlp_error)}")
                result.update({
                    'keywords': [],
                    'summary': result['text'][:500] + "..."  # Fallback to text truncation
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting article from {url}: {str(e)}")
            raise
            
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Validate URL format and accessibility.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL is valid and accessible, False otherwise
            
        Note:
            This method checks both URL format and server response
        """
        try:
            # Attempt to connect with timeout
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

class SentimentAnalyzer:
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        # Add your sentiment analysis logic here
        # For now, return a neutral sentiment
        return {"positive": 0.5, "negative": 0.5}