from typing import Dict, Tuple, Any
import gradio as gr
import logging
from pathlib import Path
from ..models.summarizer import NewsSummarizer
from ..utils.article_extractor import ArticleExtractor
from ..utils.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaFiWeb:
    """
    Web interface for the SummaFi application.
    
    This class provides a Gradio-based web interface for:
    - Article URL input and validation
    - Text extraction and processing
    - Summary generation
    - Sentiment analysis
    - Result presentation
    
    Note:
        Implements GDPR compliance by not storing any article data
    """

    def __init__(self, config_path: Path, model_path: Path) -> None:
        """
        Initialize the web interface components.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to saved model checkpoint
            
        Raises:
            FileNotFoundError: If config or model file not found
            ValueError: If configuration is invalid
        """
        try:
            # Initialize model
            self.model = NewsSummarizer(config_path)

            # Load model checkpoint
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

            self.model.load_state_dict(model_path)

            # Initialize other components
            self.article_extractor = ArticleExtractor()
            self.sentiment_analyzer = SentimentAnalyzer()

            # Create Gradio interface
            self.interface = gr.Interface(
                fn=self.process_url,
                inputs=gr.Textbox(label="Article URL"),
                outputs=[
                    gr.Textbox(label="Summary", lines=5),
                    gr.Label(label="Sentiment Analysis", num_top_classes=3)
                ],
                title="SummaFi - Financial News Summarizer",
                description="Enter a financial news article URL to get a concise summary and sentiment analysis."
            )

        except Exception as e:
            logger.error(f"Error initializing web interface: {str(e)}")
            raise

    def process_url(self, url: str) -> Tuple[str, Dict[str, float]]:
        """
        Process article URL and return summary with sentiment.
        
        Args:
            url: URL of the article to process
            
        Returns:
            Tuple containing:
                - Generated summary
                - Sentiment analysis result
                
        Raises:
            ValueError: If URL is invalid
            Exception: For processing errors
        """
        try:
            # Extract article content
            article_data = self.article_extractor.extract_article(url)

            # Use article text or fallback to summary
            content = article_data.get('text', article_data.get('summary', ''))
            if not content:
                raise ValueError("Could not extract article content")

            # Generate summary
            summary = self.model.summarize(content)

            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze(content)

            # Just pass through the sentiment scores without modification
            return summary, sentiment

        except Exception as e:
            error_msg = f"Error processing URL: {str(e)}"
            logger.error(error_msg)
            return error_msg, {"error": 1.0}

    def launch(self, **kwargs: Any) -> None:
        """
        Launch the web interface.
        
        Args:
            **kwargs: Additional arguments passed to gradio launch
            
        Note:
            Handles interface creation and server configuration
        """
        self.interface.launch(**kwargs)
