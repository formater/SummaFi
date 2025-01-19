from typing import Union
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSummarizer:
    """
    News summarization model based on BART.
    
    This class implements:
    - Model initialization and configuration
    - Text summarization
    - Model state management
    - GPU acceleration when available
    """

    def __init__(self, config_path: Union[str, Path]):
        """Initialize the summarizer model."""
        logger.info("Initializing summarizer model...")

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.model_config = config['model']
            self.generation_config = config['generation']

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['name']
        )

        # Initialize model
        self.model = None  # Will be loaded later
        logger.info("Model initialized successfully")

    def summarize(self, text: str) -> str:
        """
        Generate a summary for the input text.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Generated summary text
            
        Raises:
            ValueError: If input text is empty or too short
            Exception: If summarization fails
        """
        try:
            # Validate input
            if not text or len(text.split()) < 10:
                raise ValueError("Input text too short for summarization")

            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=self.generation_config['max_length'],
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=self.generation_config['num_beams'],
                    min_length=self.generation_config['min_length'],
                    max_length=self.generation_config['max_length'],
                    length_penalty=self.generation_config['length_penalty'],
                    early_stopping=self.generation_config['early_stopping'],
                    decoder_start_token_id=self.tokenizer.bos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    def load_state_dict(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model from checkpoint."""
        try:
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")

            # Load model using HuggingFace's from_pretrained
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path,
                local_files_only=True,
                device_map='auto'  # Automatically handle device placement
            )
            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise RuntimeError("Failed to load model checkpoint") from e
