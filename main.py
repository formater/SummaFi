import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="SummaFi - Financial News Summarizer")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "serve"],
        default="serve",
        help="Operation mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved model checkpoint"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web interface"
    )
    
    return parser.parse_args()

def train_model(config_path: Path):
    """Handle training mode."""
    from src.training.trainer import SummarizerTrainer
    
    logger.info("Starting training...")
    trainer = SummarizerTrainer(config_path)
    trainer.train()

def evaluate_model(config_path: Path, model_path: Path):
    """Handle evaluation mode."""
    from src.evaluation.evaluator import SummarizerEvaluator
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    logger.info(f"Loading model from checkpoint: {model_path}")
    
    # Load model and tokenizer from checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Initialize evaluator
    evaluator = SummarizerEvaluator(config_path, model, tokenizer)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Log results
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")

def serve_app(config_path: Path, model_path: Path, port: int):
    """Handle serve mode."""
    from src.web.app import SummaFiWeb
    
    logger.info("Starting web interface...")
    app = SummaFiWeb(config_path, model_path)
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )

def main():
    """Main application entry point."""
    args = setup_args()
    config_path = Path(args.config)
    
    try:
        if args.mode == "train":
            train_model(config_path)
            
        elif args.mode == "evaluate":
            if not args.model_path:
                raise ValueError("Model path required for evaluation")
            evaluate_model(config_path, Path(args.model_path))
            
        else:  # serve mode
            serve_app(config_path, args.model_path, args.port)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 