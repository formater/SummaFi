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
        choices=["train", "evaluate", "serve", "example"],
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
    parser.add_argument(
        "--example-index",
        type=int,
        default=None,
        help="Index of example to show (random if not specified)"
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
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.evaluation.evaluator import SummarizerEvaluator

    logger.info(f"Loading model from checkpoint: {model_path}")

    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        # Initialize evaluator
        evaluator = SummarizerEvaluator(config_path, model, tokenizer)

        # Run evaluation
        results = evaluator.evaluate()

        # Log results
        logger.info("\nEvaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def show_example(config_path: Path, model_path: Path, index: int = None):
    """Handle example mode."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.evaluation.evaluator import SummarizerEvaluator

    logger.info(f"Loading model from checkpoint: {model_path}")

    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        # Initialize evaluator
        evaluator = SummarizerEvaluator(config_path, model, tokenizer)

        # Show example
        evaluator.show_example(index)

    except Exception as e:
        logger.error(f"Error showing example: {str(e)}")
        raise


def serve_app(config_path: Path, model_path: str, port: int):
    """Handle serve mode."""
    from src.web.app import SummaFiWeb

    try:
        # Convert model_path to Path and verify it exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        logger.info("Starting web interface...")
        app = SummaFiWeb(config_path, model_path)
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False
        )

    except Exception as e:
        logger.error(f"Error starting web interface: {str(e)}")
        raise


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

        elif args.mode == "example":
            if not args.model_path:
                raise ValueError("Model path required for showing example")
            show_example(config_path, Path(args.model_path), args.example_index)

        else:  # serve mode
            if not args.model_path:
                raise ValueError("Model path required for serving")
            serve_app(config_path, Path(args.model_path), args.port)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
