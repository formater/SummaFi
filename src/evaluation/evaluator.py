from typing import Dict, List
import evaluate
import logging
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import torch
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizerEvaluator:
    """Evaluator class for the summarization model."""
    
    def __init__(self, config_path: str, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer):
        """Initialize the evaluator."""
        logger.info("Initializing evaluator...")
        
        # Check device and move model if needed
        if torch.cuda.is_available() and model.device.type != 'cuda':
            logger.info("Moving model to CUDA...")
            model = model.to('cuda')
        
        # Check final device
        device = model.device
        logger.info(f"Model is on device: {device}")
        if str(device) == 'cpu':
            logger.warning("Model is running on CPU. This might be slow!")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['evaluation']
            self.data_config = config['data']
            
        self.model = model
        self.tokenizer = tokenizer
        logger.info("Loading ROUGE metric...")
        self.rouge = evaluate.load('rouge')
        logger.info("Evaluator initialized.")
        
    def _get_generation_kwargs(self) -> Dict:
        """Get generation parameters."""
        return {
            'decoder_start_token_id': self.tokenizer.bos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'max_length': self.data_config['max_target_length'],
            'min_length': 56,
            'num_beams': 4,
            'length_penalty': 2.0,
            'early_stopping': True
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the test dataset."""
        try:
            # Set memory optimization
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # Load test dataset
            dataset = load_dataset(
                self.data_config['dataset_name'],
                self.data_config['dataset_version']
            )
            test_dataset = dataset['test']
            
            # Limit samples if specified
            if 'max_samples' in self.config:
                test_dataset = test_dataset.select(range(self.config['max_samples']))
            
            # Get predictions and references
            predictions = []
            references = []
            
            # Use smaller batch size for evaluation
            batch_size = 4  # Reduced from default
            
            # Get generation kwargs with reduced beam size
            generation_kwargs = self._get_generation_kwargs()
            generation_kwargs['num_beams'] = 2  # Override num_beams for memory efficiency
            
            # Process in batches
            for i in range(0, len(test_dataset), batch_size):
                try:
                    batch = test_dataset[i:i + batch_size]
                    
                    # Tokenize inputs
                    inputs = self.tokenizer(
                        batch['article'],
                        max_length=self.data_config['max_source_length'],
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.model.device)
                    
                    # Generate summaries
                    with torch.amp.autocast(device_type='cuda'):  # Updated autocast call
                        outputs = self.model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                    
                    # Decode predictions
                    batch_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    predictions.extend(batch_preds)
                    references.extend(batch['highlights'])
                    
                    # Clear cache after each batch
                    torch.cuda.empty_cache()
                    
                    if (i + batch_size) % 100 == 0:
                        logger.info(f"Processed {i + len(batch)}/{len(test_dataset)} examples")
                        
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM at batch {i}, trying to recover...")
                    torch.cuda.empty_cache()
                    continue
            
            # Compute ROUGE scores
            results = self.rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )
            
            return {k: v for k, v in results.items() if k in self.config['rouge_types']}
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def show_example(self, index: int = None) -> Dict[str, str]:
        """
        Show a comparison of original article, target summary, and model's summary.
        
        Args:
            index: Index of example to show. If None, picks a random example.
            
        Returns:
            Dictionary containing original, target, and generated summaries
        """
        try:
            # Load test dataset
            logger.info("Loading test dataset...")
            dataset = load_dataset(
                self.data_config['dataset_name'],
                self.data_config['dataset_version']
            )
            test_dataset = dataset['test']
            logger.info(f"Test dataset loaded. Size: {len(test_dataset)} examples")
            
            # Select example
            if index is None:
                import random
                index = random.randint(0, len(test_dataset) - 1)
            logger.info(f"Selected example index: {index}")
            
            example = test_dataset[index]
            
            # Get model's summary
            logger.info("Tokenizing input...")
            inputs = self.tokenizer(
                example['article'],
                max_length=self.data_config['max_source_length'],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.model.device)
            
            logger.info("Generating summary...")
            outputs = self.model.generate(**inputs, **self._get_generation_kwargs())
            model_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Summary generated.")
            
            # Prepare output
            comparison = {
                'article': example['article'],
                'target_summary': example['highlights'],
                'model_summary': model_summary
            }
            
            # Log the comparison
            logger.info("\n" + "="*100)
            logger.info("ORIGINAL ARTICLE:")
            logger.info(comparison['article'])
            logger.info("\n" + "-"*50)
            logger.info("TARGET SUMMARY:")
            logger.info(comparison['target_summary'])
            logger.info("\n" + "-"*50)
            logger.info("MODEL SUMMARY:")
            logger.info(comparison['model_summary'])
            logger.info("="*100)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error showing example: {str(e)}")
            raise

def evaluate_model(config_path: Path, model_path: Path):
    """Handle evaluation mode."""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        logger.info("Model and tokenizer loaded successfully")
        
        # Initialize evaluator
        evaluator = SummarizerEvaluator(config_path, model, tokenizer)
        
        # Run evaluation
        logger.info("\nRunning evaluation:")
        results = evaluator.evaluate()
        
        # Log results
        logger.info("\nEvaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise