from typing import Dict, List
import torch
import evaluate
import logging
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizerEvaluator:
    """Evaluator class for the summarization model."""
    
    def __init__(self, config_path: str, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer):
        """
        Initialize the evaluator.
        
        Args:
            config_path: Path to the configuration file
            model: Loaded model for evaluation
            tokenizer: Tokenizer for the model
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['evaluation']
            self.data_config = config['data']
            
        self.model = model
        self.tokenizer = tokenizer
        self.rouge = evaluate.load('rouge')
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def compute_metrics(
        self,
        generated_summaries: List[str],
        reference_summaries: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for generated summaries.
        
        Args:
            generated_summaries: List of generated summaries
            reference_summaries: List of reference summaries
            
        Returns:
            Dictionary containing ROUGE scores
        """
        try:
            results = self.rouge.compute(
                predictions=generated_summaries,
                references=reference_summaries,
                use_stemmer=True
            )
            
            # Extract scores for specified ROUGE types
            scores = {}
            for rouge_type in self.config['rouge_types']:
                scores[f"{rouge_type}"] = results[rouge_type]
                
            return scores
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise
            
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the test dataset.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load test dataset
        dataset = load_dataset(
            self.data_config['dataset_name'],
            self.data_config['dataset_version']
        )
        test_dataset = dataset['test']
        
        # Limit samples if specified
        if 'max_samples' in self.config:
            test_dataset = test_dataset.select(range(self.config['max_samples']))
        
        generated_summaries = []
        reference_summaries = []
        
        # Process in batches
        batch_size = self.config['batch_size']
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i + batch_size]
            
            # Generate summaries
            inputs = self.tokenizer(
                batch['article'],
                max_length=self.data_config['max_source_length'],
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    decoder_start_token_id=self.tokenizer.bos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.data_config['max_target_length'],
                    min_length=self.config.get('min_length', 56),
                    num_beams=self.config.get('num_beams', 4),
                    length_penalty=self.config.get('length_penalty', 2.0),
                    early_stopping=True
                )
            
            # Decode generated summaries
            decoded_summaries = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            generated_summaries.extend(decoded_summaries)
            reference_summaries.extend(batch['highlights'])
            
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {i + batch_size}/{len(test_dataset)} examples")
        
        # Compute metrics
        return self.compute_metrics(generated_summaries, reference_summaries)