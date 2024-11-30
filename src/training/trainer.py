from typing import Dict, Any, Optional
from pathlib import Path
import logging
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)
from datasets import Dataset
import wandb
from ..data.data_loader import SummarizationDataset
import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizerTrainer:
    """
    Trainer class using HuggingFace's Trainer API for the summarization model.
    
    Handles:
    - Model training and validation
    - Automatic optimization
    - Checkpoint management
    - Progress logging
    - Mixed precision training
    """
    
    def __init__(self, config_path: Path) -> None:
        """
        Initialize the trainer with HuggingFace components.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA device available! Training will run on CPU, which may be significantly slower. "
                "Consider using a machine with GPU for faster training."
            )
        else:
            logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
            
        # Load config first
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
        # Extract config sections
        self.training_config = self.config['training']
        self.model_config = self.config['model']
        self.logging_config = self.config['logging']
    
        # Convert numeric values
        self.training_config['learning_rate'] = float(self.training_config['learning_rate'])
        self.training_config['max_grad_norm'] = float(self.training_config['max_grad_norm'])
            
        # Initialize components
        self.dataset = SummarizationDataset(config_path)
        
        # Initialize model and tokenizer using model config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_config['name'],
            generation_config=GenerationConfig(**self.config['generation'])
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['name']
        )
        
        # Setup wandb and training args
        self._setup_training_config()
        
    def _setup_training_config(self) -> None:
        """Setup training configuration and initialize wandb logging."""
        # Initialize wandb
        wandb.init(
            project=self.logging_config['wandb_project'],
            entity=self.logging_config['wandb_entity'],
            config=self.config
        )
        
        # Setup HuggingFace training arguments
        self.training_args = TrainingArguments(
            output_dir=self.training_config.get('output_dir', 'outputs'),
            num_train_epochs=self.training_config['num_epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config['batch_size'],
            warmup_steps=self.training_config['warmup_steps'],
            weight_decay=self.training_config.get('weight_decay', 0.01),
            logging_dir=self.logging_config.get('log_dir', 'logs'),
            logging_steps=self.logging_config['log_interval'],
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=self.logging_config['log_interval'],
            eval_steps=self.logging_config['log_interval'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="wandb",
            fp16=self.training_config['mixed_precision'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            max_grad_norm=self.training_config['max_grad_norm'],
            learning_rate=self.training_config['learning_rate']
        )
        
    def train(self) -> None:
        """
        Train the summarization model using HuggingFace Trainer.
        
        This method:
        - Prepares datasets
        - Initializes HuggingFace Trainer
        - Runs training
        - Handles logging and checkpointing automatically
        
        Raises:
            Exception: If training fails
        """
        try:
            # Load and prepare datasets
            train_dataset, val_dataset = self.dataset.load_dataset()
            
            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=self.tokenizer,
            )
            
            # Train the model
            trainer.train()
            
            # Save final model
            trainer.save_model(self.training_config['final_model_path'])
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    # ... rest of the implementation (training loop, validation, etc.) 