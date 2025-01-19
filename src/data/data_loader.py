from typing import Dict, Tuple, List, Union
from datasets import load_dataset, Dataset
from transformers import BartTokenizer
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationDataset:
    """
    Dataset class for handling CNN/DailyMail summarization data using HuggingFace datasets.
    
    This class handles:
    - Dataset loading and preprocessing
    - Tokenization for both inputs and targets
    - Dynamic batching with padding
    - PyTorch format conversion
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """
        Initialize the dataset handler.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If required config keys are missing
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['data']
            self.model_name = config['model']['name']
            self.max_source_length = config['data']['max_source_length']
            self.max_target_length = config['data']['max_target_length']

        # Initialize tokenizer with proper settings
        self.tokenizer = BartTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.max_source_length
        )

    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Load and prepare the CNN/DailyMail dataset.
        
        Returns:
            Tuple containing:
                - Training dataset (Dataset)
                - Validation dataset (Dataset)
                
        Raises:
            Exception: If dataset loading or processing fails
        """
        try:
            logger.info("Loading CNN/DailyMail dataset...")
            dataset = load_dataset(
                self.config['dataset_name'],
                self.config['dataset_version']
            )

            # Handle dataset splitting
            if 'validation' in dataset:
                train_dataset = dataset['train']
                val_dataset = dataset['validation']
            else:
                train_val = dataset['train'].train_test_split(
                    test_size=self.config['val_size']
                )
                train_dataset = train_val['train']
                val_dataset = train_val['test']

            # Process datasets
            train_dataset = self._process_dataset(
                train_dataset,
                desc="Processing training data"
            )
            val_dataset = self._process_dataset(
                val_dataset,
                desc="Processing validation data"
            )

            logger.info(f"Processed {len(train_dataset)} training examples")
            logger.info(f"Processed {len(val_dataset)} validation examples")

            return train_dataset, val_dataset

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _process_dataset(
            self,
            dataset: Dataset,
            desc: str = "Processing dataset"
    ) -> Dataset:
        """
        Process a dataset with tokenization and formatting.
        
        Args:
            dataset: Input dataset to process
            desc: Description for progress bar
            
        Returns:
            Processed dataset ready for training
        """
        # Apply preprocessing
        processed = dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=desc
        )

        # Set format for PyTorch
        processed.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

        return processed

    def _preprocess_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """
        Preprocess examples by tokenizing inputs and targets.
        
        Args:
            examples: Dictionary containing articles and highlights
            
        Returns:
            Dictionary containing tokenized inputs and labels
            
        Note:
            Special tokens are handled automatically by the tokenizer
        """
        # Tokenize inputs and targets in a single call
        model_inputs = self.tokenizer(
            examples["article"],
            text_target=examples["highlights"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize targets separately for proper length
        labels = self.tokenizer(
            examples["highlights"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        # Replace padding token id in labels with -100
        # Explanation: When sequences are tokenized, they are padded to a fixed length (max_target_length) for
        # uniformity in batch processing. These padding tokens do not carry any meaningful information and should not
        # contribute to the loss computation during training. In  many frameworks (e.g., PyTorch's CrossEntropyLoss),
        # setting a target token value to -100 ensures that it is ignored during loss computation.
        for i in range(len(model_inputs["labels"])):
            model_inputs["labels"][i] = [
                -100 if token == self.tokenizer.pad_token_id else token
                for token in model_inputs["labels"][i]
            ]

        return model_inputs

    def get_tokenizer(self) -> BartTokenizer:
        """
        Get the tokenizer instance.
        
        Returns:
            The initialized BartTokenizer
        """
        return self.tokenizer
