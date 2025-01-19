import pytest
from pathlib import Path
from src.data.data_loader import SummarizationDataset


@pytest.fixture
def config_path():
    return Path("config/config.yaml")


def test_dataset_initialization(config_path):
    """Test dataset initialization."""
    dataset = SummarizationDataset(config_path)
    assert dataset.tokenizer is not None
    assert dataset.max_source_length > 0
    assert dataset.max_target_length > 0


def test_dataset_loading(config_path):
    """Test dataset loading."""
    dataset = SummarizationDataset(config_path)
    train_dataset, val_dataset = dataset.load_dataset()

    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    # Check required features
    required_features = ['input_ids', 'attention_mask', 'labels']
    for feature in required_features:
        assert feature in train_dataset.features
        assert feature in val_dataset.features
