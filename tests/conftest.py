import pytest
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Disable wandb for testing
os.environ["WANDB_MODE"] = "disabled"


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring trained model"
    )


@pytest.fixture
def check_model():
    """Check if trained model exists."""
    model_path = Path("outputs/final_model")
    if not model_path.exists():
        pytest.skip(
            "\nTrained model not found at outputs/final_model. "
            "Please either:\n"
            "1. Train the model using: python main.py --mode train\n"
            "2. Download the pre-trained model\n"
        )
