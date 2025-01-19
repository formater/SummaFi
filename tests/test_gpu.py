import pytest
import torch
from src.models.summarizer import NewsSummarizer
from pathlib import Path


@pytest.mark.gpu
def test_gpu_availability():
    """Test if CUDA is available."""
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() > 0, "No GPU devices found"


@pytest.mark.gpu
def test_model_to_gpu():
    """Test if model moves to GPU correctly."""
    config_path = Path("config/config.yaml")
    model = NewsSummarizer(config_path)

    # Load model
    model.load_state_dict("outputs/final_model")

    # Check if model is on GPU
    assert next(model.model.parameters()).device.type == "cuda", "Model not on GPU"

    # Check if model can process data on GPU
    test_text = """
    In a significant development for the technology sector, major companies announced 
    record-breaking quarterly earnings. The strong performance was driven by increased 
    cloud computing demand, artificial intelligence advancements, and robust consumer 
    spending in digital services. Analysts predict this trend will continue through 
    the next fiscal year, with particular growth expected in emerging markets.
    """
    summary = model.summarize(test_text)
    assert isinstance(summary, str), "Summary generation failed"
    assert len(summary) > 0, "Generated summary is empty"


@pytest.mark.gpu
def test_gpu_memory_usage():
    """Test GPU memory usage during processing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Record initial GPU memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    # Initialize model
    config_path = Path("config/config.yaml")
    model = NewsSummarizer(config_path)
    model.load_state_dict("outputs/final_model")

    # Record memory after model load
    post_load_memory = torch.cuda.memory_allocated()
    assert post_load_memory > initial_memory, "Model not loaded to GPU memory"

    # Generate summary
    test_text = "This is a test article for GPU memory usage monitoring."
    summary = model.summarize(test_text)

    # Record memory during processing
    processing_memory = torch.cuda.memory_allocated()
    assert processing_memory >= post_load_memory, "No additional memory used during processing"

    # Check memory cleanup
    del model
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated()
    assert final_memory < processing_memory, "Memory not properly freed"


@pytest.mark.gpu
def test_batch_processing_on_gpu():
    """Test batch processing on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config_path = Path("config/config.yaml")
    model = NewsSummarizer(config_path)
    model.load_state_dict("outputs/final_model")

    # Create test batch with longer texts
    test_texts = [
        """The technology sector saw unprecedented growth in Q4 2023, with major 
        companies reporting record profits. Cloud computing and AI initiatives led 
        the surge, while consumer electronics sales remained strong despite 
        economic headwinds.""",

        """Financial markets responded positively to the Federal Reserve's latest 
        policy announcement. Interest rates remained stable, while the Fed signaled 
        a potential shift in strategy for the coming year. Analysts expect this to 
        boost market confidence.""",

        """Renewable energy investments reached new heights as governments worldwide 
        accelerated their climate commitments. Solar and wind projects dominated 
        the landscape, while emerging technologies like green hydrogen gained 
        significant attention from investors."""
    ]

    # Process batch
    try:
        summaries = []
        for text in test_texts:
            summary = model.summarize(text)
            summaries.append(summary)

        assert len(summaries) == len(test_texts), "Not all texts processed"
        assert all(isinstance(s, str) for s in summaries), "Invalid summary format"
        assert all(len(s) > 0 for s in summaries), "Empty summary generated"

    except Exception as e:
        pytest.fail(f"Batch processing failed: {str(e)}")
