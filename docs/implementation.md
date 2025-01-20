# SummaFi Implementation Document

## 1. Architectural Overview

### 1.1 System Components
- **Core Model**: Facebook's BART model fine-tuned on CNN/DailyMail dataset
- **Data Processing Pipeline**: Custom dataset loader and preprocessor
- **Web Interface**: Gradio-based user interface
- **Utility Components**: Article extractor and sentiment analyzer
- **Testing Framework**: Comprehensive pytest suite

### 1.2 Key Dependencies
From requirements.txt:
```
wandb>=0.18.7        # ML experiment tracking
transformers>=4.46.3 # Transformer models
datasets>=3.1.0      # Dataset handling
PyYAML>=6.0.2        # Configuration management
torch>=2.5.1         # Deep learning framework
evaluate>=0.4.3      # Model evaluation
gradio>=5.7.1        # Web interface
newspaper3k>=0.2.8   # Article extraction
accelerate>=1.1.1    # Model acceleration
rouge_score>=0.1.2   # Summary evaluation
pytest>=8.3.4        # Testing framework
```

## 2. Code Documentation
Code is well-documented, with docstrings and comments. 
### 2.1 Core Components

#### Summarizer Model
```python
class NewsSummarizer:
    """Main summarization model wrapper."""
    # This class implements:
    # - Model initialization and configuration
    # - Text summarization
    # - Model state management
    # - GPU acceleration when available
    
```

#### Data Processing
```python
class SummarizationDataset:
    """Dataset loader and preprocessor."""
    # Features:
    # - Handles CNN/DailyMail dataset
    # - Implements data preprocessing
    # - Manages tokenization
```

#### Article Extractor
```python
class ArticleExtractor:
    """Extracts articles from URLs."""
    # Features:
    # - URL validation
    # - Content extraction
    # - Error handling
```

#### Sentiment Analyzer
```python
class SentimentAnalyzer:
    """Financial sentiment analysis."""
    # Features:
    # - GPU support (if available)
    # - Text chunking for long articles
    # - Probability distribution output
```

### 2.2 Testing Infrastructure

#### Test Categories
1. **Unit Tests**
   - Data loading/preprocessing
   - Model initialization
   - Article extraction
   - URL validation

2. **GPU Tests**
   - Memory management
   - Batch processing
   - Model loading
   - CUDA availability

3. **Integration Tests**
   - End-to-end processing
   - Component interaction

#### Test Markers
From pytest.ini:
```ini
markers =
    requires_model: mark test as requiring trained model
    gpu: mark test as requiring GPU
    slow: marks tests that are slow
    integration: mark test as integration test
```

## 3. Deployment Instructions

### 3.1 Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start Gradio-web interface
python main.py --mode serve --port 7860
```

### 3.2 Model Setup
Two options available:
1. Train model on your hardware:
```bash
python main.py --mode train
```
OR
2. Download pre-trained model:
Due to file-size limits, the trained model cannot be uploaded to GitHub. If you do not want to train your own model,
you can download the trained from: https://huggingface.co/formater/summarizer/tree/main

## 4. Ethical & Legal Compliance

### 4.1 Licensing
- MIT License
- Additional components:
  - BART model: Facebook AI Research (MIT)
  - CNN/DailyMail dataset: Fair use
  - Third-party libraries: Respective licenses

### 4.2 GDPR Compliance
- No persistent data storage
- Processing-only architecture

## 5. Performance Metrics

### 5.1 Model Performance
ROUGE scores:
- Rouge1: 0.4223
- Rouge2: 0.1935
- RougeL: 0.2889

### 5.2 System Requirements
- Python 3.8
- GPU (optional, for acceleration)
- 16 GB RAM
- 10 GB disk-space

## 6. Monitoring and Maintenance

### 6.1 Test Coverage
Current coverage metrics:
```plaintext
Name                              Stmts   Miss  Cover
---------------------------------------------------------------
src\data\data_loader.py              51     14    73%
src\models\summarizer.py             42      7    83%
src\utils\article_extractor.py       58     15    74%
src\utils\sentiment_analyzer.py      47     13    72%
---------------------------------------------------------------
TOTAL                               198     49    75%
```

### 6.2 Quality Gates
- Unit test pass rate: 100%
- Integration test pass rate: 95%
- Minimum coverage: 70%
