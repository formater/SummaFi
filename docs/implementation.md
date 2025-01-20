# SummaFi Implementation Document

## 1. Architectural Overview

### 1.1 System Components
- **Core Model**: Facebook's BART model fine-tuned on the CNN/DailyMail dataset for financial news summarization.
- **Data Processing Pipeline**: Custom dataset loader and preprocessor designed to handle real-world financial articles.
- **Web Interface**: A Gradio-based interactive user interface for input submission and summary visualization.
- **Utility Components**:
  - **Article Extractor**: Extracts article content from URLs.
  - **Sentiment Analyzer**: Performs financial sentiment analysis using FinBERT.
- **Testing Framework**: Comprehensive pytest suite with unit, integration, and GPU-specific tests.

### 1.2 Key Dependencies
Key dependencies from `requirements.txt`:
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

---

## 2. Code Documentation

### 2.1 Core Components
#### Summarizer Model
```python
class NewsSummarizer:
    """Main summarization model wrapper."""
    # Features:
    # - Model initialization and configuration
    # - Text summarization
    # - GPU acceleration when available
    # - Stateless processing for GDPR compliance
```

#### Data Processing
```python
class SummarizationDataset:
    """Dataset loader and preprocessor."""
    # Features:
    # - Handles dataset loading and tokenization
    # - Preprocessing for batch inference
    # - Input truncation and error handling
```

#### Article Extractor
```python
class ArticleExtractor:
    """Extracts articles from URLs."""
    # Features:
    # - URL validation
    # - HTML content extraction
    # - Error handling for inaccessible URLs
```

#### Sentiment Analyzer
```python
class SentimentAnalyzer:
    """Financial sentiment analysis."""
    # Features:
    # - FinBERT-based sentiment detection
    # - Text chunking for long articles
    # - GPU support for enhanced performance
```

---

### 2.2 Testing Infrastructure
#### Categories and Coverage
- **Unit Tests**: Validates individual components (e.g., data preprocessing, URL validation).
- **Integration Tests**: Ensures seamless interaction between components (e.g., summarization pipeline).
- **GPU Tests**: Monitors hardware performance, memory usage, and batch processing.

#### Key Testing Enhancements:
- GPU memory validation during batch processing.
- End-to-end testing of web interface functionality.
- Strict URL validation and sanitization.

#### Test Markers
```ini
[pytest]
markers =
    gpu: Mark test as requiring GPU
    integration: Mark test as integration test
    slow: Mark test as slow
    requires_model: Mark test as requiring a trained model
```

---

## 3. Deployment Instructions

### 3.1 Local Development
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run tests**:
   ```bash
   pytest tests/ -v
   ```
3. **Launch Gradio interface**:
   ```bash
   python main.py --mode serve --port 7860
   ```

### 3.2 Model Setup
#### Training
Fine-tune the BART model:
```bash
python main.py --mode train --config config/config.yaml
```
#### Pre-trained Model
Alternatively, download the pre-trained model:
```plaintext
https://huggingface.co/formater/summarizer/tree/main
```

---

## 4. Ethical & Legal Compliance

### 4.1 Licensing
- **MIT License**: Covers the SummaFi system and its components.
- **BART Model**: Licensed under MIT by Facebook AI Research.
- **CNN/DailyMail Dataset**: Processed under fair use principles.

### 4.2 GDPR Compliance
- No persistent storage of user data.
- Real-time processing ensures user privacy.
- Transparent and secure data handling practices.

---

## 5. Performance Metrics

### 5.1 Model Performance
- **ROUGE-1**: 0.4223
- **ROUGE-2**: 0.1935
- **ROUGE-L**: 0.2889

### 5.2 System Requirements
- Python 3.8+
- GPU with CUDA support (optional for acceleration)
- 16GB RAM
- 10GB available disk space

---

## 6. Monitoring and Maintenance

### 6.1 Test Coverage
Current test coverage (from `pytest --cov`):
```plaintext
---------- coverage: platform win32, python 3.10.11-final-0 ----------
Name                              Stmts   Miss  Cover
---------------------------------------------------------------
src\data\data_loader.py              51     14    73%
src\models\summarizer.py             42      7    83%
src\utils\article_extractor.py       58     15    74%
src\utils\sentiment_analyzer.py      47     13    72%
---------------------------------------------------------------
TOTAL                               198     49    75%
```

### 6.2 Maintenance
- Regularly update test cases to reflect new features.
- Maintain a minimum test coverage of 70%.
- Ensure performance benchmarks are periodically reviewed.