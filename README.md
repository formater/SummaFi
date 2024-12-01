# SummaFi - Financial News Summarization System

## Project Overview
SummaFi is an advanced financial news summarization system built for my BSc final project. It leverages Facebook's BART
model fine-tuned on the CNN/DailyMail dataset to provide accurate, concise summaries of financial news articles. The
summarizer achieves ROUGE scores:

- Rouge1: 0.4223
- Rouge2: 0.1935
- RougeL: 0.2889


### Core Features
- AI-powered text summarization using fine-tuned BART model
- Automatic article extraction from URLs using newspaper3k
- Financial sentiment analysis using FinBERT
- GDPR-compliant processing (no data storage)
- Interactive web interface using Gradio
- Comprehensive evaluation using ROUGE metrics

## Technical Stack
- **Base Model**: facebook/bart-base
- **Dataset**: CNN/DailyMail (3.0.0)
- **Sentiment Analysis**: ProsusAI/finbert
- **Framework**: PyTorch, HuggingFace Transformers
- **Web Interface**: Gradio
- **Article Extraction**: newspaper3k
- **Experiment Tracking**: Weights & Biases
- **Testing**: pytest

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB RAM (recommended)
- Git

### Setup Steps

1. Clone repository:
```bash
git clone https://github.com/formater/SummaFi.git
cd SummaFi
```

2. Create virtual environment:
```bash
python -m venv .
source bin/activate  # Windows: Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Weights & Biases:
```bash
wandb login
```

## Usage

### Training
Fine-tune the model:
```bash
python main.py --mode train --config config/config.yaml
```

### Evaluation
Evaluate model performance:
```bash
python main.py --mode evaluate --config config/config.yaml --model-path outputs/final_model
```

### Web Interface
Launch the Gradio interface:
```bash
python main.py --mode serve --model-path outputs/final_model --port 7860
```

## Project Structure
```
summa_fi/
├── config/
│   └── config.yaml          # Configuration settings
├── src/
│   ├── data/               # Data processing
│   ├── models/             # Model architecture
│   ├── training/           # Training logic
│   ├── evaluation/         # Evaluation metrics
│   ├── utils/              # Utilities
│   └── web/               # Web interface
├── tests/                  # Test suite
├── docs/                   # Documentation
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Configuration
Key configuration sections in `config/config.yaml`:

### Model Settings
```yaml
model:
  name: "facebook/bart-base"
  max_length: 1024
  min_length: 56
  length_penalty: 2.0
```

### Training Settings
```yaml
training:
  batch_size: 8
  learning_rate: 3e-5
  num_epochs: 3
  warmup_steps: 500
```

## Privacy & Legal Compliance

### GDPR Compliance
- No personal data storage
- Real-time processing only
- No cookies or tracking
- Transparent data handling

### Copyright Considerations
- Fair use implementation
- No article storage
- Source attribution
- robots.txt compliance

## Development Guidelines

### Code Style
- PEP 8 compliance
- Type hints (Python 3.8+)
- Comprehensive docstrings
- Clean code principles

### Testing
Run tests:
```bash
pytest tests/
pytest --cov=src tests/  # with coverage
```

## Troubleshooting

### Common Issues
1. Memory Issues
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training

2. URL Access
   - Verify URL format
   - Check site accessibility
   - Confirm robots.txt compliance

## Contributing
1. Fork repository
2. Create feature branch
3. Implement changes
4. Submit pull request

## License
MIT License - See LICENSE file

## Acknowledgments
- Facebook AI Research (BART)
- HuggingFace Team
- CNN/DailyMail Dataset
- Gradio Team

## Contact
- Author: Dudás József
- Email: jozsef.dudas@gmail.com
- GitHub: formater

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_data_loader.py
pytest tests/test_summarizer.py
pytest tests/test_article_extractor.py

# Run tests by marker
pytest -m gpu          # Run GPU-specific tests
pytest -m "not slow"   # Skip slow tests
pytest -m integration  # Run integration tests

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
  - Data loading and preprocessing
  - Model initialization and inference
  - Article extraction and URL validation

- **Integration Tests**: Component interaction testing
  - Complete training pipeline
  - End-to-end summarization process
  - Web interface functionality

- **GPU Tests**: Hardware-specific testing
  - Model GPU utilization
  - Mixed precision training
  - Memory optimization

### Coverage Requirements
- Minimum overall coverage: 80%
- Critical paths coverage: 90%
- Web interface coverage: 85%

### Continuous Integration
Tests are automatically run on:
- Every push to main branch
- Pull request creation
- Release tag creation