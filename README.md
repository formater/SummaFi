# SummaFi - Financial News Summarization System

## Project Overview
SummaFi is an advanced financial news summarization system designed for my BSc final project. It uses Facebook's BART model fine-tuned on the CNN/DailyMail dataset to deliver concise, accurate summaries of financial news articles. Key metrics achieved by the summarizer:

- **Rouge1**: 0.4223  
- **Rouge2**: 0.1935  
- **RougeL**: 0.2889  

### Core Features
- **AI-powered summarization** using fine-tuned `BART`
- **Article extraction** from URLs with `newspaper3k`
- **Financial sentiment analysis** using `FinBERT`
- **Interactive web interface** built with `Gradio`
- **GDPR compliance** with real-time, stateless processing
- **Evaluation metrics** using ROUGE scores

---

## Technical Stack
- **Base Model**: facebook/bart-base  
- **Dataset**: CNN/DailyMail (3.0.0)  
- **Sentiment Analysis**: ProsusAI/finbert  
- **Frameworks**: PyTorch, HuggingFace Transformers  
- **Web Interface**: Gradio  
- **Experiment Tracking**: Weights & Biases  
- **Testing Framework**: pytest  

---

## Installation

### Prerequisites
- Python 3.8+ (Python 3.10 recommended)  
- CUDA-capable GPU (optional but recommended)  
- At least 16GB RAM  

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/formater/SummaFi.git
   cd SummaFi
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Configure Weights & Biases (if applicable):
   ```bash
   wandb login
   ```

---

## Usage

### Training
Fine-tune the model using:
```bash
python main.py --mode train --config config/config.yaml
```

### Alternative - Download pre-trained model
Due to file-size limits, the trained model cannot be uploaded to GitHub. If you do not want to train your own model,
you can download the trained from: https://huggingface.co/formater/summarizer/tree/main
### Evaluation
Evaluate model performance:
```bash
python main.py --mode evaluate --config config/config.yaml --model-path outputs/final_model
```

### Web Interface
Launch the interactive Gradio web interface:
```bash
python main.py --mode serve --model-path outputs/final_model --port 7860
```

---

## Project Structure
```
summa_fi/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── data/                # Data processing utilities
│   ├── models/              # Model definition and handling
│   ├── training/            # Training pipeline
│   ├── evaluation/          # Evaluation utilities
│   ├── utils/               # Helper functions
│   └── web/                 # Web interface implementation
├── tests/                   # Test suite
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## Configuration
Configuration details can be adjusted in `config/config.yaml`.

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

---

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_data_loader.py
pytest tests/test_article_extractor.py
pytest tests/test_sentiment_analyzer.py

# Run tests by marker
pytest -m gpu          # Run GPU-specific tests
pytest -m integration  # Run integration tests

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Test Categories
1. **Unit Tests**: Testing individual components like:
   - Data loading and preprocessing
   - Model initialization and inference
   - URL validation and article extraction  

2. **Integration Tests**: Ensuring components work together, including:
   - Complete training pipeline
   - End-to-end summarization process
   - Web interface functionality  

3. **GPU Tests**: Testing hardware-specific features:
   - Model GPU utilization
   - Mixed precision training
   - Memory management  


Please find detailed testing documentation at [docs/testing.md](docs/testing.md).

---

## Privacy & Legal Compliance

### GDPR Compliance
- No personal data storage  
- Real-time processing only  
- No cookies or tracking  
- Transparent data handling practices  

### Copyright Considerations
- Articles are processed under fair use principles  
- No article content is stored  
- Source attribution provided  

---

## Development Guidelines
- Code adheres to PEP 8 style guidelines
- Comprehensive docstrings and type hints
- Minimum test coverage: 70%

---

## Contributing
1. Fork the repository  
2. Create a feature branch  
3. Implement your changes  
4. Submit a pull request  

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [Amsterdam.Tech](https://amsterdam.tech/)
- [Facebook AI Research (BART)](https://huggingface.co/docs/transformers/en/model_doc/bart)  
- [HuggingFace](https://huggingface.co/)  
- [CNN/DailyMail Dataset](https://paperswithcode.com/dataset/cnn-daily-mail-1)  
- [Gradio](https://www.gradio.app/)  
- [FinBERT](https://huggingface.co/ProsusAI/finbert)  

---

## Contact
- **Author**: Dudás József  
- **Email**: [jozsef.dudas@gmail.com](mailto:jozsef.dudas@gmail.com)  
- **GitHub**: [formater](https://github.com/formater/)  
- **LinkedIn**: [Dudás József](https://www.linkedin.com/in/dudasjozsef/)
