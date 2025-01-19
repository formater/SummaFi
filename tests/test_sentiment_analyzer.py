import pytest
from src.utils.sentiment_analyzer import SentimentAnalyzer


@pytest.mark.requires_model
def test_sentiment_analysis(check_model):
    """Test sentiment analysis."""
    analyzer = SentimentAnalyzer()

    # Test positive text
    positive_text = "The company reported strong earnings and increased dividends."
    positive_result = analyzer.analyze(positive_text)
    assert isinstance(positive_result, dict)
    assert sum(positive_result.values()) == pytest.approx(1.0, rel=1e-4)

    # Test negative text
    negative_text = "The stock crashed after poor quarterly results."
    negative_result = analyzer.analyze(negative_text)
    assert isinstance(negative_result, dict)
    assert sum(negative_result.values()) == pytest.approx(1.0, rel=1e-4)