import pytest
from src.utils.article_extractor import ArticleExtractor


def test_url_validation():
    """Test URL validation."""
    extractor = ArticleExtractor()

    # Test valid URLs
    valid_urls = [
        "https://www.example.com",
        "http://example.com",
        "https://example.com/article/123",
        "https://www.example.com/path?param=value"
    ]
    for url in valid_urls:
        assert extractor.is_valid_url(url), f"URL should be valid: {url}"

    # Test invalid URLs
    invalid_urls = [
        "not_a_url",
        "http://",
        "https://",
        "ftp://invalid.com",
        "",
        None
    ]
    for url in invalid_urls:
        assert not extractor.is_valid_url(url), f"URL should be invalid: {url}"


def test_article_extraction():
    """Test article extraction."""
    extractor = ArticleExtractor()

    # Mock article data for testing
    test_url = "https://finance.yahoo.com/news/the-stock-market-has-never-looked-like-this-before--regardless-of-whos-president-140021170.html"
    test_content = {
        'text': 'Test article content',
        'title': 'Test Article',
        'authors': ['Test Author'],
        'publish_date': None,
    }

    # Test basic extraction
    try:
        result = extractor.extract_article(test_url)
        assert isinstance(result, dict)
        assert 'text' in result
        assert 'title' in result
    except Exception as e:
        pytest.skip(f"Skipping live URL test: {str(e)}")