"""
Unit tests for TopicModeler module
"""

import pytest
import pandas as pd
from src.topic_modeler import TopicModeler


@pytest.fixture
def modeler():
    """Create TopicModeler instance."""
    return TopicModeler()


class TestTopicModeler:
    """Test suite for TopicModeler."""

    def test_initialization(self):
        """Test modeler initialization."""
        modeler = TopicModeler()
        assert modeler is not None
        assert modeler.stop_words is not None

    def test_text_preprocessing(self, modeler):
        """Test text preprocessing."""
        text = "Check out this amazing product at https://example.com #awesome @user"
        cleaned = modeler.preprocess_text(text)

        # Should be lowercase, no URLs, no special chars
        assert cleaned.islower()
        assert "http" not in cleaned
        assert "@" not in cleaned
        assert "#" not in cleaned

    def test_tokenization(self, modeler):
        """Test tokenization."""
        text = "This is a great product"
        tokens = modeler.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_keyword_extraction(self, modeler):
        """Test keyword extraction."""
        text = "This amazing product is really great and awesome. I love it!"
        keywords = modeler.extract_keywords(text, top_n=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5

    def test_empty_text_keywords(self, modeler):
        """Test keyword extraction on empty text."""
        keywords = modeler.extract_keywords("", top_n=5)
        assert keywords == []

    def test_extract_topics(self, modeler):
        """Test topic extraction."""
        texts = [
            "This product is amazing",
            "Great customer service",
            "I love this company"
        ]

        topics = modeler.extract_topics(texts, n_topics=2, n_words=3)

        assert isinstance(topics, list)
        assert len(topics) == 3

    def test_trending_topics(self, modeler):
        """Test trending topics identification."""
        texts = [
            "This product is great",
            "The product quality is excellent",
            "I love this product",
        ]

        trending = modeler.get_trending_topics(texts, top_n=5)

        assert isinstance(trending, list)
        assert all(isinstance(t, tuple) for t in trending)
        assert all(len(t) == 2 for t in trending)

    def test_tfidf_extraction(self, modeler):
        """Test TF-IDF based topic extraction."""
        texts = [
            "Python machine learning artificial intelligence",
            "Data science analytics visualization",
            "Natural language processing NLP models"
        ]

        topics = modeler.extract_topics_tfidf(texts, n_topics=3, n_words=3)

        assert isinstance(topics, dict)
        assert len(topics) <= 3

    def test_topics_analysis(self, modeler):
        """Test topics analysis on DataFrame."""
        df = pd.DataFrame({
            "text": [
                "Great product quality",
                "Excellent customer service",
                "Amazing support team"
            ]
        })

        result_df = modeler.analyze_topics(df)

        assert "topics" in result_df.columns
        assert len(result_df) == 3

    def test_topics_summary(self, modeler):
        """Test topics summary generation."""
        df = pd.DataFrame({
            "topics": ["product, quality", "service, support", "product, great"]
        })

        summary = modeler.get_topics_summary(df)

        assert isinstance(summary, dict)
        assert summary.get("product", 0) >= 2

    def test_top_topics(self, modeler):
        """Test top topics extraction."""
        df = pd.DataFrame({
            "topics": ["product, quality", "service, support", "product, great"] * 3
        })

        top = modeler.get_top_topics(df, top_n=5)

        assert isinstance(top, list)
        assert all(isinstance(t, tuple) for t in top)

    def test_preprocessor_url_removal(self, modeler):
        """Test URL removal in preprocessing."""
        text = "Visit https://example.com and www.test.org for more"
        cleaned = modeler.preprocess_text(text)

        assert "http" not in cleaned
        assert "example.com" not in cleaned

    def test_preprocessor_lowercase(self, modeler):
        """Test lowercase conversion."""
        text = "This Is A TEST"
        cleaned = modeler.preprocess_text(text)

        assert cleaned == cleaned.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
