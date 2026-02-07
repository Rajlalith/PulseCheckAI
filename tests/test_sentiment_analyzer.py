"""
Unit tests for SentimentAnalyzer module
"""

import pytest
import pandas as pd
from src.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    """Create SentimentAnalyzer instance."""
    return SentimentAnalyzer()


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
        assert analyzer.sentiment_pipeline is not None or analyzer.emotion_pipeline is not None

    def test_sentiment_analysis(self, analyzer):
        """Test sentiment analysis."""
        if analyzer.sentiment_pipeline is None:
            pytest.skip("Sentiment pipeline not available")

        text = "This product is absolutely amazing!"
        sentiment, score = analyzer.analyze_sentiment(text)

        assert sentiment in analyzer.SENTIMENT_LABELS
        assert 0 <= score <= 1

    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment detection."""
        if analyzer.sentiment_pipeline is None:
            pytest.skip("Sentiment pipeline not available")

        text = "This is the worst product ever!"
        sentiment, score = analyzer.analyze_sentiment(text)

        assert sentiment in analyzer.SENTIMENT_LABELS
        assert 0 <= score <= 1

    def test_emotion_detection(self, analyzer):
        """Test emotion detection."""
        if analyzer.emotion_pipeline is None:
            pytest.skip("Emotion pipeline not available")

        text = "I'm so happy and excited!"
        emotion, score = analyzer.detect_emotion(text)

        assert isinstance(emotion, str)
        assert 0 <= score <= 1

    def test_empty_text(self, analyzer):
        """Test handling of empty text."""
        sentiment, sent_score = analyzer.analyze_sentiment("")
        emotion, emo_score = analyzer.detect_emotion("")

        assert sentiment == "neutral"
        assert emotion == "neutral"

    def test_batch_analysis(self, analyzer):
        """Test batch sentiment analysis."""
        if analyzer.sentiment_pipeline is None:
            pytest.skip("Sentiment pipeline not available")

        df = pd.DataFrame({
            "text": [
                "Great product!",
                "Terrible experience",
                "It's okay"
            ]
        })

        result_df = analyzer.analyze_batch(df, show_progress=False)

        assert "sentiment" in result_df.columns
        assert "sentiment_score" in result_df.columns
        assert "emotion" in result_df.columns
        assert "emotion_score" in result_df.columns
        assert len(result_df) == 3

    def test_sentiment_summary(self, analyzer):
        """Test sentiment summary generation."""
        df = pd.DataFrame({
            "sentiment": ["positive", "negative", "neutral", "positive"]
        })

        summary = analyzer.get_sentiment_summary(df)

        assert "total_texts" in summary
        assert "positive_count" in summary
        assert "negative_count" in summary
        assert "neutral_count" in summary
        assert summary["total_texts"] == 4

    def test_emotion_summary(self, analyzer):
        """Test emotion summary generation."""
        df = pd.DataFrame({
            "emotion": ["joy", "sadness", "joy", "anger"]
        })

        summary = analyzer.get_emotion_summary(df)

        assert isinstance(summary, dict)
        assert summary.get("joy", 0) == 2

    def test_top_emotions(self, analyzer):
        """Test top emotions extraction."""
        df = pd.DataFrame({
            "emotion": ["joy", "joy", "sadness", "joy", "anger"]
        })

        top = analyzer.get_top_emotions(df, top_n=3)

        assert isinstance(top, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
