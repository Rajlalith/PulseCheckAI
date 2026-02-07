"""
Unit tests for TwitterCollector module
"""

import pytest
import pandas as pd
from src.twitter_collector import TwitterCollector


@pytest.fixture
def collector():
    """Create TwitterCollector instance without API token."""
    return TwitterCollector(bearer_token=None)


class TestTwitterCollector:
    """Test suite for TwitterCollector."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = TwitterCollector()
        assert collector is not None
        assert collector.bearer_token is None

    def test_sample_data_generation(self, collector):
        """Test sample data generation."""
        df = collector.search_sample_data(query="test", n_samples=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "text" in df.columns
        assert "likes" in df.columns
        assert "retweets" in df.columns

    def test_sample_data_columns(self, collector):
        """Test that sample data has all required columns."""
        df = collector.search_sample_data(query="test", n_samples=5)

        required_columns = ["tweet_id", "text", "created_at", "likes", "retweets", "replies", "quotes"]
        for col in required_columns:
            assert col in df.columns

    def test_sample_data_values(self, collector):
        """Test sample data values are reasonable."""
        df = collector.search_sample_data(query="test", n_samples=5)

        # Check that numeric columns contain valid values
        assert (df["likes"] >= 0).all()
        assert (df["retweets"] >= 0).all()
        assert (df["replies"] >= 0).all()

    def test_collection_stats(self, collector):
        """Test collection statistics."""
        df = collector.search_sample_data(query="test", n_samples=10)
        stats = collector.get_collection_stats(df)

        assert "total_tweets" in stats
        assert "total_engagement" in stats
        assert "avg_likes" in stats
        assert stats["total_tweets"] == 10

    def test_empty_dataframe_stats(self, collector):
        """Test statistics on empty DataFrame."""
        df = pd.DataFrame(columns=["likes", "retweets", "replies"])
        stats = collector.get_collection_stats(df)

        assert stats["total_tweets"] == 0
        assert stats["total_engagement"] == 0

    def test_sample_data_quantity(self, collector):
        """Test different sample sizes."""
        for n_samples in [10, 50, 100]:
            df = collector.search_sample_data(query="test", n_samples=n_samples)
            assert len(df) == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
