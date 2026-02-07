"""
Twitter Data Collection Module for PulseCheck AI

This module handles real-time tweet collection from Twitter API v2 with
rate limiting, error handling, and sample data fallback functionality.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import requests
import pandas as pd
import random

logger = logging.getLogger(__name__)


class TwitterCollector:
    """Collects tweets from Twitter API v2 with rate limiting and error handling."""

    BASE_URL = "https://api.twitter.com/2/tweets/search/recent"
    SAMPLE_KEYWORDS = [
        "technology", "product", "customer service", "innovation", "ai",
        "machine learning", "python", "cloud", "startup", "digital",
        "software", "development", "data science", "analytics", "business"
    ]

    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize TwitterCollector.

        Args:
            bearer_token: Twitter API v2 bearer token. If None, sample data mode is used.
        """
        self.bearer_token = bearer_token
        self.headers = self._prepare_headers() if bearer_token else None
        self.logger = logging.getLogger(__name__)

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "PulseCheckAI/1.0"
        }

    def collect_tweets(
        self,
        query: str,
        max_results: int = 100,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch tweets from Twitter API v2.

        Args:
            query: Search query string (keywords, hashtags, mentions)
            max_results: Maximum tweets to fetch (10-500, max 100 per request)
            days_back: Number of days to look back (1-7)

        Returns:
            Pandas DataFrame with tweet data
        """
        if not self.bearer_token:
            self.logger.info(f"No bearer token provided. Generating sample data for query: {query}")
            return self.search_sample_data(query, n_samples=max_results)

        try:
            return self._fetch_from_api(query, max_results, days_back)
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}. Falling back to sample data.")
            return self.search_sample_data(query, n_samples=max_results)

    def _fetch_from_api(
        self,
        query: str,
        max_results: int,
        days_back: int
    ) -> pd.DataFrame:
        """
        Internal method to fetch tweets from Twitter API.

        Args:
            query: Search query string
            max_results: Maximum tweets to fetch
            days_back: Number of days to look back

        Returns:
            DataFrame with tweets
        """
        # Calculate start time
        start_time = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Prepare request parameters
        query_params = {
            "query": query,
            "max_results": min(max_results, 100),  # API limits to 100 per request
            "tweet.fields": "created_at,public_metrics,author_id",
            "user.fields": "username,verified",
            "expansions": "author_id",
            "start_time": start_time
        }

        all_tweets = []
        next_token = None
        batches = 0
        max_batches = (max_results // 100) + 1

        while batches < max_batches and len(all_tweets) < max_results:
            if next_token:
                query_params["next_token"] = next_token

            response = requests.get(
                self.BASE_URL,
                headers=self.headers,
                params=query_params,
                timeout=10
            )

            if response.status_code == 429:
                self.logger.warning("Rate limit hit. Falling back to sample data.")
                raise Exception("Rate limit exceeded")

            response.raise_for_status()
            data = response.json()

            if "data" not in data:
                break

            tweets = data["data"]
            if not tweets:
                break

            all_tweets.extend(tweets)
            next_token = data.get("meta", {}).get("next_token")

            if not next_token:
                break

            batches += 1

        return self._parse_tweets(all_tweets[:max_results])

    def _parse_tweets(self, tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse tweets into a structured DataFrame.

        Args:
            tweets: List of tweet dictionaries from API

        Returns:
            Structured DataFrame
        """
        parsed_tweets = []

        for tweet in tweets:
            parsed_tweet = {
                "tweet_id": tweet.get("id"),
                "text": tweet.get("text"),
                "created_at": tweet.get("created_at"),
                "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                "replies": tweet.get("public_metrics", {}).get("reply_count", 0),
                "quotes": tweet.get("public_metrics", {}).get("quote_count", 0),
            }
            parsed_tweets.append(parsed_tweet)

        return pd.DataFrame(parsed_tweets)

    def search_sample_data(self, query: str, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate realistic sample data for testing without API access.

        Args:
            query: Query context (for sample generation)
            n_samples: Number of sample tweets to generate

        Returns:
            DataFrame with sample tweets
        """
        sample_sentiments = [
            "This product is amazing! Highly recommend.",
            "Not satisfied with the service quality.",
            "Just started using this. Let's see how it goes.",
            "Love the new features! Best update yet.",
            "Worst experience ever. Won't be using anymore.",
            "Decent product, but could use some improvements.",
            "Finally got what we needed! Thanks for the great work.",
            "The interface is confusing and needs redesign.",
            "Pretty good overall. Some minor issues though.",
            "Absolutely fantastic! Changed our workflow completely.",
            "Disappointed with the recent changes.",
            "Solid performance. Would definitely recommend.",
            "Customer support was incredibly helpful!",
            "The pricing is way too high for what you get.",
            "Amazing innovation in the industry!",
        ]

        tweets_data = []
        base_time = datetime.utcnow()

        for i in range(n_samples):
            tweet_time = base_time - timedelta(hours=random.randint(0, 168))
            tweets_data.append({
                "tweet_id": f"sample_{i:06d}",
                "text": random.choice(sample_sentiments),
                "created_at": tweet_time.isoformat() + "Z",
                "likes": random.randint(0, 1000),
                "retweets": random.randint(0, 500),
                "replies": random.randint(0, 100),
                "quotes": random.randint(0, 50),
            })

        return pd.DataFrame(tweets_data)

    def get_collection_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get collection statistics from DataFrame.

        Args:
            df: DataFrame with tweet data

        Returns:
            Dictionary with statistics
        """
        return {
            "total_tweets": len(df),
            "total_engagement": int(df[["likes", "retweets", "replies"]].sum().sum()),
            "avg_likes": float(df["likes"].mean()) if len(df) > 0 else 0,
            "avg_retweets": float(df["retweets"].mean()) if len(df) > 0 else 0,
        }
