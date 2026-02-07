"""
Topic Modeling and Keyword Extraction Module for PulseCheck AI

This module provides keyword extraction, topic modeling, and trending
topic identification using TF-IDF and frequency analysis.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

logger = logging.getLogger(__name__)


class TopicModeler:
    """Performs topic modeling and keyword extraction on text data."""

    def __init__(self):
        """Initialize TopicModeler with stopwords."""
        self.logger = logging.getLogger(__name__)
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception as e:
            self.logger.warning(f"Could not load stopwords: {e}")
            self.stop_words = set()

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for analysis.

        Args:
            text: Input text to preprocess

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove mentions and hashtags symbols (keep the words)
        text = re.sub(r"@|#", "", text)

        # Remove special characters and extra whitespace
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text.lower())
            # Filter out stopwords and short tokens
            tokens = [
                t for t in tokens
                if t.isalnum() and len(t) > 2 and t not in self.stop_words
            ]
            return tokens
        except Exception as e:
            self.logger.error(f"Tokenization error: {e}")
            return []

    def extract_keywords(
        self,
        text: str,
        top_n: int = 10
    ) -> List[str]:
        """
        Extract most important keywords from text.

        Args:
            text: Input text
            top_n: Number of keywords to return

        Returns:
            List of keyword strings
        """
        cleaned_text = self.preprocess_text(text)
        tokens = self.tokenize(cleaned_text)

        if not tokens:
            return []

        # Count token frequencies
        counter = Counter(tokens)
        top_keywords = [token for token, _ in counter.most_common(top_n)]

        return top_keywords

    def extract_topics(
        self,
        texts: List[str],
        n_topics: int = 5,
        n_words: int = 5
    ) -> List[List[str]]:
        """
        Extract topics from multiple texts using frequency analysis.

        Args:
            texts: List of input texts
            n_topics: Number of topics (not used in current implementation)
            n_words: Words per topic

        Returns:
            List of topic lists (one per text)
        """
        topics = []

        for text in texts:
            keywords = self.extract_keywords(text, top_n=n_words)
            topics.append(keywords)

        return topics

    def get_trending_topics(
        self,
        texts: List[str],
        top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """
        Identify overall trending topics across corpus.

        Args:
            texts: Corpus of texts
            top_n: Number of top topics to return

        Returns:
            List of (topic, frequency) tuples
        """
        all_keywords = []

        for text in texts:
            keywords = self.extract_keywords(text, top_n=20)
            all_keywords.extend(keywords)

        if not all_keywords:
            return []

        # Count frequencies
        keyword_counts = Counter(all_keywords)
        trending_topics = keyword_counts.most_common(top_n)

        return trending_topics

    def extract_topics_tfidf(
        self,
        texts: List[str],
        n_topics: int = 5,
        n_words: int = 10,
        max_features: int = 100
    ) -> Dict[int, List[str]]:
        """
        Advanced topic extraction using TF-IDF vectorization.

        Args:
            texts: List of input texts
            n_topics: Number of topics to extract
            n_words: Words per topic
            max_features: Maximum features for vectorizer

        Returns:
            Dictionary mapping topic IDs to keyword lists
        """
        if not texts or len(texts) < 2:
            return {}

        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]

            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),
                lowercase=True,
                stop_words="english"
            )

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Extract top features per topic
            topics_dict = {}

            for topic_id in range(min(n_topics, len(processed_texts))):
                # Use document as topic
                doc_vector = tfidf_matrix[topic_id].toarray().flatten()
                top_indices = doc_vector.argsort()[-n_words:][::-1]
                top_words = [feature_names[idx] for idx in top_indices if doc_vector[idx] > 0]
                topics_dict[topic_id] = top_words

            return topics_dict

        except Exception as e:
            self.logger.error(f"TF-IDF extraction error: {e}")
            return {}

    def analyze_topics(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Analyze topics for all texts in DataFrame.

        Args:
            df: Input DataFrame
            text_column: Column name containing text

        Returns:
            DataFrame with added topics column
        """
        df = df.copy()
        all_topics = []

        for text in df[text_column]:
            keywords = self.extract_keywords(str(text), top_n=5)
            all_topics.append(", ".join(keywords) if keywords else "general")

        df["topics"] = all_topics
        return df

    def get_topics_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get summary of topics from DataFrame.

        Args:
            df: DataFrame with topics column

        Returns:
            Dictionary with topic frequencies
        """
        if "topics" not in df.columns:
            return {}

        topic_list = []
        for topics_str in df["topics"]:
            if isinstance(topics_str, str):
                topic_list.extend(topics_str.split(", "))

        topic_counts = Counter(topic_list)
        return dict(topic_counts.most_common(20))

    def get_top_topics(
        self,
        df: pd.DataFrame,
        top_n: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Get top topics from DataFrame.

        Args:
            df: DataFrame with topics column
            top_n: Number of top topics

        Returns:
            List of (topic, count) tuples
        """
        summary = self.get_topics_summary(df)
        return sorted(summary.items(), key=lambda x: x[1], reverse=True)[:top_n]
