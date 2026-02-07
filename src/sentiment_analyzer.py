"""
Sentiment Analysis and Emotion Detection Module for PulseCheck AI

This module provides sentiment classification and emotion detection using
state-of-the-art transformer models from Hugging Face.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
import warnings
import torch
from transformers import pipeline
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment and emotions using transformer models."""

    SENTIMENT_LABELS = ["negative", "neutral", "positive"]
    EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "love"]

    def __init__(self, device: Optional[str] = None):
        """
        Initialize SentimentAnalyzer with pre-trained models.

        Args:
            device: Device to run models on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or (0 if torch.cuda.is_available() else -1)
        self.logger = logging.getLogger(__name__)

        # Load sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device,
                truncation=True,
                max_length=512
            )
            self.logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load sentiment model: {e}")
            self.sentiment_pipeline = None

        # Load emotion detection pipeline
        try:
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device,
                truncation=True,
                max_length=512
            )
            self.logger.info("Emotion detection model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load emotion model: {e}")
            self.emotion_pipeline = None

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Classify sentiment of single text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (sentiment_label, confidence_score)
            Labels: 'positive', 'negative', 'neutral'
        """
        if not text or not self.sentiment_pipeline:
            return "neutral", 0.0

        try:
            # Truncate text to 512 tokens
            text = text[:512] if len(text) > 512 else text

            result = self.sentiment_pipeline(text)[0]

            # Map model labels to standard labels
            label = result["label"]
            score = result["score"]

            # Convert label format (e.g., "LABEL_0" -> "negative")
            if "LABEL" in label:
                label_idx = int(label.split("_")[1])
                label = self.SENTIMENT_LABELS[label_idx]

            return label.lower(), float(score)

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return "neutral", 0.0

    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """
        Identify primary emotion in text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (emotion_label, confidence_score)
            Emotions: joy, sadness, anger, fear, surprise, love, neutral
        """
        if not text or not self.emotion_pipeline:
            return "neutral", 0.0

        try:
            # Truncate text to 512 tokens
            text = text[:512] if len(text) > 512 else text

            result = self.emotion_pipeline(text)[0]

            label = result["label"].lower()
            score = result["score"]

            return label, float(score)

        except Exception as e:
            self.logger.error(f"Error detecting emotion: {e}")
            return "neutral", 0.0

    def analyze_batch(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Batch process multiple texts efficiently.

        Args:
            df: Input DataFrame
            text_column: Column name containing text to analyze
            show_progress: Whether to show progress (for Streamlit)

        Returns:
            DataFrame with added sentiment/emotion columns
        """
        df = df.copy()
        sentiments = []
        sentiment_scores = []
        emotions = []
        emotion_scores = []

        total = len(df)

        for idx, text in enumerate(df[text_column]):
            if show_progress and idx % max(1, total // 10) == 0:
                self.logger.info(f"Processing {idx}/{total} texts")

            sentiment, sent_score = self.analyze_sentiment(str(text))
            emotion, emo_score = self.detect_emotion(str(text))

            sentiments.append(sentiment)
            sentiment_scores.append(sent_score)
            emotions.append(emotion)
            emotion_scores.append(emo_score)

        df["sentiment"] = sentiments
        df["sentiment_score"] = sentiment_scores
        df["emotion"] = emotions
        df["emotion_score"] = emotion_scores

        return df

    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistical summary of sentiment analysis.

        Args:
            df: DataFrame with sentiment column

        Returns:
            Dictionary with counts, percentages, and statistics
        """
        if "sentiment" not in df.columns or len(df) == 0:
            return {
                "total_texts": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_percentage": 0.0,
                "negative_percentage": 0.0,
                "neutral_percentage": 0.0,
                "avg_sentiment_score": 0.0,
            }

        value_counts = df["sentiment"].value_counts()
        total = len(df)

        return {
            "total_texts": total,
            "positive_count": int(value_counts.get("positive", 0)),
            "negative_count": int(value_counts.get("negative", 0)),
            "neutral_count": int(value_counts.get("neutral", 0)),
            "positive_percentage": float((value_counts.get("positive", 0) / total * 100) if total > 0 else 0),
            "negative_percentage": float((value_counts.get("negative", 0) / total * 100) if total > 0 else 0),
            "neutral_percentage": float((value_counts.get("neutral", 0) / total * 100) if total > 0 else 0),
            "avg_sentiment_score": float(df.get("sentiment_score", [0]).mean()),
        }

    def get_emotion_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Generate statistical summary of emotion detection.

        Args:
            df: DataFrame with emotion column

        Returns:
            Dictionary with emotion counts
        """
        if "emotion" not in df.columns:
            return {}

        emotion_counts = df["emotion"].value_counts().to_dict()
        return {k: int(v) for k, v in emotion_counts.items()}

    def get_top_emotions(self, df: pd.DataFrame, top_n: int = 6) -> List[Tuple[str, int]]:
        """
        Get top emotions from DataFrame.

        Args:
            df: DataFrame with emotion column
            top_n: Number of top emotions to return

        Returns:
            List of (emotion, count) tuples
        """
        if "emotion" not in df.columns:
            return []

        return df["emotion"].value_counts().head(top_n).to_list()
