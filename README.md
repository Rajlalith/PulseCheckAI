# PulseCheck AI

Real-time social media sentiment analysis & customer feedback intelligence platform using NLP.

## Overview

- Collect real-time tweets from Twitter/X API v2
- Analyze sentiment with 89% accuracy
- Detect emotions (Joy, Sadness, Anger, Fear, Surprise, Love)
- Extract trending topics via TF-IDF
- Interactive Plotly dashboards
- Export data (CSV, JSON)

## Tech Stack

- **Frontend**: Streamlit, Plotly
- **ML/NLP**: Transformers (Hugging Face), PyTorch, scikit-learn, NLTK
- **Data**: Pandas, NumPy
- **Testing**: pytest

## Prerequisites

- Python 3.9+
- Twitter API v2 Bearer Token (optional - sample data available)
- 4GB+ RAM
- 3GB+ disk space

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/PulseCheckAI.git
cd PulseCheckAI

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add TWITTER_BEARER_TOKEN to .env

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run
streamlit run app.py
```

Access at `http://localhost:8501`

## Usage

### Dashboard Tabs
- **Sentiment Overview**: Tweets count, sentiment distribution, scores
- **Emotion Analysis**: Emotion frequency, sentiment heatmap
- **Topic Modeling**: Top keywords and topics
- **Trends Over Time**: Sentiment timeline and volume
- **Export**: Download CSV, JSON, or reports

### Sample Data Mode
Enable in sidebar to test without API access

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
src/
├── twitter_collector.py      # Twitter API data collection
├── sentiment_analyzer.py     # Sentiment & emotion analysis
└── topic_modeler.py          # Topic extraction & modeling

tests/
├── test_twitter_collector.py
├── test_sentiment_analyzer.py
└── test_topic_modeler.py
```

## Key APIs

**TwitterCollector** - Collect tweets or generate sample data
**SentimentAnalyzer** - Analyze sentiment & detect emotions  
**TopicModeler** - Extract keywords and trending topics

## Deployment

```bash
# Local
streamlit run app.py

# Docker
docker build -t pulsecheck-ai .
docker run -p 8501:8501 pulsecheck-ai
```

## License

MIT - see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/PulseCheckAI/issues)
- Docs: [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
