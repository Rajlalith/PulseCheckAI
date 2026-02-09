# 🎯 PulseCheck AI

> Real-time social media sentiment analysis & customer feedback intelligence platform

Monitor brand mentions, detect emotions, and gain actionable insights with AI-powered NLP analysis.

---

### ✨ What You Can Do

✅ **Collect** real-time tweets from Twitter/X API v2  
✅ **Analyze** sentiment with **89% accuracy**  
✅ **Detect** 6+ emotions in real-time  
✅ **Extract** trending topics automatically  
✅ **Visualize** with interactive dashboards  
✅ **Export** to CSV, JSON & reports  

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| 🎨 **Frontend** | Streamlit, Plotly |
| 🤖 **ML/NLP** | Transformers, PyTorch, scikit-learn, NLTK |
| 📊 **Data** | Pandas, NumPy |
| ✔️ **Testing** | pytest |

## 📋 Prerequisites

- Python 3.9+
- Twitter API v2 Bearer Token (optional - sample data available)
- 4GB+ RAM  |  3GB+ disk space

## 🚀 Quick Start

```bash
# 1️⃣ Clone Repository
git clone https://github.com/yourusername/PulseCheckAI.git
cd PulseCheckAI

# 2️⃣ Setup Environment
python -m venv venv
source venv/bin/activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt

# 4️⃣ Configure API (Optional)
cp .env.example .env
# Add TWITTER_BEARER_TOKEN to .env

# 5️⃣ Download NLTK Data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 6️⃣ Launch Dashboard
streamlit run app.py
```

🌐 Open http://localhost:8501

## 📊 Dashboard Features

| Tab | Features |
|-----|----------|
| 📈 **Sentiment Overview** | Total tweets, distribution pie chart, score histogram |
| 😊 **Emotion Analysis** | Emotion frequency, sentiment heatmap |
| 📝 **Topic Modeling** | Top keywords, trending topics |
| ⏰ **Trends Over Time** | Sentiment timeline, volume tracking |
| 💾 **Export** | Download CSV, JSON, text reports |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Project Structure

```
src/
├── 🐦 twitter_collector.py      # Data collection
├── 😊 sentiment_analyzer.py     # Sentiment & emotion
└── 📝 topic_modeler.py          # Topics & keywords

tests/
├── test_twitter_collector.py
├── test_sentiment_analyzer.py
└── test_topic_modeler.py
```

---

## 🚀 Deployment

### 🏠 Local
```bash
streamlit run app.py
```

### 🐳 Docker
```bash
docker build -t pulsecheck-ai .
docker run -p 8501:8501 pulsecheck-ai
```

---

## 📜 License & Links

📄 **MIT License** - [View](LICENSE)  
📚 **Docs** - [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)  
🤝 **Contributing** - [CONTRIBUTING.md](CONTRIBUTING.md)  
🐛 **Issues** - [GitHub Issues](https://github.com/yourusername/PulseCheckAI/issues)
