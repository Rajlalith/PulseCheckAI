# PulseCheck AI - Architecture Guide

## System Architecture Overview

PulseCheck AI follows a modular, three-tier architecture designed for scalability, maintainability, and extensibility.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│                    (Streamlit Dashboard)                        │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │  Sentiment   │  Emotion     │  Topics      │  Trends      │ │
│  │  Overview    │  Analysis    │  Modeling    │  Analysis    │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                   PROCESSING LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Twitter    │  │ Sentiment &  │  │     Topic            │ │
│  │  Collector   │  │   Emotion    │  │    Modeler           │ │
│  │   Module     │  │   Analyzer   │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                    DATA LAYER                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │     Data Processing & Storage (Pandas/NumPy)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │     Pre-trained NLP Models (Hugging Face)               │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
            ┌───────▼──────┐  ┌───▼────────┐
            │ Twitter API  │  │ Local/GPU  │
            │     v2       │  │ Inference  │
            └──────────────┘  └────────────┘
```

## Component Architecture

### 1. Presentation Layer

**Technology**: Streamlit 1.29.0

**Responsibilities**:
- User interface and interactions
- Data visualization with Plotly
- Session state management
- Data export functionality

**Key Features**:
- Multi-tab dashboard design
- Real-time reactive updates
- Downloadable exports
- Responsive layout

### 2. Processing Layer

#### A. TwitterCollector Module (`src/twitter_collector.py`)

**Purpose**: Real-time data ingestion from social media

**Key Functions**:
```python
collect_tweets(bearer_token, query, max_results, days_back)
    ↓
    ├─ _fetch_from_api() [if token available]
    │   ├─ HTTP request to Twitter API v2
    │   ├─ Pagination handling
    │   └─ Response parsing
    │
    └─ search_sample_data() [fallback]
        └─ Generate synthetic tweets
```

**Features**:
- OAuth 2.0 Bearer Token authentication
- Automatic rate limit handling
- Sample data generator for testing
- Metadata extraction (engagement metrics)

#### B. SentimentAnalyzer Module (`src/sentiment_analyzer.py`)

**Purpose**: Sentiment classification and emotion detection

**Models**:
- **Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Fine-tuned on 124M tweets
  - 89% accuracy on Twitter data
  - 3-class classification (Positive, Negative, Neutral)

- **Emotion**: `j-hartmann/emotion-english-distilroberta-base`
  - Based on DistilRoBERTa
  - 82% accuracy on GoEmotions dataset
  - 6+ emotion detection

**Key Functions**:
```python
analyze_sentiment(text) → (label, confidence)
detect_emotion(text) → (emotion, confidence)
analyze_batch(df, text_column) → df with predictions
get_sentiment_summary(df) → statistics
```

**Architecture**:
```
Input Text
    ↓
Hugging Face Pipeline
    ├─ Tokenization
    ├─ Model Inference
    └─ Post-processing
    ↓
(Label, Confidence Score)
```

#### C. TopicModeler Module (`src/topic_modeler.py`)

**Purpose**: Topic extraction and keyword identification

**Algorithms**:
1. **TF-IDF Vectorization**
   - Frequency-based importance weighting
   - N-gram support (unigrams, bigrams)
   - Feature selection and filtering

2. **Frequency Analysis**
   - Term frequency counting
   - Stopword removal
   - Tokenization with NLTK

**Key Functions**:
```python
extract_keywords(text, top_n) → [keywords]
get_trending_topics(texts, top_n) → [(topic, freq)]
extract_topics_tfidf(texts, n_topics, n_words) → {topic_id: [words]}
analyze_topics(df) → df with topic column
```

### 3. Data Layer

**Data Processing**:
- Pandas DataFrames for tabular data
- NumPy arrays for numerical operations
- Efficient batch operations

**Model Storage**:
- Hugging Face cache directory
- ~2GB total for all models
- Automatic model downloading on first run

## Data Flow

### Complete Analysis Pipeline

```
1. Data Collection
   ↓
   [TwitterCollector] ← Twitter API v2
   ↓
   Raw DataFrame (tweets, metadata)

2. Sentiment Analysis
   ↓
   [SentimentAnalyzer]
   │  ├─ Sentiment classification
   │  └─ Emotion detection
   ↓
   DataFrame with sentiment/emotion columns

3. Topic Modeling
   ↓
   [TopicModeler]
   │  ├─ Text preprocessing
   │  ├─ Keyword extraction
   │  └─ TF-IDF vectorization
   ↓
   DataFrame with topics column

4. Visualization
   ↓
   [Streamlit + Plotly]
   ├─ Charts and graphs
   ├─ Statistics and summaries
   └─ Export options
   ↓
   Interactive Dashboard
```

## Key Design Patterns

### 1. Pipeline Pattern
Each module handles one step of processing:
```python
collector → analyzer → modeler → dashboard
```

### 2. Caching Strategy
```python
@st.cache_resource
def initialize_modules():
    return collectors, analyzer, modeler
```

### 3. Error Handling with Fallbacks
```python
try:
    use_api()
except Exception:
    use_sample_data()
```

### 4. Batch Processing
```python
for batch in chunks(data, batch_size):
    process_batch(batch)
```

## Performance Considerations

### Model Inference
- **CPU**: ~10 texts/second
- **GPU**: ~50 texts/second
- Batch processing for efficiency
- Model caching for reuse

### Memory Management
- Pandas for efficient data handling
- NumPy for vectorized operations
- Model loading once at initialization
- Session state for persistence

### Latency Targets
- Data collection: < 30 seconds (100 tweets)
- Sentiment analysis: < 10 seconds (CPU)
- Topic extraction: < 3 seconds
- Dashboard rendering: < 3 seconds

## Scalability Architecture

### Horizontal Scaling
```
Load Balancer
    ├─ Instance 1 [Dashboard + Models]
    ├─ Instance 2 [Dashboard + Models]
    └─ Instance 3 [Dashboard + Models]
         ↓
    Shared Data Store (optional)
```

### Containerization (Docker)
```dockerfile
FROM python:3.9-slim
├─ Dependencies
├─ Models
└─ Application Code
```

### Cloud Deployment
- **Stateless design**: Easy replication
- **API-based**: Can separate services
- **Environment variables**: Configuration management

## Extension Points

### Adding New Data Sources
```python
class LinkedInCollector(DataCollectorBase):
    def collect_posts(self, ...):
        pass
```

### Adding New ML Models
```python
class AspectSentimentAnalyzer:
    def analyze_aspect_sentiment(self, ...):
        pass
```

### Custom Visualizations
```python
def create_custom_chart(df):
    return go.Figure(...)
```

## Dependency Graph

```
app.py (Main Application)
├── src/twitter_collector.py
│   ├── requests (HTTP)
│   ├── pandas (DataFrames)
│   └── logging
├── src/sentiment_analyzer.py
│   ├── transformers (Models)
│   ├── torch (PyTorch)
│   ├── pandas (DataFrames)
│   └── logging
└── src/topic_modeler.py
    ├── nltk (NLP)
    ├── scikit-learn (TF-IDF)
    ├── pandas (DataFrames)
    └── logging

External
├── streamlit (UI)
├── plotly (Visualization)
├── python-dotenv (Config)
└── numpy (Numerical)
```

## Testing Architecture

```
tests/
├── test_twitter_collector.py
│   └─ Unit tests for data collection
├── test_sentiment_analyzer.py
│   └─ Unit tests for sentiment/emotion
└── test_topic_modeler.py
    └─ Unit tests for topic extraction

Coverage Target: > 80%
```

## Configuration Management

**Environment Variables** (`.env`):
```
TWITTER_BEARER_TOKEN=...
USE_GPU=false
LOG_LEVEL=INFO
```

**Streamlit Config** (`~/.streamlit/config.toml`):
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

## Security Architecture

### API Security
- OAuth 2.0 for Twitter authentication
- No hardcoded credentials
- Environment variable storage
- Rate limiting

### Data Security
- No persistent credential storage
- Input validation on search queries
- GDPR-compliant data handling

### Code Security
- Dependency vulnerability scanning
- Input sanitization
- Safe model loading

---

**Last Updated**: February 2024
