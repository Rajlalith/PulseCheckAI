# PulseCheck AI - Troubleshooting Guide

## Common Issues and Solutions

### 1. Installation Issues

#### Problem: "ModuleNotFoundError" when importing transformers
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution**:
```bash
pip install transformers torch -U
# or
pip install -r requirements.txt
```

#### Problem: "SSL: CERTIFICATE_VERIFY_FAILED" when downloading models
**Solution**:
```bash
# On macOS (one-time)
/Applications/Python\ 3.x/Install\ Certificates.command

# Or use environment variable
export SSL_NO_VERIFY=False
```

---

### 2. Model Download Issues

#### Problem: "Unable to connect to model hub"
**Solution**:
```bash
# Pre-download models manually
python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')"
```

#### Problem: Models not caching properly
**Solution**:
```bash
# Set custom cache directory
export HF_HOME=/path/to/cache
# or in Python
import os
os.environ['HF_HOME'] = '/custom/path'
```

---

### 3. Memory Issues

#### Problem: "RuntimeError: CUDA out of memory"
**Solution**:
```python
# Force CPU usage
import torch
torch.cuda.is_available = lambda: False

# Or in .env
USE_GPU=false
```

#### Problem: "Process killed" or "Out of memory"
**Solution**:
```bash
# Reduce batch size in code
max_results = 50  # instead of 100

# Increase system swap
# On Linux: sudo fallocate -l 4G /swapfile
```

---

### 4. Twitter API Issues

#### Problem: "Invalid Bearer Token"
```
403 Forbidden: Invalid Bearer Token
```

**Solution**:
1. Verify token in [Twitter Developer Portal](https://developer.twitter.com)
2. Ensure Bearer Token is active
3. Add `Bearer ` prefix is handled by code
4. Check token hasn't expired

```python
# Test token
import requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("https://api.twitter.com/2/tweets/search/recent", headers=headers)
print(response.status_code)
```

#### Problem: Rate limit exceeded
```
429: Too Many Requests
```

**Solution**:
```bash
# Wait 15 minutes for rate limit reset
# Use sample data instead
# Or use smaller batch sizes
```

#### Problem: No tweets found
**Solution**:
```bash
# Try different search query
"python OR AI OR machine learning"  # Boolean operators
"#AI"                               # Hashtag only
"@username"                         # Username mention
"from:username"                     # From specific user

# Adjust date range
days_back = 7  # Maximum is 7
```

---

### 5. Streamlit Issues

#### Problem: "StreamlitAPIException: No elements to add"
**Solution**:
- Restart Streamlit: `Ctrl+C` then `streamlit run app.py`
- Clear cache: `streamlit cache clear`

#### Problem: Dashboard not updating
**Solution**:
```bash
# Hard refresh
streamlit run app.py --logger.level=debug

# Or clear browser cache (Ctrl+Shift+Delete)
```

#### Problem: Charts not rendering
**Solution**:
```python
# Update Plotly
pip install plotly -U

# Check browser compatibility
# Chrome 90+, Firefox 88+, Safari 14+
```

---

### 6. Sentiment Analysis Issues

#### Problem: All sentiments classified as "Neutral"
**Solution**:
```python
# Check text quality
# Model works best with:
# - Complete sentences
# - English text
# - 10-500 characters

# Test with known text
text = "This product is amazing!"
sentiment, score = analyzer.analyze_sentiment(text)
```

#### Problem: Low confidence scores
**Solution**:
```bash
# This is normal for ambiguous text
# Neutral text typically has lower confidence
# Consider setting minimum threshold:

if score < 0.5:
    sentiment = "neutral"  # Low confidence → neutral
```

---

### 7. Topic Extraction Issues

#### Problem: "No topics found"
**Solution**:
```python
# Ensure adequate text
minimum_texts = 2  # Need at least 2 documents for TF-IDF

# Check text preprocessing
cleaned = modeler.preprocess_text(text)
print(f"Cleaned text: {cleaned}")

# Ensure topics exist after stopword removal
if len(text.split()) < 5:
    # Text too short for topic extraction
    pass
```

#### Problem: "Irrelevant keywords extracted"
**Solution**:
```python
# Adjust stopwords or TF-IDF parameters
vectorizer = TfidfVectorizer(
    max_features=100,      # Increase for more topics
    min_df=2,              # Increase to filter noise
    max_df=0.8,            # Adjust for common words
    ngram_range=(1, 2)     # Use bigrams
)
```

---

### 8. Data Export Issues

#### Problem: "File download failed"
**Solution**:
```bash
# Check file permissions
# Ensure download folder is writable
# Try different browser

# Or manually export from DataFrame
df.to_csv("export.csv", index=False)
```

#### Problem: "JSON export contains NaN values"
**Solution**:
```python
# Handle NaN before export
df = df.fillna("null")
df.to_json("export.json", orient="records")
```

---

### 9. Performance Issues

#### Problem: "Analysis taking too long"
**Solution**:
```bash
# Use fewer tweets
max_results = 50  # instead of 500

# Use GPU if available
# Check with: python -c "import torch; print(torch.cuda.is_available())"

# Pre-cache models
# First run trains cache, subsequent runs are faster
```

#### Problem: "High CPU usage"
**Solution**:
```bash
# Reduce batch size
batch_size = 16  # instead of 32

# Process in chunks
# Reduce concurrent operations
```

---

### 10. Environment Issues

#### Problem: "Command not found: python3"
**Solution**:
```bash
# Check Python installation
which python
python --version

# Create alias if needed
alias python3=python
```

#### Problem: ".env file not being read"
**Solution**:
```bash
# Ensure file is in root directory
ls -la .env

# Check format
export TWITTER_BEARER_TOKEN=your_token

# Or in Python:
from dotenv import load_dotenv
load_dotenv()  # Loads from .env
```

---

### 11. NLTK Data Issues

#### Problem: "LookupError: Resource punkt not found"
**Solution**:
```bash
python -c "import nltk; nltk.download('punkt')"
# or
python -c "import nltk; nltk.download('stopwords')"
```

#### Problem: "Error unpacking NLTK data"
**Solution**:
```bash
# Remove and reinstall
rm -rf ~/nltk_data
python -c "import nltk; nltk.download('all')"
```

---

### 12. Database/Storage Issues

#### Problem: "CSV contains special characters"
**Solution**:
```python
# Specify encoding
df.to_csv("export.csv", index=False, encoding='utf-8')
```

#### Problem: "Data not persisting"
**Solution**:
```python
# Note: PulseCheck AI is stateless by design
# Store data manually if needed:
import pickle
pickle.dump(df, open("data.pkl", "wb"))
```

---

### 13. Version Compatibility

#### Problem: "Incompatible with Python 3.8"
**Solution**:
```bash
# PulseCheck AI requires Python 3.9+
python --version

# Update Python if needed
# macOS: brew install python@3.10
# Windows: Download from python.org
```

#### Problem: "Transformers version incompatible"
**Solution**:
```bash
# Pin versions to requirements.txt
pip install transformers==4.36.2 torch==2.1.2
```

---

## Debugging Tips

### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Check Model Status
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
```

### 3. Test Individual Components
```python
# Test collector
from src.twitter_collector import TwitterCollector
collector = TwitterCollector()
df = collector.search_sample_data("test", n_samples=10)

# Test analyzer
from src.sentiment_analyzer import SentimentAnalyzer
analyzer = SentimentAnalyzer()
sentiment, score = analyzer.analyze_sentiment("Test text")

# Test modeler
from src.topic_modeler import TopicModeler
modeler = TopicModeler()
keywords = modeler.extract_keywords("Test text")
```

### 4. Monitor Resource Usage
```bash
# macOS/Linux
top

# Windows
tasklist

# Or use Python
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"CPU: {psutil.cpu_percent()}%")
```

---

## Getting Help

1. **Check Documentation**: See README.md and ARCHITECTURE.md
2. **Search Issues**: GitHub issues might have solutions
3. **Open Issue**: Provide:
   - Error message/traceback
   - Steps to reproduce
   - Environment (Python version, OS)
   - Code snippet

4. **Contact**: Email or create discussion

---

**Last Updated**: February 2024
