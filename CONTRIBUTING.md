# PulseCheck AI - Contributing Guide

Thank you for your interest in contributing to PulseCheck AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Getting Started

### 1. Fork the Repository
```bash
git clone https://github.com/yourusername/PulseCheckAI.git
cd PulseCheckAI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes
- Follow PEP 8 code style
- Add type hints to functions
- Include docstrings
- Keep functions focused and small

### 3. Format Your Code
```bash
black src/ tests/ app.py
flake8 src/ tests/ app.py
mypy src/ tests/ app.py
```

### 4. Write Tests
Every new feature should include tests:
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

Aim for >80% code coverage.

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature"
# or
git commit -m "fix: resolve issue with X"
git commit -m "docs: update README"
git commit -m "refactor: improve performance of Y"
git commit -m "test: add tests for feature Z"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Related issue numbers (if applicable)
- Screenshots/gifs for UI changes

## Code Style Guide

### Python
- Follow PEP 8
- Use 4 spaces for indentation
- Max line length: 100 characters
- Use type hints for all functions

```python
def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Analyze sentiment of text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (sentiment_label, confidence_score)
    """
    pass
```

### Documentation
- Write clear docstrings
- Include examples where helpful
- Keep README updated
- Document breaking changes

### Commits
- Use meaningful commit messages
- Reference issues when applicable
- Keep commits focused and logical

## Pull Request Guidelines

### Before Submitting
- [ ] Code is formatted with black
- [ ] Linting passes (flake8)
- [ ] Type hints pass (mypy)
- [ ] Tests pass and cover new code
- [ ] No merge conflicts
- [ ] README updated if needed

### PR Description Template
```markdown
## Description
Briefly describe the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Fixes #(issue number)

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass
- [ ] Code is formatted
- [ ] Documentation updated
```

## Common Contributions

### Bug Reports
Include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment (Python version, OS)
- Error messages/logs

### Feature Requests
Include:
- Use case
- Expected behavior
- Proposed implementation (optional)
- Benefits

### Documentation
- Fix typos
- Clarify instructions
- Add examples
- Improve organization

## Testing Guidelines

### Unit Tests
```python
def test_sentiment_analysis():
    analyzer = SentimentAnalyzer()
    sentiment, score = analyzer.analyze_sentiment("Great!")
    
    assert sentiment == "positive"
    assert 0 <= score <= 1
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_sentiment_analyzer.py -v

# Specific test
pytest tests/test_sentiment_analyzer.py::TestSentimentAnalyzer::test_initialization -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance

- Profile code with large datasets
- Consider memory usage for models
- Use caching where appropriate
- Document performance impact

## Documentation

- Update docstrings
- Add type hints
- Update README if public API changes
- Add examples for new features

## Areas for Contribution

### Easy (Good for Beginners)
- Documentation improvements
- Bug fixes
- Unit tests
- Code cleanup

### Medium
- New visualization types
- Optimization improvements
- Feature enhancements
- Error handling

### Advanced
- New NLP models
- Performance optimization
- Architecture improvements
- Advanced analytics features

## Questions?

- Open a [Discussion](https://github.com/yourusername/PulseCheckAI/discussions)
- Check existing issues
- Review documentation
- Ask in comments

---

**Thank you for contributing! 🎉**
