# Contributing to Household Energy Analysis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Provide detailed description
- Include error messages and logs
- Specify environment details

### Feature Requests

- Open an issue with "enhancement" label
- Describe the proposed feature
- Explain the use case and benefits
- Provide implementation suggestions if possible

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 🔧 Development Setup

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/household-energy-analysis.git
cd household-energy-analysis

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .
```

### Development Dependencies

```bash
# Install additional development tools
pip install pytest pytest-cov black flake8 isort pre-commit
```

## 📝 Code Standards

### Python Style Guide

- Follow PEP 8
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md for significant changes
- Include inline comments for complex logic

### Testing

- Write unit tests for new functions
- Maintain >80% code coverage
- Use pytest framework
- Include integration tests for workflows

## 🚀 Commit Guidelines

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples

```
feat(models): add XGBoost regression model
fix(preprocessing): handle missing timestamps correctly
docs(readme): update installation instructions
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

### Test Structure

```
tests/
├── test_data_preprocessing.py
├── test_models.py
├── test_visualization.py
└── fixtures/
    └── sample_data.csv
```

## 📋 Pull Request Process

1. **Update Documentation**: Ensure README and docs are updated
2. **Add Tests**: Include tests for new functionality
3. **Code Review**: All PRs require review
4. **CI Checks**: Ensure all automated checks pass
5. **Merge**: Use "Squash and merge" for clean history

## 🏷️ Release Process

### Version Numbering

Follow Semantic Versioning (SemVer):

- `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

- [ ] Update version in `setup.py`
- [ ] Update CHANGELOG.md
- [ ] Create release tag
- [ ] Update documentation
- [ ] Publish to PyPI (if applicable)

## 💬 Communication

### Discussions

- Use GitHub Discussions for questions
- Join our community forum
- Participate in monthly video calls

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Follow open source best practices

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

### High Priority

- Real-time data streaming
- Additional ML models
- Performance optimizations
- Mobile app development

### Medium Priority

- Weather data integration
- Advanced visualizations
- API development
- Docker improvements

### Low Priority

- Code style improvements
- Documentation enhancements
- Example notebooks
- Tutorial content

## 🏆 Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes
- Project documentation
- Social media announcements

Thank you for helping make this project better! 🚀
