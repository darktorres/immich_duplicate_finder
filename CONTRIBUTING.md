# Contributing to Duplicate Finder

Thank you for your interest in contributing to the Duplicate Finder project! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites
- Python 3.12 or higher
- Poetry
- Git

### Setting up the Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/local_duplicate_finder.git
   cd local_duplicate_finder
   ```

2. **Install dependencies**
   ```bash
   poetry install
   poetry shell
   ```

3. **Install development tools**
   ```bash
   # Using Poetry
   poetry run pre-commit install
   
   # Using Make
   make dev
   ```

## Development Workflow

### Running the Application
```bash
# Using Poetry
poetry run streamlit run app.py

# Using Make
make run
```

### Running Tests
```bash
# Run all tests
make test
# or
poetry run pytest -v

# Run tests with coverage
make test-cov
# or
poetry run pytest --cov=. --cov-report=html -v

# Run tests in watch mode
make test-watch
# or
poetry run pytest -f -v

# Run specific test file
pytest test_validation.py -v

# Run tests with specific markers
pytest -m unit -v          # Run only unit tests
pytest -m "not slow" -v    # Skip slow tests
```

### Code Quality

#### Linting and Formatting
```bash
# Check and fix linting issues
make lint
# or
poetry run ruff check --fix .

# Format code
make format
# or
poetry run ruff format .

# Check without making changes
make check
# or
poetry run ruff check . && poetry run ruff format --diff .
```

#### Pre-commit Hooks
We use pre-commit hooks to ensure code quality. They will run automatically on commit if installed:

```bash
poetry run pre-commit install
```

## Testing Guidelines

### Test Structure
- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test interactions between components
- **Use pytest fixtures**: For setup and teardown of test data
- **Use parametrized tests**: For testing multiple scenarios

### Writing Tests
1. **File naming**: Test files should be named `test_*.py`
2. **Function naming**: Test functions should be named `test_*`
3. **Use descriptive names**: Test names should clearly describe what is being tested
4. **Use fixtures**: For common setup and teardown
5. **Mock external dependencies**: Use `pytest-mock` for mocking

### Example Test
```python
import pytest
from unittest.mock import patch

def test_function_with_valid_input():
    """Test function behavior with valid input."""
    result = my_function("valid_input")
    assert result == "expected_output"

@pytest.mark.parametrize("input_val,expected", [
    ("input1", "output1"),
    ("input2", "output2"),
])
def test_function_with_multiple_inputs(input_val, expected):
    """Test function with multiple input scenarios."""
    result = my_function(input_val)
    assert result == expected

def test_function_with_mock(mocker):
    """Test function with mocked dependencies."""
    mock_dependency = mocker.patch('module.dependency')
    mock_dependency.return_value = "mocked_value"
    
    result = my_function()
    assert result == "expected_result"
    mock_dependency.assert_called_once()
```

## Code Style Guidelines

### Python Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all functions and classes
- Use meaningful variable and function names
- Keep functions small and focused

### Documentation
- Use Google-style docstrings
- Include parameter types and descriptions
- Include return type and description
- Include usage examples for complex functions

### Example Function
```python
def process_image(file_path: str, quality: float = 0.8) -> Optional[Image.Image]:
    """
    Process an image file and return a PIL Image object.
    
    Args:
        file_path: Path to the image file to process
        quality: Image quality factor (0.0 to 1.0)
        
    Returns:
        PIL Image object if successful, None if processing fails
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If quality is not between 0.0 and 1.0
        
    Example:
        >>> image = process_image("/path/to/image.jpg", quality=0.9)
        >>> if image:
        ...     print(f"Image size: {image.size}")
    """
    if not 0.0 <= quality <= 1.0:
        raise ValueError("Quality must be between 0.0 and 1.0")
    
    # Implementation here
    pass
```

## Submitting Changes

### Pull Request Process
1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   make ci-test  # Run the same tests as CI
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all CI checks pass

### Commit Message Format
We follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `style:` for formatting changes
- `chore:` for maintenance tasks

## Issue Reporting

### Bug Reports
When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages (if any)
- Screenshots (if applicable)

### Feature Requests
When requesting features, please include:
- Clear description of the feature
- Use case and motivation
- Possible implementation approach
- Any relevant examples or mockups

## Getting Help

- **Documentation**: Check the README.md and code comments
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to create a welcoming environment for all contributors.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉