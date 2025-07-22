"""Unit tests for validation utilities."""

import os
import tempfile
import pytest
from pathlib import Path

from validation import validate_folder_path, validate_limit, validate_threshold_range


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        os.rmdir(temp_dir)
    except (OSError, FileNotFoundError):
        pass


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    yield temp_file.name
    try:
        os.unlink(temp_file.name)
    except (OSError, FileNotFoundError):
        pass
    
def test_validate_folder_path_valid(temp_dir):
    """Test validation of valid folder path."""
    is_valid, error = validate_folder_path(temp_dir)
    assert is_valid is True
    assert error is None


def test_validate_folder_path_empty():
    """Test validation of empty folder path."""
    is_valid, error = validate_folder_path("")
    assert is_valid is False
    assert "cannot be empty" in error


def test_validate_folder_path_whitespace():
    """Test validation of whitespace-only folder path."""
    is_valid, error = validate_folder_path("   ")
    assert is_valid is False
    assert "whitespace" in error


def test_validate_folder_path_nonexistent():
    """Test validation of non-existent folder path."""
    is_valid, error = validate_folder_path("/nonexistent/path")
    assert is_valid is False
    assert "does not exist" in error


def test_validate_folder_path_file_not_dir(temp_file):
    """Test validation when path points to file, not directory."""
    is_valid, error = validate_folder_path(temp_file)
    assert is_valid is False
    assert "not a directory" in error
    
@pytest.mark.parametrize("min_val,max_val", [
    (0.0, 100.0),
    (10.5, 50.5),
    (25.0, 75.0),
])
def test_validate_threshold_range_valid(min_val, max_val):
    """Test validation of valid threshold ranges."""
    is_valid, error = validate_threshold_range(min_val, max_val)
    assert is_valid is True
    assert error is None


def test_validate_threshold_range_invalid_order():
    """Test validation when min > max."""
    is_valid, error = validate_threshold_range(50.0, 10.0)
    assert is_valid is False
    assert "cannot be greater than" in error


def test_validate_threshold_range_negative():
    """Test validation of negative thresholds."""
    is_valid, error = validate_threshold_range(-1.0, 50.0)
    assert is_valid is False
    assert "cannot be negative" in error


def test_validate_threshold_range_too_high():
    """Test validation of thresholds > 100."""
    is_valid, error = validate_threshold_range(0.0, 150.0)
    assert is_valid is False
    assert "cannot exceed 100" in error


def test_validate_threshold_range_non_numeric():
    """Test validation of non-numeric thresholds."""
    is_valid, error = validate_threshold_range("invalid", 50.0)
    assert is_valid is False
    assert "must be numeric" in error
    
@pytest.mark.parametrize("limit_val", [1, 10, 100, 500, 1000])
def test_validate_limit_valid(limit_val):
    """Test validation of valid limits."""
    is_valid, error = validate_limit(limit_val)
    assert is_valid is True
    assert error is None


@pytest.mark.parametrize("limit_val", [0, -1, -5, -100])
def test_validate_limit_zero_or_negative(limit_val):
    """Test validation of zero or negative limits."""
    is_valid, error = validate_limit(limit_val)
    assert is_valid is False
    assert "must be greater than 0" in error


def test_validate_limit_too_high():
    """Test validation of limits > 1000."""
    is_valid, error = validate_limit(1001)
    assert is_valid is False
    assert "cannot exceed 1000" in error


def test_validate_limit_non_numeric():
    """Test validation of non-numeric limits."""
    is_valid, error = validate_limit("invalid")
    assert is_valid is False
    assert "must be a numeric" in error