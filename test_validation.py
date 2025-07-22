"""Unit tests for validation utilities."""

import os
import tempfile
import unittest
from pathlib import Path

from validation import validate_folder_path, validate_limit, validate_threshold_range


class TestValidation(unittest.TestCase):
    """Test cases for validation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.rmdir(self.temp_dir)
            os.unlink(self.temp_file.name)
        except (OSError, FileNotFoundError):
            pass
    
    def test_validate_folder_path_valid(self):
        """Test validation of valid folder path."""
        is_valid, error = validate_folder_path(self.temp_dir)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_folder_path_empty(self):
        """Test validation of empty folder path."""
        is_valid, error = validate_folder_path("")
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", error)
    
    def test_validate_folder_path_whitespace(self):
        """Test validation of whitespace-only folder path."""
        is_valid, error = validate_folder_path("   ")
        self.assertFalse(is_valid)
        self.assertIn("whitespace", error)
    
    def test_validate_folder_path_nonexistent(self):
        """Test validation of non-existent folder path."""
        is_valid, error = validate_folder_path("/nonexistent/path")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", error)
    
    def test_validate_folder_path_file_not_dir(self):
        """Test validation when path points to file, not directory."""
        is_valid, error = validate_folder_path(self.temp_file.name)
        self.assertFalse(is_valid)
        self.assertIn("not a directory", error)
    
    def test_validate_threshold_range_valid(self):
        """Test validation of valid threshold range."""
        is_valid, error = validate_threshold_range(0.0, 100.0)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        is_valid, error = validate_threshold_range(10.5, 50.5)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_threshold_range_invalid_order(self):
        """Test validation when min > max."""
        is_valid, error = validate_threshold_range(50.0, 10.0)
        self.assertFalse(is_valid)
        self.assertIn("cannot be greater than", error)
    
    def test_validate_threshold_range_negative(self):
        """Test validation of negative thresholds."""
        is_valid, error = validate_threshold_range(-1.0, 50.0)
        self.assertFalse(is_valid)
        self.assertIn("cannot be negative", error)
    
    def test_validate_threshold_range_too_high(self):
        """Test validation of thresholds > 100."""
        is_valid, error = validate_threshold_range(0.0, 150.0)
        self.assertFalse(is_valid)
        self.assertIn("cannot exceed 100", error)
    
    def test_validate_threshold_range_non_numeric(self):
        """Test validation of non-numeric thresholds."""
        is_valid, error = validate_threshold_range("invalid", 50.0)
        self.assertFalse(is_valid)
        self.assertIn("must be numeric", error)
    
    def test_validate_limit_valid(self):
        """Test validation of valid limits."""
        is_valid, error = validate_limit(10)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        is_valid, error = validate_limit(1)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        is_valid, error = validate_limit(1000)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_limit_zero_or_negative(self):
        """Test validation of zero or negative limits."""
        is_valid, error = validate_limit(0)
        self.assertFalse(is_valid)
        self.assertIn("must be greater than 0", error)
        
        is_valid, error = validate_limit(-5)
        self.assertFalse(is_valid)
        self.assertIn("must be greater than 0", error)
    
    def test_validate_limit_too_high(self):
        """Test validation of limits > 1000."""
        is_valid, error = validate_limit(1001)
        self.assertFalse(is_valid)
        self.assertIn("cannot exceed 1000", error)
    
    def test_validate_limit_non_numeric(self):
        """Test validation of non-numeric limits."""
        is_valid, error = validate_limit("invalid")
        self.assertFalse(is_valid)
        self.assertIn("must be a numeric", error)


if __name__ == '__main__':
    unittest.main()