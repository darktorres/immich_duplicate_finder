"""Unit tests for local media utilities."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from local_media import bytes_to_megabytes, get_media_files, SUPPORTED_IMAGE_EXTENSIONS


class TestLocalMedia(unittest.TestCase):
    """Test cases for local media functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create some test files
        self.test_files = []
        for ext in ['.jpg', '.png', '.txt', '.JPG']:  # Mix of supported and unsupported
            test_file = Path(self.temp_dir) / f"test{ext}"
            test_file.touch()
            self.test_files.append(str(test_file))
    
    def tearDown(self):
        """Clean up test fixtures."""
        for file_path in self.test_files:
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass
    
    def test_bytes_to_megabytes_valid(self):
        """Test conversion of bytes to megabytes."""
        result = bytes_to_megabytes(1048576)  # 1 MB
        self.assertEqual(result, "1.000 MB")
        
        result = bytes_to_megabytes(2097152)  # 2 MB
        self.assertEqual(result, "2.000 MB")
        
        result = bytes_to_megabytes(1536)  # 1.5 KB
        self.assertEqual(result, "0.001 MB")
    
    def test_bytes_to_megabytes_none(self):
        """Test conversion when bytes is None."""
        result = bytes_to_megabytes(None)
        self.assertEqual(result, "0.000 MB")
    
    def test_bytes_to_megabytes_zero(self):
        """Test conversion of zero bytes."""
        result = bytes_to_megabytes(0)
        self.assertEqual(result, "0.000 MB")
    
    def test_get_media_files_valid_folder(self):
        """Test getting media files from valid folder."""
        media_files = get_media_files(self.temp_dir)
        
        # Should find .jpg, .png, and .JPG files (case insensitive)
        expected_count = 3
        self.assertEqual(len(media_files), expected_count)
        
        # Check that all returned files have supported extensions
        for file_path in media_files:
            file_ext = Path(file_path).suffix.lower()
            self.assertIn(file_ext, SUPPORTED_IMAGE_EXTENSIONS)
    
    def test_get_media_files_empty_folder(self):
        """Test getting media files from empty folder."""
        empty_dir = tempfile.mkdtemp()
        try:
            media_files = get_media_files(empty_dir)
            self.assertEqual(len(media_files), 0)
        finally:
            os.rmdir(empty_dir)
    
    def test_get_media_files_nonexistent_folder(self):
        """Test getting media files from non-existent folder."""
        media_files = get_media_files("/nonexistent/path")
        self.assertEqual(len(media_files), 0)
    
    def test_get_media_files_empty_path(self):
        """Test getting media files with empty path."""
        media_files = get_media_files("")
        self.assertEqual(len(media_files), 0)
    
    @patch('local_media.os.walk')
    def test_get_media_files_permission_error(self, mock_walk):
        """Test handling of permission errors."""
        mock_walk.side_effect = PermissionError("Access denied")
        
        media_files = get_media_files(self.temp_dir)
        self.assertEqual(len(media_files), 0)
    
    def test_supported_extensions_case_insensitive(self):
        """Test that file extension matching is case insensitive."""
        # Create files with different cases
        test_dir = tempfile.mkdtemp()
        try:
            test_files = []
            for ext in ['.jpg', '.JPG', '.Jpg', '.jPg']:
                test_file = Path(test_dir) / f"test{ext}"
                test_file.touch()
                test_files.append(test_file)
            
            media_files = get_media_files(test_dir)
            self.assertEqual(len(media_files), 4)  # All should be found
            
        finally:
            for test_file in test_files:
                try:
                    test_file.unlink()
                except FileNotFoundError:
                    pass
            try:
                os.rmdir(test_dir)
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()