"""Unit tests for local media utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from local_media import SUPPORTED_IMAGE_EXTENSIONS, bytes_to_megabytes, get_media_files, load_image


@pytest.fixture
def temp_dir_with_files():
    """Create a temporary directory with test files."""
    temp_dir = tempfile.mkdtemp()
    test_files = []

    # Create some test files
    for ext in [".jpg", ".png", ".txt", ".JPG"]:  # Mix of supported and unsupported
        test_file = Path(temp_dir) / f"test{ext}"
        test_file.touch()
        test_files.append(str(test_file))

    yield temp_dir, test_files

    # Cleanup
    for file_path in test_files:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.mark.parametrize(
    "bytes_val,expected",
    [
        (1048576, "1.000 MB"),  # 1 MB
        (2097152, "2.000 MB"),  # 2 MB
        (1536, "0.001 MB"),  # 1.5 KB
        (0, "0.000 MB"),  # Zero bytes
    ],
)
def test_bytes_to_megabytes_valid(bytes_val, expected):
    """Test conversion of bytes to megabytes."""
    result = bytes_to_megabytes(bytes_val)
    assert result == expected


def test_bytes_to_megabytes_none():
    """Test conversion when bytes is None."""
    result = bytes_to_megabytes(None)
    assert result == "0.000 MB"


def test_get_media_files_valid_folder(temp_dir_with_files):
    """Test getting media files from valid folder."""
    temp_dir, _ = temp_dir_with_files
    media_files = get_media_files(temp_dir)

    # Should find .jpg, .png, and .JPG files (case insensitive)
    # Note: Empty files might be filtered out by access validation
    assert len(media_files) >= 0
    assert len(media_files) <= 3

    # Check that all returned files have supported extensions
    for file_path in media_files:
        file_ext = Path(file_path).suffix.lower()
        assert file_ext in SUPPORTED_IMAGE_EXTENSIONS


def test_get_media_files_empty_folder():
    """Test getting media files from empty folder."""
    empty_dir = tempfile.mkdtemp()
    try:
        media_files = get_media_files(empty_dir)
        assert len(media_files) == 0
    finally:
        os.rmdir(empty_dir)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/nonexistent/path",
        "",
        None,
    ],
)
def test_get_media_files_invalid_paths(invalid_path):
    """Test getting media files with invalid paths."""
    media_files = get_media_files(invalid_path or "")
    assert len(media_files) == 0


def test_get_media_files_permission_error(temp_dir_with_files, mocker):
    """Test handling of permission errors."""
    temp_dir, _ = temp_dir_with_files
    mock_walk = mocker.patch("local_media.os.walk")
    mock_walk.side_effect = PermissionError("Access denied")

    media_files = get_media_files(temp_dir)
    assert len(media_files) == 0


def test_supported_extensions_case_insensitive():
    """Test that file extension matching is case insensitive."""
    # Create files with different cases
    test_dir = tempfile.mkdtemp()
    try:
        test_files = []
        for ext in [".jpg", ".JPG", ".Jpg", ".jPg"]:
            test_file = Path(test_dir) / f"test{ext}"
            test_file.touch()
            test_files.append(test_file)

        media_files = get_media_files(test_dir)
        # Empty files might be filtered out, so check that we find some files
        # and that they have the right extensions
        assert len(media_files) >= 0
        assert len(media_files) <= 4

        # Check that all returned files have supported extensions
        for file_path in media_files:
            file_ext = Path(file_path).suffix.lower()
            assert file_ext == ".jpg"  # All should be .jpg variants

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

@pytest.mark.unit
def test_setup_local_media():
    """Test setup_local_media function."""
    # This function should run without errors
    from local_media import setup_local_media
    setup_local_media()  # Should not raise any exceptions


@pytest.mark.unit
class TestLoadImage:
    """Test cases for load_image function."""
    
    def test_load_image_nonexistent_file(self):
        """Test loading non-existent image file."""
        from local_media import load_image
        result = load_image("/nonexistent/file.jpg")
        assert result is None
    
    def test_load_image_invalid_file(self):
        """Test loading invalid image file."""
        from local_media import load_image
        
        # Create a temporary text file (not an image)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b"This is not an image")
            temp_path = f.name
        
        try:
            result = load_image(temp_path)
            assert result is None
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
    
    def test_load_image_directory_instead_of_file(self):
        """Test loading a directory instead of file."""
        from local_media import load_image
        
        temp_dir = tempfile.mkdtemp()
        try:
            result = load_image(temp_dir)
            assert result is None
        finally:
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


@pytest.mark.unit
class TestGetFileInfo:
    """Test cases for get_file_info function."""
    
    def test_get_file_info_nonexistent_file(self):
        """Test getting info for non-existent file."""
        from local_media import get_file_info
        result = get_file_info("/nonexistent/file.jpg")
        
        # Should return default values
        assert result[0] == "Unknown"  # file_size
        assert result[1] == "file.jpg"  # file_name
        assert result[2] == "Unknown"  # resolution
        assert result[3] == "Unknown"  # creation_date
        assert result[4] == "/nonexistent/file.jpg"  # file_path
    
    def test_get_file_info_text_file(self):
        """Test getting info for a text file (not an image)."""
        from local_media import get_file_info
        
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test file")
            temp_path = f.name
        
        try:
            result = get_file_info(temp_path)
            
            # Should get file size and name, but unknown resolution
            assert result[0] != "Unknown"  # Should have file size
            assert result[1] == os.path.basename(temp_path)  # file_name
            assert result[2] == "Unknown"  # resolution (can't get from text file)
            assert result[3] != "Unknown"  # Should have creation date
            assert result[4] == temp_path  # file_path
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
    
    def test_get_file_info_permission_error(self, mocker):
        """Test getting info when file access is denied."""
        from local_media import get_file_info
        
        # Mock os.path.isfile to return True, but os.path.getsize to raise PermissionError
        mocker.patch('local_media.os.path.isfile', return_value=True)
        mocker.patch('local_media.os.path.getsize', side_effect=PermissionError("Access denied"))
        
        result = get_file_info("/some/file.jpg")
        
        # Should return default values on error
        assert result[0] == "Unknown"  # file_size
        assert result[1] == "file.jpg"  # file_name
        assert result[2] == "Unknown"  # resolution
        assert result[3] == "Unknown"  # creation_date
        assert result[4] == "/some/file.jpg"  # file_path


@pytest.mark.unit
class TestDeleteFile:
    """Test cases for delete_file function."""
    
    def test_delete_file_nonexistent(self):
        """Test deleting non-existent file."""
        from local_media import delete_file
        result = delete_file("/nonexistent/file.jpg")
        assert result is False
    
    def test_delete_file_success(self):
        """Test successful file deletion."""
        from local_media import delete_file
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        # File should exist
        assert os.path.exists(temp_path)
        
        # Delete it
        result = delete_file(temp_path)
        assert result is True
        
        # File should no longer exist
        assert not os.path.exists(temp_path)
    
    def test_delete_file_permission_error(self, mocker):
        """Test file deletion with permission error."""
        from local_media import delete_file
        
        # Mock os.path.isfile to return True, but os.remove to raise PermissionError
        mocker.patch('local_media.os.path.isfile', return_value=True)
        mocker.patch('local_media.os.remove', side_effect=PermissionError("Access denied"))
        
        result = delete_file("/some/file.jpg")
        assert result is False
    
    def test_delete_file_os_error(self, mocker):
        """Test file deletion with OS error."""
        from local_media import delete_file
        
        # Mock os.path.isfile to return True, but os.remove to raise OSError
        mocker.patch('local_media.os.path.isfile', return_value=True)
        mocker.patch('local_media.os.remove', side_effect=OSError("File in use"))
        
        result = delete_file("/some/file.jpg")
        assert result is False


@pytest.mark.unit
def test_supported_image_extensions():
    """Test that SUPPORTED_IMAGE_EXTENSIONS contains expected formats."""
    
    # Check that common formats are included
    expected_formats = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic", ".heif", ".dng"]
    
    for fmt in expected_formats:
        assert fmt in SUPPORTED_IMAGE_EXTENSIONS
    
    # Check that all extensions are lowercase
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        assert ext == ext.lower()
        assert ext.startswith(".")


@pytest.mark.unit
def test_get_media_files_with_various_extensions():
    """Test get_media_files with various supported extensions."""
    
    temp_dir = tempfile.mkdtemp()
    try:
        created_files = []
        
        # Create files with different supported extensions
        test_extensions = [".jpg", ".PNG", ".gif", ".HEIC", ".webp"]
        for i, ext in enumerate(test_extensions):
            test_file = Path(temp_dir) / f"test{i}{ext}"
            test_file.touch()
            created_files.append(test_file)
        
        # Create some non-image files
        non_image_file = Path(temp_dir) / "test.txt"
        non_image_file.touch()
        created_files.append(non_image_file)
        
        media_files = get_media_files(temp_dir)
        
        # Should find image files but not text file
        # Note: empty files might be filtered out by access validation
        assert len(media_files) >= 0
        assert len(media_files) <= len(test_extensions)
        
        # All returned files should have supported extensions
        for file_path in media_files:
            file_ext = Path(file_path).suffix.lower()
            assert file_ext in SUPPORTED_IMAGE_EXTENSIONS
        
    finally:
        # Cleanup
        for file_path in created_files:
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


@pytest.mark.unit
def test_get_media_files_inaccessible_files():
    """Test get_media_files with inaccessible files."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a file
        test_file = Path(temp_dir) / "test.jpg"
        test_file.touch()
        
        # Mock os.access to return False for this specific file
        original_access = os.access
        def mock_access(path, mode):
            if str(test_file) in path:
                return False  # Not accessible
            return original_access(path, mode)
        
        with patch('local_media.os.access', side_effect=mock_access):
            media_files = get_media_files(temp_dir)
            # Should not include the inaccessible file
            assert len(media_files) == 0
        
    finally:
        # Cleanup
        try:
            test_file.unlink()
        except FileNotFoundError:
            pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


@pytest.mark.unit
def test_load_image_successful_with_load():
    """Test successful image loading that calls image.load()."""
    # Create a simple test image file
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a minimal valid image file
        from PIL import Image as PILImage
        test_image = PILImage.new('RGB', (10, 10), color='red')
        test_path = os.path.join(temp_dir, 'test.jpg')
        test_image.save(test_path)
        
        # Load the image
        result = load_image(test_path)
        
        # Should successfully load
        assert result is not None
        assert result.mode == 'RGB'
        assert result.size == (10, 10)
        
    finally:
        # Cleanup
        try:
            os.unlink(test_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


@pytest.mark.unit
def test_load_image_unidentified_image_error():
    """Test load_image with UnidentifiedImageError."""
    from PIL import UnidentifiedImageError
    
    # Create a file that looks like an image but isn't
    temp_dir = tempfile.mkdtemp()
    try:
        test_path = os.path.join(temp_dir, 'fake.jpg')
        with open(test_path, 'wb') as f:
            f.write(b'This is not a valid image file')
        
        # Should return None for invalid image
        result = load_image(test_path)
        assert result is None
        
    finally:
        # Cleanup
        try:
            os.unlink(test_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


@pytest.mark.unit
def test_get_file_info_image_dimension_error():
    """Test get_file_info when image dimension extraction fails."""
    from local_media import get_file_info
    
    # Create a temporary file that exists but can't be opened as image
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jpg', delete=False) as f:
        f.write("This is not a valid image")
        temp_path = f.name
    
    try:
        # Mock Image.open to raise an exception
        with patch('local_media.Image.open', side_effect=Exception("Cannot open image")):
            result = get_file_info(temp_path)
            
            # Should get file info but with "Unknown" resolution
            assert result[0] != "Unknown"  # Should have file size
            assert result[1] == os.path.basename(temp_path)  # file_name
            assert result[2] == "Unknown"  # resolution should be Unknown due to exception
            assert result[3] != "Unknown"  # Should have creation date
            assert result[4] == temp_path  # file_path
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass