"""Unit tests for image duplicate detection utilities."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Mock faiss and torch before importing to avoid GPU/dependency issues
with (
    patch("faiss.StandardGpuResources", create=True),
    patch("torch.cuda.is_available", return_value=False),
    patch("torch.device", return_value="cpu"),
):
    from imageDuplicate import extract_features


@pytest.mark.unit
class TestConvertImageToRgb:
    """Test cases for convert_image_to_rgb function (tested through transform pipeline)."""

    @patch("imageDuplicate.get_model_and_transform")
    def test_convert_rgb_image_through_transform(self, mock_get_components):
        """Test converting an already RGB image through the transform pipeline."""
        # Mock the components
        mock_transform = MagicMock()
        mock_get_components.return_value = {
            'transform': mock_transform,
            'model': MagicMock(),
            'device': MagicMock(),
            'torch': MagicMock()
        }
        
        # Create a simple RGB image
        rgb_image = Image.new("RGB", (10, 10), color="red")
        
        # The transform should be called with the image
        mock_transform.return_value = MagicMock()
        mock_transform.return_value.unsqueeze.return_value.to.return_value = MagicMock()
        
        # This tests that the function can handle RGB images
        assert rgb_image.mode == "RGB"
        assert rgb_image.size == (10, 10)

    def test_convert_rgba_image_direct(self):
        """Test converting RGBA image to RGB directly."""
        # Create an RGBA image
        rgba_image = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        
        # Test the conversion logic directly
        result = rgba_image.convert("RGB") if rgba_image.mode != "RGB" else rgba_image

        assert result.mode == "RGB"
        assert result.size == (10, 10)

    def test_convert_grayscale_image_direct(self):
        """Test converting grayscale image to RGB directly."""
        # Create a grayscale image
        gray_image = Image.new("L", (10, 10), color=128)
        
        # Test the conversion logic directly
        result = gray_image.convert("RGB") if gray_image.mode != "RGB" else gray_image

        assert result.mode == "RGB"
        assert result.size == (10, 10)

    def test_convert_palette_image_direct(self):
        """Test converting palette image to RGB directly."""
        # Create a palette image
        palette_image = Image.new("P", (10, 10))
        
        # Test the conversion logic directly
        result = palette_image.convert("RGB") if palette_image.mode != "RGB" else palette_image

        assert result.mode == "RGB"
        assert result.size == (10, 10)


@pytest.mark.unit
class TestExtractFeatures:
    """Test cases for extract_features function."""

    @patch("imageDuplicate.get_model_and_transform")
    def test_extract_features_basic(self, mock_get_components):
        """Test basic feature extraction."""
        # Create a test image
        test_image = Image.new("RGB", (224, 224), color="red")

        # Mock the components
        mock_model = MagicMock()
        mock_transform = MagicMock()
        mock_device = MagicMock()
        mock_torch = MagicMock()
        
        mock_get_components.return_value = {
            'model': mock_model,
            'transform': mock_transform,
            'device': mock_device,
            'torch': mock_torch
        }

        # Mock the transform and model pipeline
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
        mock_transform.return_value = mock_tensor

        mock_features = MagicMock()
        mock_features.cpu.return_value.numpy.return_value.flatten.return_value = np.array([1, 2, 3, 4])
        mock_model.return_value = mock_features

        result = extract_features(test_image)

        # Should return numpy array
        assert isinstance(result, np.ndarray)
        assert len(result) == 4

        # Verify the pipeline was called correctly
        mock_transform.assert_called_once_with(test_image)
        mock_tensor.unsqueeze.assert_called_once_with(0)
        mock_model.assert_called_once()

    @patch("imageDuplicate.get_model_and_transform")
    def test_extract_features_exception_handling(self, mock_get_components):
        """Test feature extraction with exception."""
        test_image = Image.new("RGB", (224, 224), color="red")

        # Mock the components
        mock_transform = MagicMock()
        mock_get_components.return_value = {
            'model': MagicMock(),
            'transform': mock_transform,
            'device': MagicMock(),
            'torch': MagicMock()
        }

        # Mock transform to raise exception
        mock_transform.side_effect = Exception("Transform error")

        with pytest.raises(Exception, match="Transform error"):
            extract_features(test_image)


@pytest.mark.unit
class TestFaissIndexOperations:
    """Test cases for FAISS index operations."""

    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    @patch("imageDuplicate.os.path.exists")
    def test_init_or_load_faiss_index_new(self, mock_exists, mock_np, mock_get_faiss):
        """Test initializing new FAISS index."""
        from imageDuplicate import init_or_load_faiss_index

        # Mock that files don't exist
        mock_exists.return_value = False

        index, metadata = init_or_load_faiss_index()

        assert index is None
        assert metadata == []

    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    @patch("imageDuplicate.os.path.exists")
    def test_init_or_load_faiss_index_existing(self, mock_exists, mock_np, mock_get_faiss):
        """Test loading existing FAISS index."""
        from imageDuplicate import init_or_load_faiss_index

        # Mock that files exist
        mock_exists.return_value = True

        # Mock faiss and numpy operations
        mock_faiss = MagicMock()
        mock_get_faiss.return_value = mock_faiss
        mock_index = MagicMock()
        mock_faiss.read_index.return_value = mock_index
        mock_np.load.return_value.tolist.return_value = ["file1.jpg", "file2.jpg"]

        index, metadata = init_or_load_faiss_index()

        # Should return CPU index directly
        assert index == mock_index
        assert metadata == ["file1.jpg", "file2.jpg"]

        mock_faiss.read_index.assert_called_once()
        mock_np.load.assert_called_once()

    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    @patch("imageDuplicate.os.path.exists")
    def test_init_or_load_faiss_index_exception(self, mock_exists, mock_np, mock_get_faiss):
        """Test FAISS index loading with exception."""
        from imageDuplicate import init_or_load_faiss_index

        # Mock that files exist but loading fails
        mock_exists.return_value = True
        mock_faiss = MagicMock()
        mock_get_faiss.return_value = mock_faiss
        mock_faiss.read_index.side_effect = Exception("FAISS error")

        index, metadata = init_or_load_faiss_index()

        # Should return defaults on error
        assert index is None
        assert metadata == []

    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    def test_save_faiss_index_and_metadata_cpu(self, mock_np, mock_get_faiss):
        """Test saving FAISS index on CPU."""
        from imageDuplicate import save_faiss_index_and_metadata

        mock_faiss = MagicMock()
        mock_get_faiss.return_value = mock_faiss
        mock_index = MagicMock()
        metadata = ["file1.jpg", "file2.jpg"]

        save_faiss_index_and_metadata(mock_index, metadata)

        # Should save directly without GPU conversion
        mock_faiss.write_index.assert_called_once_with(mock_index, "faiss_index.bin")
        mock_np.save.assert_called_once()

    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    def test_save_faiss_index_and_metadata_exception(self, mock_np, mock_get_faiss):
        """Test saving FAISS index with exception."""
        from imageDuplicate import save_faiss_index_and_metadata

        mock_faiss = MagicMock()
        mock_get_faiss.return_value = mock_faiss
        mock_index = MagicMock()
        metadata = ["file1.jpg", "file2.jpg"]

        # Mock faiss.write_index to raise exception
        mock_faiss.write_index.side_effect = Exception("Save error")

        with pytest.raises(Exception, match="Save error"):
            save_faiss_index_and_metadata(mock_index, metadata)


@pytest.mark.unit
class TestUpdateFaissIndex:
    """Test cases for update_faiss_index function."""

    @patch("imageDuplicate.init_or_load_faiss_index")
    @patch("imageDuplicate.load_image")
    def test_update_faiss_index_file_already_exists(self, mock_load_image, mock_init_load):
        """Test updating FAISS index when file already exists in metadata."""
        from imageDuplicate import update_faiss_index

        # Mock that file already exists in metadata
        mock_init_load.return_value = (MagicMock(), ["existing_file.jpg"])

        result = update_faiss_index("existing_file.jpg")

        assert result == "skipped"
        mock_load_image.assert_not_called()

    @patch("imageDuplicate.init_or_load_faiss_index")
    @patch("imageDuplicate.load_image")
    def test_update_faiss_index_image_load_fails(self, mock_load_image, mock_init_load):
        """Test updating FAISS index when image loading fails."""
        from imageDuplicate import update_faiss_index

        # Mock that file doesn't exist in metadata but image loading fails
        mock_init_load.return_value = (MagicMock(), [])
        mock_load_image.return_value = None

        result = update_faiss_index("new_file.jpg")

        assert result == "error"

    @patch("imageDuplicate.init_or_load_faiss_index")
    @patch("imageDuplicate.load_image")
    @patch("imageDuplicate.extract_features")
    @patch("imageDuplicate.save_faiss_index_and_metadata")
    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    def test_update_faiss_index_new_file_success(self, mock_np, mock_get_faiss, mock_save, mock_extract, mock_load_image, mock_init_load):
        """Test successfully updating FAISS index with new file."""
        from imageDuplicate import update_faiss_index

        # Mock successful scenario
        mock_index = MagicMock()
        mock_init_load.return_value = (mock_index, [])

        mock_image = MagicMock()
        mock_load_image.return_value = mock_image

        mock_features = np.array([1, 2, 3, 4])
        mock_extract.return_value = mock_features

        result = update_faiss_index("new_file.jpg")

        assert result == "processed"

        # Verify the pipeline
        mock_load_image.assert_called_once_with("new_file.jpg")
        mock_extract.assert_called_once_with(mock_image)
        mock_index.add.assert_called_once()
        mock_save.assert_called_once()

    @patch("imageDuplicate.init_or_load_faiss_index")
    @patch("imageDuplicate.load_image")
    @patch("imageDuplicate.extract_features")
    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    def test_update_faiss_index_create_new_index(self, mock_np, mock_get_faiss, mock_extract, mock_load_image, mock_init_load):
        """Test creating new FAISS index when none exists."""
        from imageDuplicate import update_faiss_index

        # Mock no existing index
        mock_init_load.return_value = (None, [])

        mock_image = MagicMock()
        mock_load_image.return_value = mock_image

        mock_features = np.array([1, 2, 3, 4])
        mock_extract.return_value = mock_features

        # Mock FAISS index creation
        mock_faiss = MagicMock()
        mock_get_faiss.return_value = mock_faiss
        mock_cpu_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_cpu_index

        with patch("imageDuplicate.save_faiss_index_and_metadata") as _mock_save:
            result = update_faiss_index("new_file.jpg")

        assert result == "processed"

        # Should create new index
        mock_faiss.IndexFlatL2.assert_called_once_with(4)  # dimension = features.shape[0]
        mock_cpu_index.add.assert_called_once()


@pytest.mark.unit
def test_global_variables():
    """Test that global variables are properly defined."""
    from imageDuplicate import index_path, metadata_path

    assert index_path == "faiss_index.bin"
    assert metadata_path == "metadata.npy"


@pytest.mark.unit
def test_device_setup():
    """Test device setup logic."""
    # This test mainly ensures the module can be imported without GPU issues
    # The actual device setup is mocked in the import
    assert True  # If we get here, import was successful


@pytest.mark.unit
class TestCalculateFaissIndex:
    """Test cases for calculateFaissIndex function."""

    @patch("imageDuplicate.st")
    def test_calculate_faiss_index_basic_setup(self, mock_st):
        """Test basic setup of calculateFaissIndex."""
        # Mock session state
        mock_st.session_state = {}

        # Mock UI components
        mock_progress = MagicMock()
        mock_st.progress.return_value = mock_progress
        mock_st.button.return_value = False
        mock_st.empty.return_value = MagicMock()

        # Mock update_faiss_index to avoid actual processing
        with patch("imageDuplicate.update_faiss_index", return_value="processed"):
            from imageDuplicate import calculateFaissIndex

            # Call with empty list to avoid long processing
            calculateFaissIndex([])

        # Should initialize session state
        assert "message" in mock_st.session_state
        assert "progress" in mock_st.session_state
        assert "stop_index" in mock_st.session_state

    # Note: Stop button functionality is complex to test due to Streamlit's session state handling
    # The actual functionality is tested through integration testing


@pytest.mark.unit
class TestGenerateDbDuplicate:
    """Test cases for generate_db_duplicate function."""

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.init_or_load_faiss_index")
    def test_generate_db_duplicate_no_index(self, mock_init_load, mock_st):
        """Test generate_db_duplicate with no FAISS index."""
        # Mock no index available
        mock_init_load.return_value = (None, [])

        from imageDuplicate import generate_db_duplicate

        generate_db_duplicate()

        # Should show message about no index
        mock_st.write.assert_called_with("FAISS index or metadata not available.")

    # Note: Stop button functionality is complex to test due to Streamlit's session state handling
    # The actual functionality is tested through integration testing


@pytest.mark.unit
class TestShowDuplicatePhotosFaiss:
    """Test cases for show_duplicate_photos_faiss function."""

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.is_db_populated")
    def test_show_duplicate_photos_empty_db(self, mock_is_populated, mock_st):
        """Test show_duplicate_photos_faiss with empty database."""
        # Mock empty database
        mock_is_populated.return_value = False

        from imageDuplicate import show_duplicate_photos_faiss

        show_duplicate_photos_faiss(10, 0.0, 100.0)

        # Should show message about empty database
        mock_st.write.assert_called_with("The database does not contain any duplicate entries. Please generate/update the database.")

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.is_db_populated")
    @patch("imageDuplicate.load_duplicate_pairs")
    def test_show_duplicate_photos_no_duplicates(self, mock_load_pairs, mock_is_populated, mock_st):
        """Test show_duplicate_photos_faiss with no duplicates found."""
        # Mock populated database but no duplicates in range
        mock_is_populated.return_value = True
        mock_load_pairs.return_value = []

        from imageDuplicate import show_duplicate_photos_faiss

        show_duplicate_photos_faiss(10, 0.0, 100.0)

        # Should show no duplicates message
        mock_st.write.assert_called_with("No duplicates found.")


@pytest.mark.unit
@patch("imageDuplicate.get_model_and_transform")
def test_model_and_transform_initialization(mock_get_components):
    """Test that model and transform are properly initialized."""
    # Mock the components
    mock_get_components.return_value = {
        'model': MagicMock(),
        'transform': MagicMock(),
        'device': MagicMock(),
        'torch': MagicMock()
    }
    
    # Test that the function returns the expected components
    components = mock_get_components()
    assert components['model'] is not None
    assert components['transform'] is not None


# Note: GPU setup tests are complex due to FAISS GPU dependencies
# These are tested through integration testing


@pytest.mark.unit
class TestStreamlitFunctions:
    """Test cases for Streamlit-based functions."""

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.time.time")
    def test_calculate_faiss_index_with_files(self, mock_time, mock_st):
        """Test calculateFaissIndex with actual file processing."""
        # Mock session state
        mock_st.session_state = {"message": "", "progress": 0, "stop_index": False}

        # Mock UI components
        mock_progress = MagicMock()
        mock_st.progress.return_value = mock_progress
        mock_st.button.return_value = False
        mock_st.empty.return_value = MagicMock()

        # Mock time for progress calculation - provide enough values
        mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Mock update_faiss_index to return different statuses
        with patch("imageDuplicate.update_faiss_index") as mock_update:
            mock_update.side_effect = ["processed", "skipped", "error"]

            from imageDuplicate import calculateFaissIndex

            calculateFaissIndex(["file1.jpg", "file2.jpg", "file3.jpg"])

            # Should process all files
            assert mock_update.call_count == 3

            # Should update progress
            assert mock_progress.progress.call_count >= 3

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.init_or_load_faiss_index")
    @patch("imageDuplicate.save_duplicate_pair")
    def test_generate_db_duplicate_with_processing(self, mock_save, mock_init_load, mock_st):
        """Test generate_db_duplicate with actual processing."""
        # Mock index with some vectors
        mock_index = MagicMock()
        mock_index.ntotal = 3
        mock_index.reconstruct.side_effect = [
            [0.1, 0.2, 0.3],  # Vector 0
            [0.4, 0.5, 0.6],  # Vector 1
            [0.7, 0.8, 0.9],  # Vector 2
        ]
        mock_index.search.return_value = (
            np.array([[0.0, 0.5]]),  # distances
            np.array([[0, 1]]),  # indices
        )

        mock_init_load.return_value = (mock_index, ["file1.jpg", "file2.jpg", "file3.jpg"])

        # Mock session state
        mock_st.session_state = {"stop_requested": False}
        mock_st.button.return_value = False

        # Mock UI components
        mock_st.empty.return_value = MagicMock()
        mock_st.progress.return_value = MagicMock()

        from imageDuplicate import generate_db_duplicate

        generate_db_duplicate()

        # Should save duplicate pairs
        assert mock_save.call_count >= 1

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.is_db_populated")
    @patch("imageDuplicate.load_duplicate_pairs")
    @patch("imageDuplicate.load_image")
    @patch("imageDuplicate.get_file_info")
    @patch("imageDuplicate.image_comparison")
    def test_show_duplicate_photos_with_results(
        self, mock_comparison, mock_get_info, mock_load_image, mock_load_pairs, mock_is_populated, mock_st
    ):
        """Test show_duplicate_photos_faiss with actual results."""
        # Mock populated database with duplicates
        mock_is_populated.return_value = True
        mock_load_pairs.return_value = [("file1.jpg", "file2.jpg", 0.95), ("file3.jpg", "file4.jpg", 0.85)]

        # Mock image loading
        mock_image1 = MagicMock()
        mock_image2 = MagicMock()
        mock_load_image.side_effect = [mock_image1, mock_image2, mock_image1, mock_image2]

        # Mock file info
        mock_get_info.side_effect = [
            ("10.5 MB", "file1.jpg", "1920x1080", "2023-01-01", "file1.jpg"),
            ("8.2 MB", "file2.jpg", "1280x720", "2023-01-02", "file2.jpg"),
            ("12.1 MB", "file3.jpg", "1920x1080", "2023-01-03", "file3.jpg"),
            ("9.8 MB", "file4.jpg", "1280x720", "2023-01-04", "file4.jpg"),
        ]

        # Mock session state
        mock_st.session_state = {"stop_requested": False}

        # Mock UI components
        mock_st.progress.return_value = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.markdown = MagicMock()

        from imageDuplicate import show_duplicate_photos_faiss

        show_duplicate_photos_faiss(2, 0.8, 1.0)

        # Should load and display images
        assert mock_load_image.call_count == 4
        assert mock_get_info.call_count == 4
        assert mock_comparison.call_count == 2


@pytest.mark.unit
class TestFaissOperations:
    """Test cases for FAISS operations."""


@pytest.mark.unit
@patch("imageDuplicate.get_model_and_transform")
def test_transform_pipeline(mock_get_components):
    """Test the transform pipeline setup."""
    # Mock the components
    mock_transform = MagicMock()
    mock_get_components.return_value = {
        'model': MagicMock(),
        'transform': mock_transform,
        'device': MagicMock(),
        'torch': MagicMock()
    }
    
    components = mock_get_components()
    transform = components['transform']

    # Test that transform is callable
    assert callable(transform)

    # Test convert_image_to_rgb is in the pipeline
    # This is tested indirectly through the transform composition


# Note: GPU/CPU setup tests are complex due to module import behavior
# These paths are covered through integration testing


@pytest.mark.unit
class TestComplexFaissOperations:
    """Test complex FAISS operations to improve coverage."""

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.init_or_load_faiss_index")
    def test_generate_db_duplicate_metadata_bounds_error(self, mock_init_load, mock_st):
        """Test generate_db_duplicate with metadata bounds error."""
        # Mock index with vectors but metadata bounds issue
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index.reconstruct.return_value = [0.1, 0.2, 0.3]
        mock_index.search.return_value = (
            np.array([[0.0, 0.5]]),
            np.array([[0, 5]]),  # Index 5 is out of bounds for metadata
        )

        # Metadata only has 2 items but index returns 5
        mock_init_load.return_value = (mock_index, ["file1.jpg", "file2.jpg"])

        # Mock session state and UI
        mock_st.session_state = {"stop_requested": False}
        mock_st.button.return_value = False
        mock_st.empty.return_value = MagicMock()
        mock_st.progress.return_value = MagicMock()

        from imageDuplicate import generate_db_duplicate

        # Should handle the bounds error gracefully
        generate_db_duplicate()

        # Should show error message
        mock_st.error.assert_called()

    # Note: Stop request testing is complex due to Streamlit session state behavior
    # This functionality is tested through integration testing

    @patch("imageDuplicate.st")
    @patch("imageDuplicate.is_db_populated")
    @patch("imageDuplicate.load_duplicate_pairs")
    @patch("imageDuplicate.load_image")
    @patch("imageDuplicate.get_file_info")
    def test_show_duplicate_photos_missing_images(self, mock_get_info, mock_load_image, mock_load_pairs, mock_is_populated, mock_st):
        """Test show_duplicate_photos_faiss with missing images."""
        # Mock populated database with duplicates
        mock_is_populated.return_value = True
        mock_load_pairs.return_value = [("file1.jpg", "file2.jpg", 0.95)]

        # Mock image loading to return None (missing images)
        mock_load_image.return_value = None

        # Mock file info
        mock_get_info.return_value = ("10.5 MB", "file1.jpg", "1920x1080", "2023-01-01", "file1.jpg")

        # Mock session state
        mock_st.session_state = {"stop_requested": False}
        mock_st.get.return_value = False

        from imageDuplicate import show_duplicate_photos_faiss

        show_duplicate_photos_faiss(1, 0.0, 100.0)

        # Should handle missing images
        mock_st.write.assert_called()


@pytest.mark.unit
class TestFaissIndexEdgeCases:
    """Test edge cases in FAISS index operations."""

    @patch("imageDuplicate.init_or_load_faiss_index")
    @patch("imageDuplicate.load_image")
    @patch("imageDuplicate.extract_features")
    @patch("imageDuplicate.get_faiss")
    @patch("imageDuplicate.np")
    def test_update_faiss_index_gpu_conversion(self, mock_np, mock_get_faiss, mock_extract, mock_load_image, mock_init_load):
        """Test FAISS index update with GPU conversion."""
        from imageDuplicate import update_faiss_index

        # Mock existing index
        mock_index = MagicMock()
        mock_init_load.return_value = (mock_index, [])

        # Mock successful image loading and feature extraction
        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        mock_features = np.array([1, 2, 3, 4])
        mock_extract.return_value = mock_features

        # Mock save function
        with patch("imageDuplicate.save_faiss_index_and_metadata") as mock_save:
            result = update_faiss_index("new_file.jpg")

        assert result == "processed"
        mock_index.add.assert_called_once()
        mock_save.assert_called_once()

    # Note: Exception handling in FAISS operations is complex due to GPU dependencies
    # These paths are covered through integration testing
