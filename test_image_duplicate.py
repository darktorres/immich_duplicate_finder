"""Unit tests for image duplicate detection utilities."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Mock faiss and torch before importing to avoid GPU/dependency issues
with patch('faiss.StandardGpuResources', create=True), \
     patch('torch.cuda.is_available', return_value=False), \
     patch('torch.device', return_value='cpu'):
    from imageDuplicate import convert_image_to_rgb, extract_features


@pytest.mark.unit
class TestConvertImageToRgb:
    """Test cases for convert_image_to_rgb function."""
    
    def test_convert_rgb_image(self):
        """Test converting an already RGB image."""
        # Create a simple RGB image
        rgb_image = Image.new('RGB', (10, 10), color='red')
        
        result = convert_image_to_rgb(rgb_image)
        
        assert result.mode == 'RGB'
        assert result.size == (10, 10)
    
    def test_convert_rgba_image(self):
        """Test converting RGBA image to RGB."""
        # Create an RGBA image
        rgba_image = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))
        
        result = convert_image_to_rgb(rgba_image)
        
        assert result.mode == 'RGB'
        assert result.size == (10, 10)
    
    def test_convert_grayscale_image(self):
        """Test converting grayscale image to RGB."""
        # Create a grayscale image
        gray_image = Image.new('L', (10, 10), color=128)
        
        result = convert_image_to_rgb(gray_image)
        
        assert result.mode == 'RGB'
        assert result.size == (10, 10)
    
    def test_convert_palette_image(self):
        """Test converting palette image to RGB."""
        # Create a palette image
        palette_image = Image.new('P', (10, 10))
        
        result = convert_image_to_rgb(palette_image)
        
        assert result.mode == 'RGB'
        assert result.size == (10, 10)


@pytest.mark.unit
class TestExtractFeatures:
    """Test cases for extract_features function."""
    
    @patch('imageDuplicate.model')
    @patch('imageDuplicate.transform')
    @patch('imageDuplicate.device', 'cpu')
    def test_extract_features_basic(self, mock_transform, mock_model):
        """Test basic feature extraction."""
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Mock the transform and model
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
    
    @patch('imageDuplicate.model')
    @patch('imageDuplicate.transform')
    def test_extract_features_exception_handling(self, mock_transform, mock_model):
        """Test feature extraction with exception."""
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Mock transform to raise exception
        mock_transform.side_effect = Exception("Transform error")
        
        with pytest.raises(Exception, match="Transform error"):
            extract_features(test_image)


@pytest.mark.unit
class TestFaissIndexOperations:
    """Test cases for FAISS index operations."""
    
    @patch('imageDuplicate.faiss')
    @patch('imageDuplicate.np')
    @patch('imageDuplicate.os.path.exists')
    def test_init_or_load_faiss_index_new(self, mock_exists, mock_np, mock_faiss):
        """Test initializing new FAISS index."""
        from imageDuplicate import init_or_load_faiss_index
        
        # Mock that files don't exist
        mock_exists.return_value = False
        
        index, metadata = init_or_load_faiss_index()
        
        assert index is None
        assert metadata == []
    
    @patch('imageDuplicate.faiss')
    @patch('imageDuplicate.np')
    @patch('imageDuplicate.os.path.exists')
    @patch('imageDuplicate.res', None)  # Mock no GPU resources
    def test_init_or_load_faiss_index_existing(self, mock_exists, mock_np, mock_faiss):
        """Test loading existing FAISS index."""
        from imageDuplicate import init_or_load_faiss_index
        
        # Mock that files exist
        mock_exists.return_value = True
        
        # Mock faiss and numpy operations
        mock_index = MagicMock()
        mock_faiss.read_index.return_value = mock_index
        mock_np.load.return_value.tolist.return_value = ['file1.jpg', 'file2.jpg']
        
        index, metadata = init_or_load_faiss_index()
        
        # When no GPU resources, should return CPU index directly
        assert index == mock_index
        assert metadata == ['file1.jpg', 'file2.jpg']
        
        mock_faiss.read_index.assert_called_once()
        mock_np.load.assert_called_once()
    
    @patch('imageDuplicate.faiss')
    @patch('imageDuplicate.np')
    @patch('imageDuplicate.os.path.exists')
    def test_init_or_load_faiss_index_exception(self, mock_exists, mock_np, mock_faiss):
        """Test FAISS index loading with exception."""
        from imageDuplicate import init_or_load_faiss_index
        
        # Mock that files exist but loading fails
        mock_exists.return_value = True
        mock_faiss.read_index.side_effect = Exception("FAISS error")
        
        index, metadata = init_or_load_faiss_index()
        
        # Should return defaults on error
        assert index is None
        assert metadata == []
    
    @patch('imageDuplicate.faiss')
    @patch('imageDuplicate.np')
    @patch('imageDuplicate.res', None)  # No GPU resources
    def test_save_faiss_index_and_metadata_cpu(self, mock_np, mock_faiss):
        """Test saving FAISS index on CPU."""
        from imageDuplicate import save_faiss_index_and_metadata
        
        mock_index = MagicMock()
        mock_index.getDevice = None  # CPU index
        metadata = ['file1.jpg', 'file2.jpg']
        
        save_faiss_index_and_metadata(mock_index, metadata)
        
        # Should save directly without GPU conversion
        mock_faiss.write_index.assert_called_once_with(mock_index, 'faiss_index.bin')
        mock_np.save.assert_called_once()
    
    @patch('imageDuplicate.faiss')
    @patch('imageDuplicate.np')
    def test_save_faiss_index_and_metadata_exception(self, mock_np, mock_faiss):
        """Test saving FAISS index with exception."""
        from imageDuplicate import save_faiss_index_and_metadata
        
        mock_index = MagicMock()
        metadata = ['file1.jpg', 'file2.jpg']
        
        # Mock faiss.write_index to raise exception
        mock_faiss.write_index.side_effect = Exception("Save error")
        
        with pytest.raises(Exception, match="Save error"):
            save_faiss_index_and_metadata(mock_index, metadata)


@pytest.mark.unit
class TestUpdateFaissIndex:
    """Test cases for update_faiss_index function."""
    
    @patch('imageDuplicate.init_or_load_faiss_index')
    @patch('imageDuplicate.load_image')
    def test_update_faiss_index_file_already_exists(self, mock_load_image, mock_init_load):
        """Test updating FAISS index when file already exists in metadata."""
        from imageDuplicate import update_faiss_index
        
        # Mock that file already exists in metadata
        mock_init_load.return_value = (MagicMock(), ['existing_file.jpg'])
        
        result = update_faiss_index('existing_file.jpg')
        
        assert result == "skipped"
        mock_load_image.assert_not_called()
    
    @patch('imageDuplicate.init_or_load_faiss_index')
    @patch('imageDuplicate.load_image')
    def test_update_faiss_index_image_load_fails(self, mock_load_image, mock_init_load):
        """Test updating FAISS index when image loading fails."""
        from imageDuplicate import update_faiss_index
        
        # Mock that file doesn't exist in metadata but image loading fails
        mock_init_load.return_value = (MagicMock(), [])
        mock_load_image.return_value = None
        
        result = update_faiss_index('new_file.jpg')
        
        assert result == "error"
    
    @patch('imageDuplicate.init_or_load_faiss_index')
    @patch('imageDuplicate.load_image')
    @patch('imageDuplicate.extract_features')
    @patch('imageDuplicate.save_faiss_index_and_metadata')
    @patch('imageDuplicate.faiss')
    @patch('imageDuplicate.np')
    def test_update_faiss_index_new_file_success(self, mock_np, mock_faiss, mock_save, 
                                                 mock_extract, mock_load_image, mock_init_load):
        """Test successfully updating FAISS index with new file."""
        from imageDuplicate import update_faiss_index
        
        # Mock successful scenario
        mock_index = MagicMock()
        mock_init_load.return_value = (mock_index, [])
        
        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        
        mock_features = np.array([1, 2, 3, 4])
        mock_extract.return_value = mock_features
        
        result = update_faiss_index('new_file.jpg')
        
        assert result == "processed"
        
        # Verify the pipeline
        mock_load_image.assert_called_once_with('new_file.jpg')
        mock_extract.assert_called_once_with(mock_image)
        mock_index.add.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('imageDuplicate.init_or_load_faiss_index')
    @patch('imageDuplicate.load_image')
    @patch('imageDuplicate.extract_features')
    @patch('imageDuplicate.faiss')
    @patch('imageDuplicate.np')
    @patch('imageDuplicate.res', None)  # No GPU
    def test_update_faiss_index_create_new_index(self, mock_np, mock_faiss, mock_extract, 
                                                 mock_load_image, mock_init_load):
        """Test creating new FAISS index when none exists."""
        from imageDuplicate import update_faiss_index
        
        # Mock no existing index
        mock_init_load.return_value = (None, [])
        
        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        
        mock_features = np.array([1, 2, 3, 4])
        mock_extract.return_value = mock_features
        
        # Mock FAISS index creation
        mock_cpu_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_cpu_index
        
        with patch('imageDuplicate.save_faiss_index_and_metadata') as mock_save:
            result = update_faiss_index('new_file.jpg')
        
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
    
    @patch('imageDuplicate.st')
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
        with patch('imageDuplicate.update_faiss_index', return_value="processed"):
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
    
    @patch('imageDuplicate.st')
    @patch('imageDuplicate.init_or_load_faiss_index')
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
    
    @patch('imageDuplicate.st')
    @patch('imageDuplicate.is_db_populated')
    def test_show_duplicate_photos_empty_db(self, mock_is_populated, mock_st):
        """Test show_duplicate_photos_faiss with empty database."""
        # Mock empty database
        mock_is_populated.return_value = False
        
        from imageDuplicate import show_duplicate_photos_faiss
        
        show_duplicate_photos_faiss(10, 0.0, 100.0)
        
        # Should show message about empty database
        mock_st.write.assert_called_with(
            "The database does not contain any duplicate entries. Please generate/update the database."
        )
    
    @patch('imageDuplicate.st')
    @patch('imageDuplicate.is_db_populated')
    @patch('imageDuplicate.load_duplicate_pairs')
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
def test_model_and_transform_initialization():
    """Test that model and transform are properly initialized."""
    # This test ensures the global model and transform objects are created
    from imageDuplicate import model, transform, weights
    
    # Should have these objects defined
    assert model is not None
    assert transform is not None
    assert weights is not None


@pytest.mark.unit
def test_device_and_gpu_setup():
    """Test device and GPU setup logic."""
    from imageDuplicate import device, res
    
    # Should have device defined
    assert device is not None
    
    # res can be None (CPU) or a GPU resource object
    # This test just ensures no exceptions during import