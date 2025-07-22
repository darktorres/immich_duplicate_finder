"""Unit tests for main application module."""

import pytest
from unittest.mock import MagicMock, patch

# Mock faiss before importing app to avoid GPU issues in tests
with patch('faiss.StandardGpuResources', create=True):
    from app import setup_session_state, configure_sidebar, main


@pytest.mark.unit
class TestSetupSessionState:
    """Test cases for setup_session_state function."""
    
    def test_setup_session_state_new_session(self, mocker):
        """Test session state setup with new session."""
        mock_session_state = {}
        mocker.patch('app.st.session_state', mock_session_state)
        
        setup_session_state()
        
        # Check that all expected keys are set with default values
        expected_keys = [
            "calculate_faiss", "generate_db_duplicate", "show_faiss_duplicate",
            "stop_index", "stop_requested", "message", "progress",
            "faiss_min_threshold", "faiss_max_threshold", "limit"
        ]
        
        for key in expected_keys:
            assert key in mock_session_state
        
        # Check specific default values
        assert mock_session_state["calculate_faiss"] is False
        assert mock_session_state["faiss_min_threshold"] == 0.0
        assert mock_session_state["faiss_max_threshold"] == 100.0
        assert mock_session_state["limit"] == 10
    
    def test_setup_session_state_existing_session(self, mocker):
        """Test session state setup with existing values."""
        mock_session_state = {
            "calculate_faiss": True,
            "faiss_min_threshold": 5.0,
            "limit": 20
        }
        mocker.patch('app.st.session_state', mock_session_state)
        
        setup_session_state()
        
        # Existing values should not be overwritten
        assert mock_session_state["calculate_faiss"] is True
        assert mock_session_state["faiss_min_threshold"] == 5.0
        assert mock_session_state["limit"] == 20
        
        # Missing values should be added with defaults
        assert mock_session_state["generate_db_duplicate"] is False
        assert mock_session_state["faiss_max_threshold"] == 100.0


@pytest.mark.unit
class TestConfigureSidebar:
    """Test cases for configure_sidebar function - simplified tests."""
    
    @patch('app.st')
    @patch('app.validate_threshold_range')
    @patch('app.validate_limit')
    def test_configure_sidebar_validation_calls(self, mock_validate_limit, mock_validate_threshold, mock_st):
        """Test that validation functions are called."""
        # Mock streamlit components
        mock_st.sidebar.expander.return_value.__enter__ = MagicMock()
        mock_st.sidebar.expander.return_value.__exit__ = MagicMock()
        mock_st.number_input.side_effect = [0.0, 100.0, 10]
        mock_st.button.return_value = False
        mock_st.session_state = {"faiss_min_threshold": 0.0, "faiss_max_threshold": 100.0, "limit": 10}
        
        # Mock validation returns
        mock_validate_threshold.return_value = (True, None)
        mock_validate_limit.return_value = (True, None)
        
        configure_sidebar()
        
        # Should call validation functions
        mock_validate_threshold.assert_called_once_with(0.0, 100.0)
        mock_validate_limit.assert_called_once_with(10)
    
    @patch('app.st')
    @patch('app.validate_threshold_range')
    @patch('app.validate_limit')
    def test_configure_sidebar_invalid_inputs(self, mock_validate_limit, mock_validate_threshold, mock_st):
        """Test sidebar behavior with invalid inputs."""
        # Mock streamlit components
        mock_st.sidebar.expander.return_value.__enter__ = MagicMock()
        mock_st.sidebar.expander.return_value.__exit__ = MagicMock()
        mock_st.number_input.side_effect = [50.0, 25.0, 0]  # Invalid values
        mock_st.button.return_value = False
        mock_st.session_state = {"faiss_min_threshold": 0.0, "faiss_max_threshold": 100.0, "limit": 10}
        
        # Mock validation returns - both invalid
        mock_validate_threshold.return_value = (False, "Invalid threshold")
        mock_validate_limit.return_value = (False, "Invalid limit")
        
        configure_sidebar()
        
        # Should show error messages
        assert mock_st.error.call_count == 2
        
        # Button should be disabled
        mock_st.button.assert_called_with("Find duplicate photos", disabled=True)


@pytest.mark.unit
class TestMain:
    """Test cases for main function."""
    
    @pytest.fixture
    def mock_dependencies(self, mocker):
        """Mock all dependencies for main function."""
        mocks = {
            'setup_session_state': mocker.patch('app.setup_session_state'),
            'configure_sidebar': mocker.patch('app.configure_sidebar'),
            'startup_sidebar': mocker.patch('app.startup_sidebar', return_value="/valid/path"),
            'validate_folder_path': mocker.patch('app.validate_folder_path', return_value=(True, None)),
            'get_media_files': mocker.patch('app.get_media_files', return_value=["file1.jpg", "file2.jpg"]),
            'calculateFaissIndex': mocker.patch('app.calculateFaissIndex'),
            'generate_db_duplicate': mocker.patch('app.generate_db_duplicate'),
            'show_duplicate_photos_faiss': mocker.patch('app.show_duplicate_photos_faiss'),
            'st': mocker.patch('app.st')
        }
        
        # Mock session state
        mocks['st'].session_state = {
            "calculate_faiss": False,
            "generate_db_duplicate": False,
            "show_faiss_duplicate": False,
            "limit": 10,
            "faiss_min_threshold": 0.0,
            "faiss_max_threshold": 100.0
        }
        
        return mocks
    
    def test_main_no_operations(self, mock_dependencies):
        """Test main function with no operations triggered."""
        mocks = mock_dependencies
        
        main()
        
        # Should call setup functions
        mocks['setup_session_state'].assert_called_once()
        mocks['configure_sidebar'].assert_called_once()
        mocks['startup_sidebar'].assert_called_once()
        
        # Should not call processing functions
        mocks['calculateFaissIndex'].assert_not_called()
        mocks['generate_db_duplicate'].assert_not_called()
        mocks['show_duplicate_photos_faiss'].assert_not_called()
    
    def test_main_calculate_faiss(self, mock_dependencies):
        """Test main function with FAISS calculation triggered."""
        mocks = mock_dependencies
        mocks['st'].session_state["calculate_faiss"] = True
        
        main()
        
        # Should call FAISS calculation
        mocks['get_media_files'].assert_called_once_with("/valid/path")
        mocks['calculateFaissIndex'].assert_called_once()
        mocks['st'].write.assert_called_with("Found 2 image files to process.")
        
        # Should reset flag
        assert mocks['st'].session_state["calculate_faiss"] is False
    
    def test_main_calculate_faiss_no_files(self, mock_dependencies):
        """Test main function with FAISS calculation but no files found."""
        mocks = mock_dependencies
        mocks['st'].session_state["calculate_faiss"] = True
        mocks['get_media_files'].return_value = []
        
        main()
        
        # Should show warning
        mocks['st'].warning.assert_called_with("No image files found in the specified folder.")
        mocks['calculateFaissIndex'].assert_not_called()
    
    def test_main_generate_db_duplicate(self, mock_dependencies):
        """Test main function with duplicate DB generation triggered."""
        mocks = mock_dependencies
        mocks['st'].session_state["generate_db_duplicate"] = True
        
        main()
        
        # Should call DB generation
        mocks['generate_db_duplicate'].assert_called_once()
        
        # Should reset flag
        assert mocks['st'].session_state["generate_db_duplicate"] is False
    
    def test_main_show_duplicates(self, mock_dependencies):
        """Test main function with show duplicates triggered."""
        mocks = mock_dependencies
        mocks['st'].session_state["show_faiss_duplicate"] = True
        
        main()
        
        # Should call show duplicates
        mocks['show_duplicate_photos_faiss'].assert_called_once_with(10, 0.0, 100.0)
        
        # Should reset flag
        assert mocks['st'].session_state["show_faiss_duplicate"] is False
    
    def test_main_invalid_folder_path(self, mock_dependencies):
        """Test main function with invalid folder path."""
        mocks = mock_dependencies
        mocks['st'].session_state["calculate_faiss"] = True
        mocks['validate_folder_path'].return_value = (False, "Path does not exist")
        
        main()
        
        # Should show error and reset flags
        mocks['st'].error.assert_called_with("Folder Path Error: Path does not exist")
        assert mocks['st'].session_state["calculate_faiss"] is False
        assert mocks['st'].session_state["generate_db_duplicate"] is False
        assert mocks['st'].session_state["show_faiss_duplicate"] is False
        
        # Should not call processing functions
        mocks['calculateFaissIndex'].assert_not_called()
    
    def test_main_exception_handling(self, mock_dependencies):
        """Test main function exception handling."""
        mocks = mock_dependencies
        mocks['setup_session_state'].side_effect = Exception("Test error")
        
        main()
        
        # Should show error message
        mocks['st'].error.assert_called()
        error_calls = [call[0][0] for call in mocks['st'].error.call_args_list]
        assert any("Unexpected error in main application" in call for call in error_calls)


@pytest.mark.unit
def test_configure_sidebar_button_clicks():
    """Test button click handlers in configure_sidebar."""
    with patch('app.st') as mock_st, \
         patch('app.validate_threshold_range', return_value=(True, None)), \
         patch('app.validate_limit', return_value=(True, None)):
        
        # Mock session state
        mock_st.session_state = {
            "faiss_min_threshold": 0.0, 
            "faiss_max_threshold": 100.0, 
            "limit": 10
        }
        
        # Mock sidebar components
        mock_st.sidebar.expander.return_value.__enter__ = MagicMock()
        mock_st.sidebar.expander.return_value.__exit__ = MagicMock()
        mock_st.number_input.side_effect = [0.0, 100.0, 10]
        
        # Test different button clicks
        button_calls = []
        def mock_button(text, **kwargs):
            button_calls.append((text, kwargs))
            if "Create/Update FAISS index" in text:
                return True
            return False
        
        mock_st.button = mock_button
        
        configure_sidebar()
        
        # Should have set the FAISS flag
        assert mock_st.session_state.get("calculate_faiss") is True


@pytest.mark.unit  
def test_configure_sidebar_duplicate_db_button():
    """Test duplicate DB button click."""
    with patch('app.st') as mock_st, \
         patch('app.validate_threshold_range', return_value=(True, None)), \
         patch('app.validate_limit', return_value=(True, None)):
        
        # Mock session state
        mock_st.session_state = {
            "faiss_min_threshold": 0.0, 
            "faiss_max_threshold": 100.0, 
            "limit": 10
        }
        
        # Mock sidebar components
        mock_st.sidebar.expander.return_value.__enter__ = MagicMock()
        mock_st.sidebar.expander.return_value.__exit__ = MagicMock()
        mock_st.number_input.side_effect = [0.0, 100.0, 10]
        
        # Mock button to return True for duplicate DB button
        def mock_button(text, **kwargs):
            if "Create/Update duplicate DB" in text:
                return True
            return False
        
        mock_st.button = mock_button
        
        configure_sidebar()
        
        # Should have set the duplicate DB flag
        assert mock_st.session_state.get("generate_db_duplicate") is True


@pytest.mark.unit
def test_configure_sidebar_find_photos_button():
    """Test find duplicate photos button click."""
    with patch('app.st') as mock_st, \
         patch('app.validate_threshold_range', return_value=(True, None)), \
         patch('app.validate_limit', return_value=(True, None)):
        
        # Mock session state
        mock_st.session_state = {
            "faiss_min_threshold": 0.0, 
            "faiss_max_threshold": 100.0, 
            "limit": 10
        }
        
        # Mock sidebar components
        mock_st.sidebar.expander.return_value.__enter__ = MagicMock()
        mock_st.sidebar.expander.return_value.__exit__ = MagicMock()
        mock_st.number_input.side_effect = [0.0, 100.0, 10]
        
        # Mock button to return True for find photos button
        def mock_button(text, **kwargs):
            if "Find duplicate photos" in text:
                return True
            return False
        
        mock_st.button = mock_button
        
        configure_sidebar()
        
        # Should have set the show duplicate flag
        assert mock_st.session_state.get("show_faiss_duplicate") is True