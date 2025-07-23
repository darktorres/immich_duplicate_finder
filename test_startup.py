"""Unit tests for startup utilities."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from startup import startup_sidebar


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    yield temp_dir

    # Cleanup
    os.chdir(original_cwd)
    for db_file in ["settings.db"]:
        db_path = os.path.join(temp_dir, db_file)
        if os.path.exists(db_path):
            os.unlink(db_path)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.mark.unit
def test_startup_sidebar_with_existing_settings(temp_db_dir, mocker):
    """Test startup sidebar with existing settings."""
    # Mock streamlit components
    mock_sidebar = mocker.patch("startup.st.sidebar")
    mock_expander = MagicMock()
    mock_sidebar.expander.return_value.__enter__.return_value = mock_expander

    mock_text_input = mocker.patch("startup.st.text_input")
    mock_button = mocker.patch("startup.st.button")
    mock_success = mocker.patch("startup.st.sidebar.success")
    mock_error = mocker.patch("startup.st.sidebar.error")
    mock_rerun = mocker.patch("startup.st.rerun")

    # Setup test data
    test_path = "/test/path"
    mock_text_input.return_value = test_path
    mock_button.return_value = False  # Don't trigger save

    # Mock database functions
    mocker.patch("startup.load_settings_from_db", return_value=test_path)
    mocker.patch("startup.validate_folder_path", return_value=(True, None))

    result = startup_sidebar()

    assert result == test_path
    mock_text_input.assert_called_once_with("Media Folder Path", test_path)


@pytest.mark.unit
def test_startup_sidebar_save_valid_path(temp_db_dir, mocker):
    """Test saving valid path in startup sidebar."""
    # Mock streamlit components
    mock_sidebar = mocker.patch("startup.st.sidebar")
    mock_expander = MagicMock()
    mock_sidebar.expander.return_value.__enter__.return_value = mock_expander

    mock_text_input = mocker.patch("startup.st.text_input")
    mock_button = mocker.patch("startup.st.button")
    mock_success = mocker.patch("startup.st.sidebar.success")
    mock_rerun = mocker.patch("startup.st.rerun")

    # Setup test data
    test_path = "/valid/path"
    mock_text_input.return_value = test_path
    mock_button.return_value = True  # Trigger save

    # Mock database and validation functions
    mock_load = mocker.patch("startup.load_settings_from_db", return_value="")
    mock_save = mocker.patch("startup.save_settings_to_db")
    mocker.patch("startup.validate_folder_path", return_value=(True, None))

    result = startup_sidebar()

    mock_save.assert_called_once_with(test_path)
    mock_success.assert_called_once_with("Settings saved!")
    mock_rerun.assert_called_once()


@pytest.mark.unit
def test_startup_sidebar_save_invalid_path(temp_db_dir, mocker):
    """Test saving invalid path in startup sidebar."""
    # Mock streamlit components
    mock_sidebar = mocker.patch("startup.st.sidebar")
    mock_expander = MagicMock()
    mock_sidebar.expander.return_value.__enter__.return_value = mock_expander

    mock_text_input = mocker.patch("startup.st.text_input")
    mock_button = mocker.patch("startup.st.button")
    mock_error = mocker.patch("startup.st.sidebar.error")

    # Setup test data
    test_path = "/invalid/path"
    mock_text_input.return_value = test_path
    mock_button.return_value = True  # Trigger save

    # Mock database and validation functions
    mock_load = mocker.patch("startup.load_settings_from_db", return_value="")
    mock_save = mocker.patch("startup.save_settings_to_db")
    mocker.patch("startup.validate_folder_path", return_value=(False, "Path does not exist"))

    result = startup_sidebar()

    mock_save.assert_not_called()
    mock_error.assert_called_once_with("Cannot save invalid path: Path does not exist")


@pytest.mark.unit
def test_startup_sidebar_real_time_validation(temp_db_dir, mocker):
    """Test real-time validation in startup sidebar."""
    # Mock streamlit components
    mock_sidebar = mocker.patch("startup.st.sidebar")
    mock_expander = MagicMock()
    mock_sidebar.expander.return_value.__enter__.return_value = mock_expander

    mock_text_input = mocker.patch("startup.st.text_input")
    mock_button = mocker.patch("startup.st.button")
    mock_success = mocker.patch("startup.st.success")
    mock_error = mocker.patch("startup.st.error")

    # Setup test data - new path different from existing
    existing_path = "/old/path"
    new_path = "/new/path"
    mock_text_input.return_value = new_path
    mock_button.return_value = False  # Don't trigger save

    # Mock database and validation functions
    mock_load = mocker.patch("startup.load_settings_from_db", return_value=existing_path)
    mocker.patch("startup.validate_folder_path", return_value=(True, None))

    result = startup_sidebar()

    # Should show success message for valid path
    mock_success.assert_called_once_with("✓ Valid folder path")
    assert result == new_path


@pytest.mark.unit
def test_startup_sidebar_real_time_validation_invalid(temp_db_dir, mocker):
    """Test real-time validation with invalid path."""
    # Mock streamlit components
    mock_sidebar = mocker.patch("startup.st.sidebar")
    mock_expander = MagicMock()
    mock_sidebar.expander.return_value.__enter__.return_value = mock_expander

    mock_text_input = mocker.patch("startup.st.text_input")
    mock_button = mocker.patch("startup.st.button")
    mock_error = mocker.patch("startup.st.error")

    # Setup test data
    existing_path = "/old/path"
    new_path = "/invalid/path"
    mock_text_input.return_value = new_path
    mock_button.return_value = False

    # Mock database and validation functions
    mock_load = mocker.patch("startup.load_settings_from_db", return_value=existing_path)
    mocker.patch("startup.validate_folder_path", return_value=(False, "Path does not exist"))

    result = startup_sidebar()

    # Should show error message for invalid path
    mock_error.assert_called_once_with("Invalid path: Path does not exist")
    assert result == new_path
