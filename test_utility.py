"""Unit tests for utility functions."""

from unittest.mock import MagicMock

import pytest

from utility import compare_and_color, compare_and_color_data, display_asset_column


@pytest.mark.unit
class TestCompareAndColorData:
    """Test cases for compare_and_color_data function."""

    def test_newer_date_first(self):
        """Test when first date is newer."""
        date1 = "2023-12-01T10:00:00"
        date2 = "2023-11-01T10:00:00"
        result = compare_and_color_data(date1, date2)
        assert "<span style='color: red;'>" in result
        assert date1 in result

    def test_older_date_first(self):
        """Test when first date is older."""
        date1 = "2023-11-01T10:00:00"
        date2 = "2023-12-01T10:00:00"
        result = compare_and_color_data(date1, date2)
        assert "<span style='color: green;'>" in result
        assert date1 in result

    def test_same_dates(self):
        """Test when dates are the same."""
        date1 = "2023-12-01T10:00:00"
        date2 = "2023-12-01T10:00:00"
        result = compare_and_color_data(date1, date2)
        assert "<span" not in result  # No color formatting
        assert result == date1

    def test_dates_with_z_suffix(self):
        """Test dates with Z suffix."""
        date1 = "2023-12-01T10:00:00Z"
        date2 = "2023-11-01T10:00:00Z"
        result = compare_and_color_data(date1, date2)
        assert "<span style='color: red;'>" in result
        assert date1 in result

    def test_invalid_date_format(self):
        """Test with invalid date format."""
        date1 = "invalid-date"
        date2 = "2023-12-01T10:00:00"
        result = compare_and_color_data(date1, date2)
        assert result == date1  # Should return unformatted
        assert "<span" not in result

    def test_none_date(self):
        """Test with None date."""
        date1 = None
        date2 = "2023-12-01T10:00:00"
        result = compare_and_color_data(date1, date2)
        assert result == "None"  # Should return unformatted


@pytest.mark.unit
class TestCompareAndColor:
    """Test cases for compare_and_color function."""

    def test_higher_value_first(self):
        """Test when first value is higher."""
        value1 = "10.5 MB"
        value2 = "5.2 MB"
        result = compare_and_color(value1, value2)
        assert "<span style='color: green;'>" in result
        assert value1 in result

    def test_lower_value_first(self):
        """Test when first value is lower."""
        value1 = "5.2 MB"
        value2 = "10.5 MB"
        result = compare_and_color(value1, value2)
        assert "<span style='color: red;'>" in result
        assert value1 in result

    def test_equal_values(self):
        """Test when values are equal."""
        value1 = "10.5 MB"
        value2 = "10.5 MB"
        result = compare_and_color(value1, value2)
        assert "<span" not in result  # No color formatting
        assert result == value1

    def test_resolution_comparison(self):
        """Test resolution comparison."""
        value1 = "1920 x 1080"
        value2 = "1280 x 720"
        result = compare_and_color(value1, value2)
        assert "<span style='color: green;'>" in result  # 1920 > 1280
        assert value1 in result

    def test_non_numeric_values(self):
        """Test with non-numeric values."""
        value1 = "Unknown"
        value2 = "10.5 MB"
        result = compare_and_color(value1, value2)
        assert result == value1  # Should return unformatted
        assert "<span" not in result

    def test_malformed_values(self):
        """Test with malformed values."""
        value1 = "MB"  # Missing number
        value2 = "10.5 MB"
        result = compare_and_color(value1, value2)
        assert result == value1  # Should return unformatted
        assert "<span" not in result


@pytest.mark.unit
class TestDisplayAssetColumn:
    """Test cases for display_asset_column function."""

    @pytest.fixture
    def mock_streamlit(self, mocker):
        """Mock streamlit components."""
        mock_st = mocker.patch("utility.st")
        mock_col = MagicMock()
        mock_st.columns = mock_col
        return mock_st, mock_col

    @pytest.fixture
    def sample_asset_info(self):
        """Sample asset info tuple."""
        return ("10.5 MB", "test_image.jpg", "1920 x 1080", "2023-12-01T10:00:00", "/path/to/test_image.jpg")

    @pytest.fixture
    def sample_asset_info2(self):
        """Second sample asset info tuple."""
        return ("5.2 MB", "test_image2.jpg", "1280 x 720", "2023-11-01T10:00:00", "/path/to/test_image2.jpg")

    def test_display_asset_column_basic(self, mock_streamlit, sample_asset_info, sample_asset_info2):
        """Test basic display of asset column."""
        mock_st, mock_col = mock_streamlit
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.button.return_value = False  # Don't trigger delete

        display_asset_column(mock_col, sample_asset_info, sample_asset_info2, "/path/to/file1.jpg", "/path/to/file2.jpg")

        # Should call markdown to display details
        mock_st.markdown.assert_called_once()
        # Should create a delete button
        mock_st.button.assert_called_once()

    def test_display_asset_column_delete_success(self, mock_streamlit, sample_asset_info, sample_asset_info2, mocker):
        """Test successful file deletion."""
        mock_st, mock_col = mock_streamlit
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.button.return_value = True  # Trigger delete

        # Mock the delete functions
        mock_delete_file = mocker.patch("utility.delete_file", return_value=True)
        mock_delete_pair = mocker.patch("utility.delete_duplicate_pair")
        mock_rerun = mocker.patch("utility.st.rerun")

        file_path_1 = "/path/to/file1.jpg"
        file_path_2 = "/path/to/file2.jpg"

        display_asset_column(mock_col, sample_asset_info, sample_asset_info2, file_path_1, file_path_2)

        # Should call delete functions
        mock_delete_file.assert_called_once_with(file_path_1)
        mock_delete_pair.assert_called_once_with(file_path_1, file_path_2)
        mock_st.success.assert_called_once()
        mock_rerun.assert_called_once()

    def test_display_asset_column_delete_failure(self, mock_streamlit, sample_asset_info, sample_asset_info2, mocker):
        """Test failed file deletion."""
        mock_st, mock_col = mock_streamlit
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.button.return_value = True  # Trigger delete

        # Mock the delete function to fail
        mock_delete_file = mocker.patch("utility.delete_file", return_value=False)

        file_path_1 = "/path/to/file1.jpg"
        file_path_2 = "/path/to/file2.jpg"

        display_asset_column(mock_col, sample_asset_info, sample_asset_info2, file_path_1, file_path_2)

        # Should show error message
        mock_st.error.assert_called()
        error_call_args = mock_st.error.call_args[0][0]
        assert "Failed to delete photo" in error_call_args

    def test_display_asset_column_delete_exception(self, mock_streamlit, sample_asset_info, sample_asset_info2, mocker):
        """Test exception during file deletion."""
        mock_st, mock_col = mock_streamlit
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.button.return_value = True  # Trigger delete

        # Mock the delete function to raise exception
        mock_delete_file = mocker.patch("utility.delete_file", side_effect=Exception("Test error"))

        file_path_1 = "/path/to/file1.jpg"
        file_path_2 = "/path/to/file2.jpg"

        display_asset_column(mock_col, sample_asset_info, sample_asset_info2, file_path_1, file_path_2)

        # Should show error message
        mock_st.error.assert_called()
        error_call_args = mock_st.error.call_args[0][0]
        assert "An error occurred while deleting" in error_call_args

    def test_display_asset_column_exception_handling(self, mock_streamlit, sample_asset_info, sample_asset_info2, mocker):
        """Test exception handling in display_asset_column."""
        mock_st, mock_col = mock_streamlit

        # Mock markdown to raise exception
        mock_st.markdown.side_effect = Exception("Test error")

        # Should not raise exception, should handle gracefully
        display_asset_column(mock_col, sample_asset_info, sample_asset_info2, "/path/to/file1.jpg", "/path/to/file2.jpg")

        # Should show error message
        mock_st.error.assert_called()
        error_call_args = mock_st.error.call_args[0][0]
        assert "Error displaying asset column" in error_call_args
