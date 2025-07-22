from datetime import datetime
from typing import Tuple

import streamlit as st

from db import delete_duplicate_pair
from local_media import delete_file
from logger_config import logger


def compare_and_color_data(value1: str, value2: str) -> str:
    """
    Compares two ISO date strings and colors them based on which is newer.
    
    Args:
        value1: First date string
        value2: Second date string
        
    Returns:
        HTML formatted string with color coding
    """
    try:
        date1 = datetime.fromisoformat(value1.rstrip("Z"))
        date2 = datetime.fromisoformat(value2.rstrip("Z"))
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse date strings: {value1}, {value2}. Error: {e}")
        return f"{value1}"  # Return unformatted if parsing fails

    # Compare the datetime objects
    if date1 > date2:  # value1 is newer
        return f"<span style='color: red;'>{value1}</span>"
    elif date1 < date2:  # value1 is older
        return f"<span style='color: green;'>{value1}</span>"
    else:  # They are the same
        return f"{value1}"


def compare_and_color(value1: str, value2: str) -> str:
    """
    Compares two values and colors them: green for higher, red for lower.
    
    Args:
        value1: First value to compare
        value2: Second value to compare
        
    Returns:
        HTML formatted string with color coding
    """
    try:
        # Extract numeric part for comparison (e.g., from "12.345 MB")
        num1 = float(str(value1).split()[0])
        num2 = float(str(value2).split()[0])
        if num1 > num2:
            return f"<span style='color: green;'>{value1}</span>"
        elif num1 < num2:
            return f"<span style='color: red;'>{value1}</span>"
    except (ValueError, IndexError) as e:
        logger.debug(f"Could not parse numeric values from {value1}, {value2}: {e}")
        pass  # Fallback for non-numeric or unparseable values
    return f"{value1}"


def display_asset_column(
    col: st.columns,
    asset1_info: Tuple[str, str, str, str, str],
    asset2_info: Tuple[str, str, str, str, str],
    file_path_1: str,
    file_path_2: str,
) -> None:
    """
    Displays the information for a single asset (file) in a column.
    
    Args:
        col: Streamlit column object
        asset1_info: Tuple of (file_size, file_name, resolution, creation_date, full_path)
        asset2_info: Tuple of comparison asset info
        file_path_1: Path to the first file
        file_path_2: Path to the second file
    """
    try:
        # asset_info = (formatted_file_size, file_name, resolution, creation_date, full_path)
        details = f"""
        - **File name:** {asset1_info[1]}
        - **Size:** {compare_and_color(asset1_info[0], asset2_info[0])}
        - **Resolution:** {compare_and_color(asset1_info[2], asset2_info[2])}
        - **Modified Date:** {compare_and_color_data(asset1_info[3], asset2_info[3])}
        - **Path:** `{asset1_info[4]}`
        """
        with col:
            st.markdown(details, unsafe_allow_html=True)
            delete_button_key = f"delete-{hash(file_path_1)}-paired-with-{hash(file_path_2)}"
            delete_button_label = f"Delete {asset1_info[1]}"
            if st.button(delete_button_label, key=delete_button_key):
                try:
                    logger.info(f"User requested deletion of file: {file_path_1}")
                    if delete_file(file_path_1):
                        st.success(f"Deleted photo: {file_path_1}")
                        delete_duplicate_pair(file_path_1, file_path_2)
                        logger.info(f"Successfully deleted file and database entry: {file_path_1}")
                        st.rerun()  # Rerun to refresh the view
                    else:
                        error_msg = f"Failed to delete photo: {file_path_1}"
                        st.error(error_msg)
                        logger.error(error_msg)
                except Exception as e:
                    error_msg = f"An error occurred while deleting: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error during file deletion: {e}")
    except Exception as e:
        error_msg = f"Error displaying asset column: {e}"
        logger.error(error_msg)
        st.error(error_msg)
