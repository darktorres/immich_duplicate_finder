from datetime import datetime

import streamlit as st

from db import delete_duplicate_pair
from local_media import delete_file


def compare_and_color_data(value1, value2):
    """Compares two ISO date strings and colors them based on which is newer."""
    try:
        date1 = datetime.fromisoformat(value1.rstrip("Z"))
        date2 = datetime.fromisoformat(value2.rstrip("Z"))
    except (ValueError, AttributeError):
        return f"{value1}"  # Return unformatted if parsing fails

    # Compare the datetime objects
    if date1 > date2:  # value1 is newer
        return f"<span style='color: red;'>{value1}</span>"
    elif date1 < date2:  # value1 is older
        return f"<span style='color: green;'>{value1}</span>"
    else:  # They are the same
        return f"{value1}"


def compare_and_color(value1, value2):
    """Compares two values and colors them: green for higher, red for lower."""
    try:
        # Extract numeric part for comparison (e.g., from "12.345 MB")
        num1 = float(str(value1).split()[0])
        num2 = float(str(value2).split()[0])
        if num1 > num2:
            return f"<span style='color: green;'>{value1}</span>"
        elif num1 < num2:
            return f"<span style='color: red;'>{value1}</span>"
    except (ValueError, IndexError):
        pass  # Fallback for non-numeric or unparseable values
    return f"{value1}"


def display_asset_column(col, asset1_info, asset2_info, file_path_1, file_path_2):
    """Displays the information for a single asset (file) in a column."""
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
        delete_button_key = f"delete-{file_path_1}"
        delete_button_label = f"Delete {asset1_info[1]}"
        if st.button(delete_button_label, key=delete_button_key):
            try:
                if delete_file(file_path_1):
                    st.success(f"Deleted photo: {file_path_1}")
                    delete_duplicate_pair(file_path_1, file_path_2)
                    st.rerun()  # Rerun to refresh the view
                else:
                    st.error(f"Failed to delete photo: {file_path_1}")
            except Exception as e:
                st.error(f"An error occurred while deleting: {str(e)}")
