import streamlit as st

from db import load_settings_from_db, save_settings_to_db
from logger_config import logger
from validation import validate_folder_path


def startup_sidebar() -> str:
    """
    Configures the Streamlit sidebar for settings and returns the media folder path.
    
    Returns:
        The configured media folder path
    """
    folder_path = load_settings_from_db()

    with st.sidebar.expander("Settings", expanded=True):
        new_folder_path = st.text_input("Media Folder Path", folder_path)
        
        # Validate the folder path in real-time
        if new_folder_path and new_folder_path != folder_path:
            is_valid, error_msg = validate_folder_path(new_folder_path)
            if not is_valid:
                st.error(f"Invalid path: {error_msg}")
            else:
                st.success("✓ Valid folder path")

        if st.button("Save Settings"):
            is_valid, error_msg = validate_folder_path(new_folder_path)
            if is_valid:
                save_settings_to_db(new_folder_path)
                st.sidebar.success("Settings saved!")
                logger.info(f"Settings saved with folder path: {new_folder_path}")
                # Rerun to apply the new path immediately
                st.rerun()
            else:
                st.sidebar.error(f"Cannot save invalid path: {error_msg}")
                logger.warning(f"Attempted to save invalid folder path: {new_folder_path}")

    return new_folder_path if new_folder_path else folder_path
