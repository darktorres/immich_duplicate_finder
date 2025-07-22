import streamlit as st

from db import load_settings_from_db, save_settings_to_db


def startup_sidebar():
    """Configures the Streamlit sidebar for settings and returns the media folder path."""
    folder_path = load_settings_from_db()

    with st.sidebar.expander("Settings", expanded=True):
        folder_path = st.text_input("Media Folder Path", folder_path)

        if st.button("Save Settings"):
            save_settings_to_db(folder_path)
            st.sidebar.success("Settings saved!")
            # Rerun to apply the new path immediately
            st.rerun()

    return folder_path
