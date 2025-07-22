import os

import streamlit as st

from db import startup_db_configurations, startup_processed_duplicate_faiss_db
from imageDuplicate import calculateFaissIndex, generate_db_duplicate, show_duplicate_photos_faiss
from local_media import get_media_files, setup_local_media
from startup import startup_sidebar

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
###############STARTUP#####################

# Set page title and favicon
st.set_page_config(page_title="Local duplicator finder ", page_icon="🖼️")

startup_db_configurations()
startup_processed_duplicate_faiss_db()
setup_local_media()


def setup_session_state():
    """Initialize session state with default values."""
    session_defaults = {
        "calculate_faiss": False,
        "generate_db_duplicate": False,
        "show_faiss_duplicate": False,
        "stop_index": False,
        "stop_requested": False,
        "message": "",
        "progress": 0,
    }
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def configure_sidebar():
    """Configure the sidebar for user inputs."""
    with st.sidebar:
        with st.expander("Image Duplicate Finder", expanded=True):
            # Button to generate/update the FAISS index
            if st.button("Create/Update FAISS index"):
                st.session_state["calculate_faiss"] = True

            # Button to trigger the generation of the duplicates database
            if st.button("Create/Update duplicate DB"):
                st.session_state["generate_db_duplicate"] = True

            st.markdown("---")
            # Input for setting the minimum FAISS threshold
            st.session_state["faiss_min_threshold"] = st.number_input(
                "Minimum Faiss threshold",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get("faiss_min_threshold", 0.0),
                step=0.01,
                help="Set the lower limit of the FAISS similarity threshold for considering duplicates.",
            )

            # Input for setting the maximum FAISS threshold
            st.session_state["faiss_max_threshold"] = st.number_input(
                "Maximum Faiss threshold",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get("faiss_max_threshold", 100.0),
                step=0.01,
                help="Set the upper limit of the FAISS similarity threshold for considering duplicates.",
            )

            # Input for setting the maximum FAISS threshold
            st.session_state["limit"] = st.number_input(
                "Number of Pairs to Display",
                value=st.session_state.get("limit", 10),
                step=1,
                help="Set the number of pairs to display for the comparison",
            )

            if st.button("Find duplicate photos"):
                st.session_state["show_faiss_duplicate"] = True

        st.markdown("---")
        # Display program version and additional data
        program_version = "v0.2.0-local"
        st.markdown(f"**Version:** {program_version}")


def main():
    setup_session_state()
    configure_sidebar()
    folder_path = startup_sidebar()

    # Check for folder path validity if an action is triggered
    if st.session_state["calculate_faiss"] or st.session_state["generate_db_duplicate"] or st.session_state["show_faiss_duplicate"]:
        if not folder_path or not os.path.isdir(folder_path):
            st.error("Please configure a valid media folder path in the sidebar settings.")
            # Reset all flags to prevent stuck state
            st.session_state["calculate_faiss"] = False
            st.session_state["generate_db_duplicate"] = False
            st.session_state["show_faiss_duplicate"] = False
            return  # Stop further execution since there are no assets to process

    # Calculate the FAISS index if the corresponding flag is set
    if st.session_state["calculate_faiss"]:
        media_files = get_media_files(folder_path)
        if media_files:
            st.write(f"Found {len(media_files)} image files to process.")
            calculateFaissIndex(media_files)
        else:
            st.warning("No image files found in the specified folder.")
        # Reset the flag after processing
        st.session_state["calculate_faiss"] = False

    # Generate duplicate database if the corresponding flag is set
    if st.session_state["generate_db_duplicate"]:
        generate_db_duplicate()
        # Reset the flag after processing
        st.session_state["generate_db_duplicate"] = False

    # Show FAISS duplicate photos if the corresponding flag is set
    if st.session_state["show_faiss_duplicate"]:
        show_duplicate_photos_faiss(
            st.session_state["limit"],
            st.session_state["faiss_min_threshold"],
            st.session_state["faiss_max_threshold"],
        )
        # Reset the flag after processing
        st.session_state["show_faiss_duplicate"] = False


if __name__ == "__main__":
    main()
