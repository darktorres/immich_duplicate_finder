import os

import streamlit as st

# Apply memory configuration before any heavy imports
from memory_config import MEMORY_CONFIG
MEMORY_CONFIG.apply_environment_settings()

from db import startup_db_configurations, startup_processed_duplicate_faiss_db
from local_media import setup_local_media
from logger_config import logger
from startup import startup_sidebar
from validation import validate_folder_path, validate_limit, validate_threshold_range

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
###############STARTUP#####################

# Set page title and favicon
st.set_page_config(page_title="Local duplicator finder ", page_icon="🖼️")

startup_db_configurations()
startup_processed_duplicate_faiss_db()
setup_local_media()


def setup_session_state() -> None:
    """Initialize session state with default values."""
    session_defaults = {
        "calculate_faiss": False,
        "generate_db_duplicate": False,
        "show_faiss_duplicate": False,
        "stop_index": False,
        "stop_requested": False,
        "message": "",
        "progress": 0,
        "faiss_min_threshold": 0.0,
        "faiss_max_threshold": 100.0,
        "limit": 10,
    }
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def configure_sidebar() -> None:
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
            min_threshold = st.number_input(
                "Minimum Faiss threshold",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get("faiss_min_threshold", 0.0),
                step=0.01,
                help="Set the lower limit of the FAISS similarity threshold for considering duplicates.",
            )

            # Input for setting the maximum FAISS threshold
            max_threshold = st.number_input(
                "Maximum Faiss threshold",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.get("faiss_max_threshold", 100.0),
                step=0.01,
                help="Set the upper limit of the FAISS similarity threshold for considering duplicates.",
            )

            # Input for setting the number of pairs to display
            limit = st.number_input(
                "Number of Pairs to Display",
                min_value=1,
                max_value=1000,
                value=st.session_state.get("limit", 10),
                step=1,
                help="Set the number of pairs to display for the comparison",
            )

            # Validate inputs
            threshold_valid, threshold_error = validate_threshold_range(min_threshold, max_threshold)
            limit_valid, limit_error = validate_limit(limit)

            if not threshold_valid:
                st.error(f"Threshold Error: {threshold_error}")
            if not limit_valid:
                st.error(f"Limit Error: {limit_error}")

            # Update session state only if valid
            if threshold_valid:
                st.session_state["faiss_min_threshold"] = min_threshold
                st.session_state["faiss_max_threshold"] = max_threshold
            if limit_valid:
                st.session_state["limit"] = limit

            # Only enable button if all inputs are valid
            button_disabled = not (threshold_valid and limit_valid)
            if st.button("Find duplicate photos", disabled=button_disabled):
                st.session_state["show_faiss_duplicate"] = True

        st.markdown("---")
        # Display program version and additional data
        program_version = "v0.3.0-enhanced-optimized"
        st.markdown(f"**Version:** {program_version}")

        # Show current log level
        current_log_level = logger.level
        log_level_name = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}.get(current_log_level, "UNKNOWN")
        st.markdown(f"**Log Level:** {log_level_name}")
        
        # Show memory optimization status
        st.markdown(f"**Memory Config:** Batch size {MEMORY_CONFIG.batch_size}")
        st.markdown(f"**Processing:** {'CPU Only' if MEMORY_CONFIG.use_cpu_only else 'GPU Enabled'}")


def main() -> None:
    """Main application function."""
    try:
        setup_session_state()
        configure_sidebar()
        folder_path = startup_sidebar()

        # Check for folder path validity if an action is triggered
        if st.session_state["calculate_faiss"] or st.session_state["generate_db_duplicate"] or st.session_state["show_faiss_duplicate"]:
            folder_valid, folder_error = validate_folder_path(folder_path)
            if not folder_valid:
                st.error(f"Folder Path Error: {folder_error}")
                logger.warning(f"Invalid folder path provided: {folder_path}")
                # Reset all flags to prevent stuck state
                st.session_state["calculate_faiss"] = False
                st.session_state["generate_db_duplicate"] = False
                st.session_state["show_faiss_duplicate"] = False
                return  # Stop further execution since there are no assets to process

        # Calculate the FAISS index if the corresponding flag is set
        if st.session_state["calculate_faiss"]:
            logger.info("Starting memory-optimized FAISS index calculation")
            # Lazy import heavy modules only when needed
            from imageDuplicate import calculateFaissIndex
            from local_media import get_media_files
            
            media_files = get_media_files(folder_path)
            if media_files:
                st.write(f"Found {len(media_files)} image files to process with memory optimization.")
                st.info(f"🚀 Processing in batches of {MEMORY_CONFIG.batch_size} for memory efficiency")
                calculateFaissIndex(media_files)
            else:
                st.warning("No image files found in the specified folder.")
                logger.warning(f"No media files found in folder: {folder_path}")
            st.session_state["calculate_faiss"] = False

        # Generate duplicate database if the corresponding flag is set
        if st.session_state["generate_db_duplicate"]:
            logger.info("Starting memory-optimized duplicate database generation")
            # Lazy import heavy modules only when needed
            from imageDuplicate import generate_db_duplicate
            st.info("💾 Using batch processing for memory efficiency")
            generate_db_duplicate()
            st.session_state["generate_db_duplicate"] = False

        # Show FAISS duplicate photos if the corresponding flag is set
        if st.session_state["show_faiss_duplicate"]:
            logger.info("Starting memory-optimized duplicate photo display")
            # Lazy import heavy modules only when needed
            from imageDuplicate import show_duplicate_photos_faiss
            show_duplicate_photos_faiss(
                st.session_state["limit"],
                st.session_state["faiss_min_threshold"],
                st.session_state["faiss_max_threshold"],
            )
            st.session_state["show_faiss_duplicate"] = False

    except Exception as e:
        error_msg = f"Unexpected error in main application: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        st.error("Please check the logs for more details.")


if __name__ == "__main__":
    main()
