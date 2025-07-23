"""
Worker threads for background operations.
"""

from PySide6.QtCore import QObject, Signal

from gui.image_processing import calculate_faiss_index_gui, generate_duplicate_db_gui
from local_media import get_media_files
from logger_config import logger


class FaissIndexWorker(QObject):
    """Worker for creating FAISS index."""

    progress = Signal(int, str)
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        """Run the FAISS index creation."""
        try:
            self.progress.emit(10, "Getting media files...")
            media_files = get_media_files(self.folder_path)

            if not media_files:
                self.finished.emit(False, "No media files found")
                return

            self.progress.emit(30, f"Processing {len(media_files)} files...")

            def progress_callback(progress, message):
                self.progress.emit(progress, message)

            success = calculate_faiss_index_gui(media_files, progress_callback)

            if success:
                self.progress.emit(100, "FAISS index created successfully")
                self.finished.emit(True, "Success")
            else:
                self.finished.emit(False, "Failed to create FAISS index")

        except Exception as e:
            error_msg = f"Failed to create FAISS index: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            self.finished.emit(False, error_msg)


class DuplicateDbWorker(QObject):
    """Worker for creating duplicate database."""

    progress = Signal(int, str)
    finished = Signal(bool, str)
    error = Signal(str)

    def run(self):
        """Run the duplicate database creation."""
        try:

            def progress_callback(progress, message):
                self.progress.emit(progress, message)

            success = generate_duplicate_db_gui(progress_callback)

            if success:
                self.progress.emit(100, "Duplicate database created successfully")
                self.finished.emit(True, "Success")
            else:
                self.finished.emit(False, "Failed to create duplicate database")

        except Exception as e:
            error_msg = f"Failed to create duplicate database: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            self.finished.emit(False, error_msg)


class FindDuplicatesWorker(QObject):
    """Worker for finding duplicates."""

    progress = Signal(int, str)
    results = Signal(list)
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Run the duplicate finding process."""
        try:
            self.progress.emit(30, "Searching for duplicates...")

            # This is a bit tricky since show_duplicate_photos_faiss is designed for Streamlit
            # We need to modify it to return data instead of displaying it
            # For now, let's create a mock result
            duplicates = self._find_duplicates_internal()

            self.progress.emit(100, "Duplicate search completed")
            self.results.emit(duplicates)
            self.finished.emit(True, "Success")

        except Exception as e:
            error_msg = f"Failed to find duplicates: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            self.finished.emit(False, error_msg)

    def _find_duplicates_internal(self):
        """Internal method to find duplicates and return data."""
        try:
            from db import load_duplicate_pairs

            # Get duplicate pairs from database
            pairs = load_duplicate_pairs(self.params["min_threshold"], self.params["max_threshold"])

            # Apply limit
            if self.params["limit"] > 0:
                pairs = pairs[: self.params["limit"]]

            # Convert to format expected by GUI
            duplicates = []
            for pair in pairs:
                duplicates.append({"image1_path": pair[0], "image2_path": pair[1], "similarity": pair[2] if len(pair) > 2 else 0.0})

            return duplicates

        except Exception as e:
            logger.error(f"Error in _find_duplicates_internal: {e}")
            return []
