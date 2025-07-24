"""
Memory-optimized worker threads for background operations.
"""

import gc
from PySide6.QtCore import QObject, Signal

from gui.image_processing import calculate_faiss_index_gui, generate_duplicate_db_gui
from local_media import get_media_files
from logger_config import logger
from memory_config import MEMORY_CONFIG


class FaissIndexWorker(QObject):
    """Memory-optimized worker for creating FAISS index."""

    progress = Signal(int, str)
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        """Run the FAISS index creation with memory optimization."""
        try:
            self.progress.emit(10, "Getting media files...")
            media_files = get_media_files(self.folder_path)

            if not media_files:
                self.finished.emit(False, "No media files found")
                return

            self.progress.emit(30, f"Processing {len(media_files)} files with memory optimization...")
            logger.info(f"Starting FAISS index creation with batch size: {MEMORY_CONFIG.batch_size}")

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
        finally:
            # Cleanup after processing
            if MEMORY_CONFIG.aggressive_gc:
                gc.collect()


class DuplicateDbWorker(QObject):
    """Memory-optimized worker for creating duplicate database."""

    progress = Signal(int, str)
    finished = Signal(bool, str)
    error = Signal(str)

    def run(self):
        """Run the duplicate database creation with memory optimization."""
        try:
            logger.info("Starting duplicate database creation with memory optimization")

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
        finally:
            # Cleanup after processing
            if MEMORY_CONFIG.aggressive_gc:
                gc.collect()


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
        """Internal method to find duplicates and return data with memory optimization."""
        try:
            from db import load_duplicate_pairs

            # Get duplicate pairs from database
            pairs = load_duplicate_pairs(self.params["min_threshold"], self.params["max_threshold"])

            # Apply limit with memory consideration
            limit = self.params["limit"]
            if limit > 0:
                # For memory optimization, limit the results more aggressively if needed
                max_safe_limit = min(limit, 100)  # Cap at 100 for memory safety
                pairs = pairs[:max_safe_limit]
                
                if limit > max_safe_limit:
                    logger.info(f"Limited results to {max_safe_limit} for memory optimization (requested: {limit})")

            # Convert to format expected by GUI with memory-efficient processing
            duplicates = []
            for i, pair in enumerate(pairs):
                # Process in smaller batches to avoid memory spikes
                if i > 0 and i % 20 == 0:
                    if MEMORY_CONFIG.aggressive_gc:
                        gc.collect()
                
                duplicate_entry = {
                    "image1_path": pair[0],
                    "image2_path": pair[1],
                    "similarity": pair[2] if len(pair) > 2 else 0.0
                }
                duplicates.append(duplicate_entry)

            logger.info(f"Found {len(duplicates)} duplicate pairs")
            return duplicates

        except Exception as e:
            logger.error(f"Error in _find_duplicates_internal: {e}")
            return []
