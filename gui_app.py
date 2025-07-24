#!/usr/bin/env python3
"""
Main GUI application for Local Duplicate Finder using PySide6.
"""

import os
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import GUI components
from gui.main_window import MainWindow


def setup_application():
    """Initialize the application and perform startup configurations."""
    try:
        # Import only when needed to avoid potential import issues
        from db import startup_db_configurations, startup_processed_duplicate_faiss_db
        from local_media import setup_local_media
        from logger_config import logger

        startup_db_configurations()
        startup_processed_duplicate_faiss_db()
        setup_local_media()
        logger.info("Application startup completed successfully")
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        raise


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Local Duplicate Finder")
    app.setApplicationVersion("v0.3.0-enhanced")
    app.setOrganizationName("Local Duplicate Finder")

    # Set application icon if available
    icon_path = Path("assets/icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    try:
        # Initialize backend
        setup_application()

        # Create and show main window
        window = MainWindow()
        window.show()

        # Start event loop
        return app.exec()

    except Exception as e:
        print(f"Application failed to start: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
