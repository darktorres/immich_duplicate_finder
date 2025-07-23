"""
Main content area widget for displaying results and progress.
"""

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.duplicate_viewer import DuplicateViewer
from gui.workers import DuplicateDbWorker, FaissIndexWorker, FindDuplicatesWorker
from logger_config import logger


class MainContentWidget(QWidget):
    """Main content area for displaying results and operations."""

    # Signals
    status_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.worker_thread = None
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main content area (should expand to fill available space)
        self.main_area = QWidget()
        self.setup_main_area()
        layout.addWidget(self.main_area, 1)  # Stretch factor 1 - can expand

        # Log area (anchored to bottom, fixed size)
        self.log_area = QWidget()
        self.setup_log_area()
        layout.addWidget(self.log_area, 0)  # No stretch - fixed size

    def setup_main_area(self):
        """Setup the main content display area."""
        layout = QVBoxLayout(self.main_area)
        layout.setSpacing(10)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Welcome section with better styling
        welcome_widget = QWidget()
        welcome_widget.setStyleSheet(
            """
            QWidget {
                background-color: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #dee2e6;
                margin: 0px 5px 5px 5px;
            }
        """
        )
        # Set size policy to prevent vertical expansion
        welcome_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setSpacing(10)
        welcome_layout.setContentsMargins(20, 20, 20, 20)

        # Welcome message
        self.welcome_label = QLabel("🖼️ Welcome to Local Duplicate Finder")
        welcome_font = QFont()
        welcome_font.setPointSize(18)
        welcome_font.setBold(True)
        self.welcome_label.setFont(welcome_font)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        welcome_layout.addWidget(self.welcome_label)

        # Instructions with better formatting
        instructions = QLabel(
            "📁 1. Select a folder containing images\n"
            "🔍 2. Create/Update FAISS index\n"
            "💾 3. Create/Update duplicate database\n"
            "⚙️ 4. Set search parameters and find duplicates"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet(
            """
            color: #495057;
            font-size: 11pt;
            line-height: 1.5;
            background-color: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        """
        )
        welcome_layout.addWidget(instructions)

        # Add welcome widget with no stretch factor (fixed size)
        layout.addWidget(welcome_widget, 0)

        # Progress section (initially hidden)
        self.progress_widget = QWidget()
        self.progress_widget.setVisible(False)
        self.progress_widget.setStyleSheet(
            """
            QWidget {
                background-color: #e3f2fd;
                border-radius: 8px;
                border: 1px solid #bbdefb;
                margin: 0px 5px 5px 5px;
                padding: 15px;
            }
        """
        )
        # Set size policy to prevent vertical expansion
        self.progress_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        progress_layout = QVBoxLayout(self.progress_widget)
        progress_layout.setSpacing(10)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet(
            """
            color: #1976d2;
            font-weight: bold;
            font-size: 11pt;
        """
        )
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #bbdefb;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                color: #1976d2;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                border-radius: 3px;
            }
        """
        )
        progress_layout.addWidget(self.progress_bar)

        # Add progress widget with no stretch factor (fixed size)
        layout.addWidget(self.progress_widget, 0)

        # Results area (initially hidden) - make it expand to fill remaining space
        self.results_area = QScrollArea()
        self.results_area.setVisible(False)
        self.results_area.setWidgetResizable(True)
        self.results_area.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
            }
        """
        )
        # Add with stretch factor to make it expand and fill remaining space
        layout.addWidget(self.results_area, 1)

        # Duplicate viewer
        self.duplicate_viewer = DuplicateViewer()
        self.results_area.setWidget(self.duplicate_viewer)

    def setup_log_area(self):
        """Setup the log display area."""
        log_layout = QVBoxLayout(self.log_area)
        log_layout.setSpacing(2)  # Minimal spacing
        log_layout.setContentsMargins(5, 2, 5, 5)  # Minimal margins

        # Log header
        log_header = QLabel("Application Log")
        log_header_font = QFont()
        log_header_font.setBold(True)
        log_header.setFont(log_header_font)
        log_header.setStyleSheet("margin: 0px; padding: 0px;")  # Remove all margins and padding
        log_layout.addWidget(log_header)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)  # Compact height
        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
            }
        """
        )
        log_layout.addWidget(self.log_text)

        # Clear log button - make it smaller
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        clear_btn.setMaximumHeight(25)
        clear_btn.setStyleSheet("margin: 0px; padding: 2px 8px;")
        log_layout.addWidget(clear_btn)

        # Set size policy to prevent expansion
        self.log_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_folder_path(self, folder_path):
        """Set the current folder path."""
        self.current_folder = folder_path
        self.log_message(f"Folder selected: {folder_path}")

    def show_progress(self, message="Processing..."):
        """Show progress bar and message."""
        self.progress_widget.setVisible(True)
        self.progress_label.setText(message)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

    def hide_progress(self):
        """Hide progress bar and message."""
        self.progress_widget.setVisible(False)

    def update_progress(self, value, message=""):
        """Update progress bar value and message."""
        if message:
            self.progress_label.setText(message)
        if value >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)

    def log_message(self, message):
        """Add message to log area."""
        self.log_text.append(f"[{QTimer().remainingTime()}] {message}")

    def create_faiss_index(self, folder_path):
        """Create FAISS index in background thread."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_message("Another operation is already running")
            return

        self.show_progress("Creating FAISS index...")
        self.log_message("Starting FAISS index creation")

        # Clean up previous thread if it exists
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

        self.worker_thread = QThread()
        self.worker = FaissIndexWorker(folder_path)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_faiss_index_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: setattr(self, "worker_thread", None))

        self.worker_thread.start()

    def create_duplicate_db(self):
        """Create duplicate database in background thread."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_message("Another operation is already running")
            return

        self.show_progress("Creating duplicate database...")
        self.log_message("Starting duplicate database creation")

        # Clean up previous thread if it exists
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

        self.worker_thread = QThread()
        self.worker = DuplicateDbWorker()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_duplicate_db_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: setattr(self, "worker_thread", None))

        self.worker_thread.start()

    def find_duplicates(self, params):
        """Find duplicates in background thread."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.log_message("Another operation is already running")
            return

        self.show_progress("Finding duplicates...")
        self.log_message("Starting duplicate search")

        # Clean up previous thread if it exists
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

        self.worker_thread = QThread()
        self.worker = FindDuplicatesWorker(params)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.results.connect(self.on_duplicates_found)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: setattr(self, "worker_thread", None))

        self.worker_thread.start()

    def on_faiss_index_finished(self, success, message):
        """Handle FAISS index creation completion."""
        self.hide_progress()
        if success:
            self.log_message("FAISS index created successfully")
            self.status_message.emit("FAISS index created successfully")
        else:
            self.log_message(f"FAISS index creation failed: {message}")
            self.status_message.emit("FAISS index creation failed")

    def on_duplicate_db_finished(self, success, message):
        """Handle duplicate database creation completion."""
        self.hide_progress()
        if success:
            self.log_message("Duplicate database created successfully")
            self.status_message.emit("Duplicate database created successfully")
        else:
            self.log_message(f"Duplicate database creation failed: {message}")
            self.status_message.emit("Duplicate database creation failed")

    def on_duplicates_found(self, duplicates):
        """Handle duplicate search results."""
        self.hide_progress()
        self.log_message(f"Found {len(duplicates)} duplicate pairs")

        # Limit results to prevent GUI freeze
        max_display = 50  # Only display first 50 pairs to prevent freeze
        if len(duplicates) > max_display:
            display_duplicates = duplicates[:max_display]
            self.status_message.emit(f"Showing {max_display} of {len(duplicates)} duplicate pairs")
            self.log_message(f"Displaying first {max_display} pairs (out of {len(duplicates)} total)")
        else:
            display_duplicates = duplicates
            self.status_message.emit(f"Found {len(duplicates)} duplicate pairs")

        # Show results and ensure proper sizing
        self.results_area.setVisible(True)
        self.duplicate_viewer.display_duplicates(display_duplicates)

        # Force layout update to ensure proper sizing
        self.results_area.updateGeometry()
        self.duplicate_viewer.updateGeometry()

    def on_worker_error(self, error_message):
        """Handle worker thread errors."""
        self.hide_progress()
        self.log_message(f"Error: {error_message}")
        self.status_message.emit(f"Error: {error_message}")
        logger.error(f"Worker error: {error_message}")
