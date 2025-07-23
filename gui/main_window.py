"""
Main window for the Local Duplicate Finder GUI application.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QWidget,
)

from gui.main_content import MainContentWidget
from gui.sidebar import SidebarWidget
from logger_config import logger


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.progress_dialog = None
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Local Duplicate Finder")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Create sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.setMaximumWidth(350)
        self.sidebar.setMinimumWidth(300)
        splitter.addWidget(self.sidebar)

        # Create main content area
        self.main_content = MainContentWidget()
        splitter.addWidget(self.main_content)

        # Set splitter proportions (sidebar: 25%, main: 75%)
        splitter.setSizes([300, 900])

        # Create menu bar
        self.setup_menu_bar()

        # Create status bar
        self.setup_status_bar()

    def setup_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def setup_connections(self):
        """Connect signals and slots."""
        # Connect sidebar signals to main content
        self.sidebar.folder_changed.connect(self.main_content.set_folder_path)
        self.sidebar.create_faiss_index.connect(self.handle_create_faiss_index)
        self.sidebar.create_duplicate_db.connect(self.handle_create_duplicate_db)
        self.sidebar.find_duplicates.connect(self.handle_find_duplicates)

        # Connect main content signals
        self.main_content.status_message.connect(self.status_bar.showMessage)

    def handle_create_faiss_index(self, folder_path):
        """Handle FAISS index creation."""
        logger.info(f"Creating FAISS index for folder: {folder_path}")
        self.status_bar.showMessage("Creating FAISS index...")
        self.main_content.create_faiss_index(folder_path)

    def handle_create_duplicate_db(self):
        """Handle duplicate database creation."""
        logger.info("Creating duplicate database")
        self.status_bar.showMessage("Creating duplicate database...")
        self.main_content.create_duplicate_db()

    def handle_find_duplicates(self, params):
        """Handle finding duplicates."""
        logger.info(f"Finding duplicates with params: {params}")
        self.status_bar.showMessage("Finding duplicates...")
        self.main_content.find_duplicates(params)

    def show_about(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About Local Duplicate Finder",
            """
            <h3>Local Duplicate Finder v0.3.0-enhanced</h3>
            <p>A comprehensive solution for identifying and managing duplicate photos using advanced hashing detection and ML technologies.</p>
            <p><b>Features:</b></p>
            <ul>
            <li>FAISS Vector Database with ResNet152</li>
            <li>High accuracy duplicate detection</li>
            <li>Local processing for privacy</li>
            <li>Performance optimized</li>
            </ul>
            """,
        )

    def closeEvent(self, event):
        """Handle application close event."""
        logger.info("Application closing")
        event.accept()
