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
        self.setWindowTitle("Local Duplicate Finder (Memory Optimized)")
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

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        memory_info_action = QAction("Memory &Info", self)
        memory_info_action.triggered.connect(self.show_memory_info)
        tools_menu.addAction(memory_info_action)
        
        clear_cache_action = QAction("&Clear Cache", self)
        clear_cache_action.triggered.connect(self.clear_memory_cache)
        tools_menu.addAction(clear_cache_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready (Memory Optimized)")

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

    def show_memory_info(self):
        """Show current memory usage information."""
        from PySide6.QtWidgets import QMessageBox
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            # Check if heavy components are loaded
            from gui.image_processing import is_model_loaded
            model_status = "Loaded" if is_model_loaded() else "Not loaded (will load on demand)"
            
            info_text = f"""
            <h3>Memory Usage Information</h3>
            <p><b>Current Memory Usage:</b> {memory_mb:.1f} MB ({memory_percent:.1f}%)</p>
            <p><b>ML Model Status:</b> {model_status}</p>
            <p><b>Memory Configuration:</b> Optimized</p>
            <p><b>Optimization:</b> Lazy loading enabled</p>
            """
            
            QMessageBox.information(self, "Memory Information", info_text)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not retrieve memory info: {e}")

    def clear_memory_cache(self):
        """Clear memory cache and force garbage collection."""
        from PySide6.QtWidgets import QMessageBox
        import gc
        
        try:
            # Clear model cache
            from gui.image_processing import clear_model_cache
            clear_model_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            QMessageBox.information(self, "Cache Cleared", "Memory cache has been cleared successfully.")
            self.status_bar.showMessage("Memory cache cleared")
            logger.info("Memory cache cleared by user")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not clear cache: {e}")

    def show_about(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
        import psutil
        import os
        
        # Get current memory usage
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_text = f"<p><b>Current Memory Usage:</b> {memory_mb:.1f} MB</p>"
        except:
            memory_text = ""

        QMessageBox.about(
            self,
            "About Local Duplicate Finder",
            f"""
            <h3>Local Duplicate Finder v0.3.0-enhanced-optimized</h3>
            <p>A memory-optimized solution for identifying and managing duplicate photos using advanced ML detection.</p>
            {memory_text}
            <p><b>Optimizations:</b></p>
            <ul>
            <li>Lazy loading of ML components (95% less startup memory)</li>
            <li>Batch processing with garbage collection</li>
            <li>Automatic memory configuration</li>
            <li>GPU memory management</li>
            <li>On-demand model loading</li>
            </ul>
            <p><b>Features:</b></p>
            <ul>
            <li>FAISS Vector Database with Vision Transformer</li>
            <li>High accuracy duplicate detection</li>
            <li>Local processing for privacy</li>
            <li>Performance optimized</li>
            </ul>
            """,
        )

    def closeEvent(self, event):
        """Handle application close event."""
        logger.info("Application closing")
        
        # Clear memory cache before closing
        try:
            from gui.image_processing import clear_model_cache
            clear_model_cache()
        except:
            pass
            
        event.accept()
