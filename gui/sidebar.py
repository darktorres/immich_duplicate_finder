"""
Sidebar widget for the Local Duplicate Finder GUI.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox, QFileDialog,
    QFrame
)
from PySide6.QtCore import Signal, QSettings, Qt
from PySide6.QtGui import QFont

from validation import validate_folder_path, validate_threshold_range, validate_limit
from logger_config import logger


class SidebarWidget(QWidget):
    """Sidebar widget containing controls and settings."""
    
    # Signals
    folder_changed = Signal(str)
    create_faiss_index = Signal(str)
    create_duplicate_db = Signal()
    find_duplicates = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings()
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title with better styling
        title = QLabel("🔍 Local Duplicate Finder")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: #2c3e50;
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 5px;
        """)
        layout.addWidget(title)
        
        # Folder selection group
        self.setup_folder_group(layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # FAISS operations group
        self.setup_faiss_group(layout)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)
        
        # Search parameters group
        self.setup_search_group(layout)
        
        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator3)
        
        # Version info
        self.setup_version_info(layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
    def setup_folder_group(self, parent_layout):
        """Setup folder selection group."""
        group = QGroupBox("📁 Folder Selection")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
        """)
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Folder path input
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select folder containing images...")
        self.folder_input.textChanged.connect(self.on_folder_changed)
        self.folder_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 9pt;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
        """)
        
        browse_btn = QPushButton("📂 Browse")
        browse_btn.clicked.connect(self.browse_folder)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)
        
        parent_layout.addWidget(group)
        
    def setup_faiss_group(self, parent_layout):
        """Setup FAISS operations group."""
        group = QGroupBox("🔍 FAISS Operations")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
        """)
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Create/Update FAISS index button
        self.faiss_index_btn = QPushButton("🚀 Create/Update FAISS Index")
        self.faiss_index_btn.clicked.connect(self.on_create_faiss_index)
        self.faiss_index_btn.setEnabled(False)
        self.faiss_index_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #229954;
            }
            QPushButton:pressed:enabled {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #7f8c8d;
            }
        """)
        layout.addWidget(self.faiss_index_btn)
        
        # Create/Update duplicate DB button
        self.duplicate_db_btn = QPushButton("💾 Create/Update Duplicate DB")
        self.duplicate_db_btn.clicked.connect(self.on_create_duplicate_db)
        self.duplicate_db_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        layout.addWidget(self.duplicate_db_btn)
        
        parent_layout.addWidget(group)
        
    def setup_search_group(self, parent_layout):
        """Setup search parameters group."""
        group = QGroupBox("Search Parameters")
        layout = QVBoxLayout(group)
        
        # Minimum threshold
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Threshold:"))
        self.min_threshold = QDoubleSpinBox()
        self.min_threshold.setRange(0.0, 100.0)
        self.min_threshold.setSingleStep(0.01)
        self.min_threshold.setDecimals(2)
        self.min_threshold.setValue(0.0)
        self.min_threshold.valueChanged.connect(self.validate_inputs)
        min_layout.addWidget(self.min_threshold)
        layout.addLayout(min_layout)
        
        # Maximum threshold
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max Threshold:"))
        self.max_threshold = QDoubleSpinBox()
        self.max_threshold.setRange(0.0, 100.0)
        self.max_threshold.setSingleStep(0.01)
        self.max_threshold.setDecimals(2)
        self.max_threshold.setValue(100.0)
        self.max_threshold.valueChanged.connect(self.validate_inputs)
        max_layout.addWidget(self.max_threshold)
        layout.addLayout(max_layout)
        
        # Number of pairs
        pairs_layout = QHBoxLayout()
        pairs_layout.addWidget(QLabel("Pairs to Display:"))
        self.pairs_limit = QSpinBox()
        self.pairs_limit.setRange(1, 1000)
        self.pairs_limit.setValue(10)
        self.pairs_limit.valueChanged.connect(self.validate_inputs)
        pairs_layout.addWidget(self.pairs_limit)
        layout.addLayout(pairs_layout)
        
        # Find duplicates button
        self.find_duplicates_btn = QPushButton("🔎 Find Duplicate Photos")
        self.find_duplicates_btn.clicked.connect(self.on_find_duplicates)
        self.find_duplicates_btn.setEnabled(False)
        self.find_duplicates_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover:enabled {
                background-color: #8e44ad;
            }
            QPushButton:pressed:enabled {
                background-color: #7d3c98;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #7f8c8d;
            }
        """)
        layout.addWidget(self.find_duplicates_btn)
        
        parent_layout.addWidget(group)
        
    def setup_version_info(self, parent_layout):
        """Setup version information."""
        group = QGroupBox("Information")
        layout = QVBoxLayout(group)
        
        version_label = QLabel("Version: v0.3.0-enhanced")
        layout.addWidget(version_label)
        
        # Log level info
        from logger_config import logger
        log_level_name = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}.get(logger.level, "UNKNOWN")
        log_label = QLabel(f"Log Level: {log_level_name}")
        layout.addWidget(log_label)
        
        parent_layout.addWidget(group)
        
    def browse_folder(self):
        """Open folder browser dialog."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            self.folder_input.text() or ""
        )
        if folder:
            self.folder_input.setText(folder)
            
    def on_folder_changed(self, folder_path):
        """Handle folder path change."""
        is_valid, _ = validate_folder_path(folder_path) if folder_path else (False, "")
        self.faiss_index_btn.setEnabled(is_valid)
        self.validate_inputs()
        
        if is_valid:
            self.folder_changed.emit(folder_path)
            
    def validate_inputs(self):
        """Validate all inputs and enable/disable find button."""
        folder_path = self.folder_input.text()
        min_thresh = self.min_threshold.value()
        max_thresh = self.max_threshold.value()
        limit = self.pairs_limit.value()
        
        folder_valid, _ = validate_folder_path(folder_path) if folder_path else (False, "")
        threshold_valid, _ = validate_threshold_range(min_thresh, max_thresh)
        limit_valid, _ = validate_limit(limit)
        
        self.find_duplicates_btn.setEnabled(folder_valid and threshold_valid and limit_valid)
        
    def on_create_faiss_index(self):
        """Handle create FAISS index button click."""
        folder_path = self.folder_input.text()
        if folder_path:
            self.create_faiss_index.emit(folder_path)
            
    def on_create_duplicate_db(self):
        """Handle create duplicate DB button click."""
        self.create_duplicate_db.emit()
        
    def on_find_duplicates(self):
        """Handle find duplicates button click."""
        params = {
            'limit': self.pairs_limit.value(),
            'min_threshold': self.min_threshold.value(),
            'max_threshold': self.max_threshold.value()
        }
        self.find_duplicates.emit(params)
        
    def load_settings(self):
        """Load settings from QSettings."""
        folder_path = self.settings.value("folder_path", "")
        if folder_path:
            self.folder_input.setText(folder_path)
            
        self.min_threshold.setValue(float(self.settings.value("min_threshold", 0.0)))
        self.max_threshold.setValue(float(self.settings.value("max_threshold", 100.0)))
        self.pairs_limit.setValue(int(self.settings.value("pairs_limit", 10)))
        
    def save_settings(self):
        """Save settings to QSettings."""
        self.settings.setValue("folder_path", self.folder_input.text())
        self.settings.setValue("min_threshold", self.min_threshold.value())
        self.settings.setValue("max_threshold", self.max_threshold.value())
        self.settings.setValue("pairs_limit", self.pairs_limit.value())
        
    def closeEvent(self, event):
        """Handle widget close event."""
        self.save_settings()
        super().closeEvent(event)