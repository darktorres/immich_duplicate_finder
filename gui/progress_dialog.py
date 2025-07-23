"""
Progress dialog for long-running operations.
"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PySide6.QtCore import Qt


class ProgressDialog(QDialog):
    """Dialog for showing progress of long-running operations."""
    
    def __init__(self, title="Processing", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 150)
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Cancel button (optional)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        
    def update_progress(self, value, message=""):
        """Update progress bar and message."""
        if message:
            self.status_label.setText(message)
        self.progress_bar.setValue(value)
        
    def set_indeterminate(self, message="Processing..."):
        """Set progress bar to indeterminate mode."""
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)
        
    def set_determinate(self, maximum=100):
        """Set progress bar to determinate mode."""
        self.progress_bar.setRange(0, maximum)
        
    def hide_cancel_button(self):
        """Hide the cancel button."""
        self.cancel_button.setVisible(False)