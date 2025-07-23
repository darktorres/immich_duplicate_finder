"""
Widget for displaying duplicate image pairs.
"""

import os

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from logger_config import logger


class DuplicateViewer(QWidget):
    """Widget for displaying duplicate image pairs."""
    
    def __init__(self):
        super().__init__()
        self.duplicates = []
        # Set size policy to expand
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the user interface."""
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("Duplicate Image Pairs")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 10px;
                border-radius: 6px;
                margin-bottom: 10px;
            }
        """)
        self.layout.addWidget(header)
        
    def display_duplicates(self, duplicates):
        """Display the duplicate pairs."""
        self.duplicates = duplicates
        
        # Clear existing content
        self.clear_content()
        
        if not duplicates:
            no_results = QLabel("No duplicates found with current parameters.")
            no_results.setAlignment(Qt.AlignCenter)
            no_results.setStyleSheet("color: #666; font-size: 12pt; margin: 50px;")
            self.layout.addWidget(no_results)
            # Only add stretch when there are no results
            self.layout.addStretch()
            return
            
        # Add duplicate pairs
        for i, duplicate in enumerate(duplicates):
            pair_widget = self.create_duplicate_pair_widget(duplicate, i + 1)
            self.layout.addWidget(pair_widget)
            
        # Don't add stretch when there are results - let the scroll area handle it
        
    def clear_content(self):
        """Clear all content except header."""
        # Remove all widgets except the first one (header)
        while self.layout.count() > 1:
            child = self.layout.takeAt(1)
            if child.widget():
                child.widget().deleteLater()
                
    def create_duplicate_pair_widget(self, duplicate, pair_number):
        """Create widget for a single duplicate pair."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid #ddd;
                border-radius: 8px;
                margin: 8px;
                padding: 15px;
                background-color: #fafafa;
            }
            QFrame:hover {
                border-color: #4CAF50;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(15)
        
        # Pair header with better styling
        header_layout = QHBoxLayout()
        pair_label = QLabel(f"Duplicate Pair #{pair_number}")
        pair_font = QFont()
        pair_font.setPointSize(12)
        pair_font.setBold(True)
        pair_label.setFont(pair_font)
        pair_label.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(pair_label)
        
        # Similarity score with better styling
        if 'similarity' in duplicate:
            similarity_score = duplicate['similarity']
            similarity_label = QLabel(f"Similarity: {similarity_score:.1f}%")
            similarity_label.setStyleSheet("""
                color: #27ae60;
                font-weight: bold;
                background-color: #e8f5e8;
                padding: 4px 8px;
                border-radius: 4px;
            """)
            header_layout.addWidget(similarity_label)
            
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Images layout with better spacing
        images_layout = QHBoxLayout()
        images_layout.setSpacing(20)
        
        # Image 1
        img1_widget = self.create_image_widget(
            duplicate['image1_path'], 
            "Image 1"
        )
        images_layout.addWidget(img1_widget)
        
        # VS separator with styling
        vs_widget = QWidget()
        vs_layout = QVBoxLayout(vs_widget)
        vs_layout.addStretch()
        vs_label = QLabel("VS")
        vs_label.setAlignment(Qt.AlignCenter)
        vs_label.setStyleSheet("""
            font-size: 16pt;
            font-weight: bold;
            color: #7f8c8d;
            background-color: #ecf0f1;
            border-radius: 20px;
            padding: 10px;
            min-width: 40px;
            max-width: 40px;
            min-height: 40px;
            max-height: 40px;
        """)
        vs_layout.addWidget(vs_label)
        vs_layout.addStretch()
        images_layout.addWidget(vs_widget)
        
        # Image 2
        img2_widget = self.create_image_widget(
            duplicate['image2_path'], 
            "Image 2"
        )
        images_layout.addWidget(img2_widget)
        
        layout.addLayout(images_layout)
        
        # Action buttons with better styling
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)
        
        # Delete buttons
        delete1_btn = QPushButton("🗑️ Delete Image 1")
        delete1_btn.clicked.connect(lambda: self.delete_image(duplicate['image1_path']))
        delete1_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        actions_layout.addWidget(delete1_btn)
        
        delete2_btn = QPushButton("🗑️ Delete Image 2")
        delete2_btn.clicked.connect(lambda: self.delete_image(duplicate['image2_path']))
        delete2_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        actions_layout.addWidget(delete2_btn)
        
        # Spacer
        actions_layout.addStretch()
        
        # Folder buttons
        open_folder1_btn = QPushButton("📁 Open Folder 1")
        open_folder1_btn.clicked.connect(lambda: self.open_folder(duplicate['image1_path']))
        open_folder1_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
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
        actions_layout.addWidget(open_folder1_btn)
        
        open_folder2_btn = QPushButton("📁 Open Folder 2")
        open_folder2_btn.clicked.connect(lambda: self.open_folder(duplicate['image2_path']))
        open_folder2_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
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
        actions_layout.addWidget(open_folder2_btn)
        
        layout.addLayout(actions_layout)
        
        return frame
        
    def create_image_widget(self, image_path, title):
        """Create widget for displaying a single image."""
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title with better styling
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("""
            color: #2c3e50;
            background-color: #ecf0f1;
            padding: 6px;
            border-radius: 4px;
            margin-bottom: 5px;
        """)
        layout.addWidget(title_label)
        
        # Image container
        image_container = QWidget()
        image_container.setStyleSheet("""
            QWidget {
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                background-color: #f8f9fa;
            }
        """)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        # Image
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(280, 280)
        image_label.setMaximumSize(350, 350)
        image_label.setScaledContents(False)
        
        # Load and display image
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale image to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    image_label.maximumSize(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("❌ Failed to load image")
                image_label.setStyleSheet("color: #e74c3c; font-size: 12pt;")
        else:
            image_label.setText("❌ Image not found")
            image_label.setStyleSheet("color: #e74c3c; font-size: 12pt;")
            
        image_layout.addWidget(image_label)
        layout.addWidget(image_container)
        
        # File info section
        info_widget = QWidget()
        info_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        info_layout = QVBoxLayout(info_widget)
        info_layout.setSpacing(3)
        
        # File path (shortened)
        filename = os.path.basename(image_path)
        folder = os.path.dirname(image_path)
        short_folder = "..." + folder[-30:] if len(folder) > 30 else folder
        
        filename_label = QLabel(f"📄 {filename}")
        filename_label.setStyleSheet("color: #2c3e50; font-weight: bold; font-size: 9pt;")
        info_layout.addWidget(filename_label)
        
        path_label = QLabel(f"📁 {short_folder}")
        path_label.setStyleSheet("color: #7f8c8d; font-size: 8pt;")
        path_label.setWordWrap(True)
        info_layout.addWidget(path_label)
        
        # File size and dimensions
        try:
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                size_mb = file_size / (1024 * 1024)
                
                # Get image dimensions
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        dimensions_text = f"📐 {width}×{height} • 💾 {size_mb:.1f} MB"
                except Exception:
                    dimensions_text = f"💾 {size_mb:.1f} MB"
                
                size_label = QLabel(dimensions_text)
                size_label.setStyleSheet("color: #7f8c8d; font-size: 8pt;")
                info_layout.addWidget(size_label)
        except Exception as e:
            logger.warning(f"Could not get file info for {image_path}: {e}")
            
        layout.addWidget(info_widget)
        
        return widget
        
    def delete_image(self, image_path):
        """Delete an image file after confirmation."""
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete this image?\n\n{image_path}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                os.remove(image_path)
                QMessageBox.information(self, "Success", "Image deleted successfully.")
                logger.info(f"Deleted image: {image_path}")
                # Refresh the display
                # Note: In a full implementation, you'd want to refresh the duplicate search
            except Exception as e:
                error_msg = f"Failed to delete image: {str(e)}"
                QMessageBox.critical(self, "Error", error_msg)
                logger.error(error_msg)
                
    def open_folder(self, image_path):
        """Open the folder containing the image."""
        try:
            folder_path = os.path.dirname(image_path)
            if os.path.exists(folder_path):
                # Windows
                if os.name == 'nt':
                    os.startfile(folder_path)
                # macOS
                elif os.name == 'posix' and os.uname().sysname == 'Darwin':
                    os.system(f'open "{folder_path}"')
                # Linux
                else:
                    os.system(f'xdg-open "{folder_path}"')
            else:
                QMessageBox.warning(self, "Warning", "Folder does not exist.")
        except Exception as e:
            error_msg = f"Failed to open folder: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            logger.error(error_msg)