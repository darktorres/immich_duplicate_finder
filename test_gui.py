#!/usr/bin/env python3
"""
Simple test for GUI components without full startup.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel

def main():
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Test GUI")
    window.setCentralWidget(QLabel("Hello, PySide6!"))
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())