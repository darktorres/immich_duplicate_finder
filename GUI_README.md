# Local Duplicate Finder - GUI Version

This is a native desktop GUI version of the Local Duplicate Finder, built with PySide6 (Qt for Python).

## Features

- **Native Desktop Interface**: Clean, responsive desktop application
- **Background Processing**: All heavy operations run in background threads
- **Real-time Progress**: Live progress updates for all operations
- **Image Preview**: Side-by-side comparison of duplicate images
- **File Management**: Delete duplicates or open containing folders
- **Settings Persistence**: Automatically saves your preferences

## Running the GUI

### Quick Start
```bash
# Install dependencies
poetry install

# Run the GUI application
poetry run poe gui
```

### Manual Run
```bash
poetry run python gui_minimal.py
```

## How to Use

1. **Select Folder**: Click "Browse" to select your image folder
2. **Create FAISS Index**: Click "Create/Update FAISS Index" to analyze your images
3. **Generate Duplicate DB**: Click "Create/Update Duplicate DB" to find similarities
4. **Set Parameters**: Adjust similarity thresholds and result limits
5. **Find Duplicates**: Click "Find Duplicate Photos" to see results
6. **Review Results**: Use the image viewer to compare and manage duplicates

## GUI Components

### Sidebar
- **Folder Selection**: Browse and select image directories
- **FAISS Operations**: Create and update the search index
- **Search Parameters**: Configure similarity thresholds and result limits
- **Version Info**: Application version and log level

### Main Content
- **Welcome Screen**: Instructions and current status
- **Progress Display**: Real-time operation progress
- **Results Viewer**: Side-by-side duplicate image comparison
- **Log Area**: Application logs and status messages

### Duplicate Viewer
- **Image Comparison**: Side-by-side view of duplicate pairs
- **Similarity Scores**: Numerical similarity ratings
- **File Actions**: Delete images or open containing folders
- **File Information**: Path and size details

## Advantages over Streamlit Version

- **Better Performance**: Native desktop application with optimized UI
- **No Browser Required**: Standalone desktop application
- **Better Threading**: Background operations don't block the UI
- **Native File Operations**: Direct integration with OS file management
- **Persistent Settings**: Automatically saves configuration
- **Better Error Handling**: More robust error reporting and recovery

## Technical Details

- **Framework**: PySide6 (Qt 6.8.1)
- **Threading**: QThread for background operations
- **Image Processing**: Same ViT-B/16 model as Streamlit version
- **Database**: SQLite for settings and duplicate storage
- **FAISS Integration**: Efficient similarity search

## Troubleshooting

### Common Issues

1. **GPU Not Detected**: The app will automatically fall back to CPU
2. **Large Image Collections**: Processing may take time, check progress bar
3. **Memory Issues**: Close other applications if processing large datasets

### Logs
Check the application log area at the bottom of the window for detailed error messages.

## Development

The GUI is modular with separate components:

- `gui/main_window.py` - Main application window
- `gui/sidebar.py` - Control panel and settings
- `gui/main_content.py` - Results display and progress
- `gui/duplicate_viewer.py` - Image comparison interface
- `gui/workers.py` - Background processing threads
- `gui/image_processing.py` - Core image analysis (Streamlit-free)

## Future Enhancements

- Batch operations for multiple duplicates
- Custom similarity algorithms
- Export/import of duplicate lists
- Advanced filtering options
- Thumbnail caching for faster preview