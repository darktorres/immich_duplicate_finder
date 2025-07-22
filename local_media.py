import os
from datetime import datetime

from PIL import Image, ImageFile, UnidentifiedImageError
from pillow_heif import register_heif_opener

# Supported image extensions (case-insensitive)
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic", ".heif", ".dng")


def setup_local_media():
    """Initializes support for various image formats."""
    register_heif_opener()
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_media_files(folder_path):
    """Recursively finds all supported image files in a given folder."""
    if not folder_path or not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return []

    filepaths = []
    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    # Validate the file exists and is readable
                    if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                        filepaths.append(full_path)
    except (OSError, PermissionError) as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return []
    
    return filepaths


def load_image(file_path):
    """Loads an image from a file path into a Pillow Image object."""
    try:
        image = Image.open(file_path)
        # load() is called to read the image data. This is important for some formats.
        image.load()
        return image
    except (UnidentifiedImageError, FileNotFoundError, IsADirectoryError, OSError) as e:
        print(f"Failed to load image {file_path}. Error: {e}")
        return None


def bytes_to_megabytes(bytes_size):
    """Converts bytes to megabytes (MB) and formats to 3 decimal places."""
    if bytes_size is None:
        return "0.000 MB"
    megabytes = bytes_size / (1024 * 1024)
    return f"{megabytes:.3f} MB"


def get_file_info(file_path):
    """Gathers and returns information for a given file."""
    try:
        file_size = os.path.getsize(file_path)
        formatted_file_size = bytes_to_megabytes(file_size)
        file_name = os.path.basename(file_path)

        resolution = "Unknown"
        with Image.open(file_path) as img:
            width, height = img.size
            # The original code used H x W, so we maintain that format.
            resolution = f"{height} x {width}"

        creation_timestamp = os.path.getmtime(file_path)
        creation_date = datetime.fromtimestamp(creation_timestamp).isoformat()

        # Returns a tuple consistent with the data expected by the UI.
        # Original Immich fields like lens, offline, trashed, favorite are omitted.
        return formatted_file_size, file_name, resolution, creation_date, file_path
    except Exception as e:
        print(f"Error getting info for file {file_path}: {e}")
        return "Unknown", os.path.basename(file_path), "Unknown", "Unknown", file_path


def delete_file(file_path):
    """Deletes a file from the filesystem."""
    try:
        os.remove(file_path)
        print(f"Successfully deleted file: {file_path}")
        return True
    except OSError as e:
        print(f"Error deleting file {file_path}: {e}")
        return False
