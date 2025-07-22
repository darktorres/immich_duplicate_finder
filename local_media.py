import os
from datetime import datetime
from typing import List, Optional, Tuple

from PIL import Image, ImageFile, UnidentifiedImageError
from pillow_heif import register_heif_opener

from logger_config import logger

# Supported image extensions (case-insensitive)
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic", ".heif", ".dng")


def setup_local_media():
    """Initializes support for various image formats."""
    register_heif_opener()
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_media_files(folder_path: str) -> List[str]:
    """
    Recursively finds all supported image files in a given folder.

    Args:
        folder_path: Path to the folder to search

    Returns:
        List of file paths to supported image files
    """
    if not folder_path or not os.path.isdir(folder_path):
        logger.error(f"Folder not found or invalid: {folder_path}")
        return []

    filepaths = []
    try:
        logger.info(f"Scanning folder for media files: {folder_path}")

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    # Validate the file exists and is readable
                    if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                        filepaths.append(full_path)
                    else:
                        logger.warning(f"File not accessible: {full_path}")

        logger.info(f"Found {len(filepaths)} media files in {folder_path}")
    except (OSError, PermissionError) as e:
        logger.error(f"Error accessing folder {folder_path}: {e}")
        return []

    return filepaths


def load_image(file_path: str) -> Optional[Image.Image]:
    """
    Loads an image from a file path into a Pillow Image object.

    Args:
        file_path: Path to the image file

    Returns:
        PIL Image object or None if loading fails
    """
    try:
        if not os.path.isfile(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None

        image = Image.open(file_path)
        # load() is called to read the image data. This is important for some formats.
        image.load()
        logger.debug(f"Successfully loaded image: {file_path}")
        return image
    except (UnidentifiedImageError, FileNotFoundError, IsADirectoryError, OSError) as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        return None


def bytes_to_megabytes(bytes_size: Optional[int]) -> str:
    """
    Converts bytes to megabytes (MB) and formats to 3 decimal places.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string with MB suffix
    """
    if bytes_size is None:
        return "0.000 MB"
    megabytes = bytes_size / (1024 * 1024)
    return f"{megabytes:.3f} MB"


def get_file_info(file_path: str) -> Tuple[str, str, str, str, str]:
    """
    Gathers and returns information for a given file.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (file_size, file_name, resolution, creation_date, file_path)
    """
    try:
        if not os.path.isfile(file_path):
            logger.error(f"File does not exist: {file_path}")
            return "Unknown", os.path.basename(file_path), "Unknown", "Unknown", file_path

        file_size = os.path.getsize(file_path)
        formatted_file_size = bytes_to_megabytes(file_size)
        file_name = os.path.basename(file_path)

        resolution = "Unknown"
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                # The original code used H x W, so we maintain that format.
                resolution = f"{height} x {width}"
        except Exception as img_error:
            logger.warning(f"Could not get image dimensions for {file_path}: {img_error}")

        creation_timestamp = os.path.getmtime(file_path)
        creation_date = datetime.fromtimestamp(creation_timestamp).isoformat()

        # Returns a tuple consistent with the data expected by the UI.
        return formatted_file_size, file_name, resolution, creation_date, file_path
    except Exception as e:
        logger.error(f"Error getting info for file {file_path}: {e}")
        return "Unknown", os.path.basename(file_path), "Unknown", "Unknown", file_path


def delete_file(file_path: str) -> bool:
    """
    Deletes a file from the filesystem.

    Args:
        file_path: Path to the file to delete

    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        if not os.path.isfile(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False

        os.remove(file_path)
        logger.info(f"Successfully deleted file: {file_path}")
        return True
    except OSError as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False
