"""Input validation utilities for the duplicate finder application."""

import os
from pathlib import Path
from typing import Optional, Tuple

from logger_config import logger


def validate_folder_path(folder_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a folder path exists and is accessible.

    Args:
        folder_path: Path to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not folder_path:
        return False, "Folder path cannot be empty"

    if not folder_path.strip():
        return False, "Folder path cannot be only whitespace"

    try:
        path_obj = Path(folder_path)

        if not path_obj.exists():
            return False, f"Folder does not exist: {folder_path}"

        if not path_obj.is_dir():
            return False, f"Path is not a directory: {folder_path}"

        if not os.access(folder_path, os.R_OK):
            return False, f"Folder is not readable: {folder_path}"

        logger.debug(f"Folder path validation successful: {folder_path}")
        return True, None

    except (OSError, PermissionError) as e:
        error_msg = f"Error accessing folder {folder_path}: {e}"
        logger.error(error_msg)
        return False, error_msg


def validate_threshold_range(min_threshold: float, max_threshold: float) -> Tuple[bool, Optional[str]]:
    """
    Validate threshold range for similarity matching.

    Args:
        min_threshold: Minimum threshold value
        max_threshold: Maximum threshold value

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        min_val = float(min_threshold)
        max_val = float(max_threshold)
    except (ValueError, TypeError):
        return False, "Thresholds must be numeric values"

    if min_val < 0 or max_val < 0:
        return False, "Thresholds cannot be negative"

    if min_val > 100 or max_val > 100:
        return False, "Thresholds cannot exceed 100"

    if min_val > max_val:
        return False, "Minimum threshold cannot be greater than maximum threshold"

    logger.debug(f"Threshold validation successful: min={min_val}, max={max_val}")
    return True, None


def validate_limit(limit: int) -> Tuple[bool, Optional[str]]:
    """
    Validate the limit for number of pairs to display.

    Args:
        limit: Number of pairs to display

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        limit_val = int(limit)
    except (ValueError, TypeError):
        return False, "Limit must be a numeric value"

    if limit_val <= 0:
        return False, "Limit must be greater than 0"

    if limit_val > 1000:
        return False, "Limit cannot exceed 1000 (performance reasons)"

    logger.debug(f"Limit validation successful: {limit_val}")
    return True, None
