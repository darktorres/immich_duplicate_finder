"""Configuration settings for the duplicate finder application."""

import os
from typing import Dict, Any


class Config:
    """Application configuration settings."""
    
    # Logging configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "duplicate_finder.log")
    
    # Database configuration
    SETTINGS_DB: str = "settings.db"
    DUPLICATES_DB: str = "duplicates.db"
    
    # FAISS configuration
    FAISS_INDEX_PATH: str = "faiss_index.bin"
    FAISS_METADATA_PATH: str = "metadata.npy"
    
    # Processing configuration
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # UI configuration
    DEFAULT_MIN_THRESHOLD: float = 0.0
    DEFAULT_MAX_THRESHOLD: float = 100.0
    DEFAULT_LIMIT: int = 10
    MAX_LIMIT: int = 1000
    
    # Performance settings
    ENABLE_GPU: bool = os.getenv("ENABLE_GPU", "true").lower() == "true"
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """
        Get all configuration settings as a dictionary.
        
        Returns:
            Dictionary of all configuration settings
        """
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate numeric settings
            assert cls.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
            assert cls.MAX_WORKERS > 0, "MAX_WORKERS must be positive"
            assert 0 <= cls.DEFAULT_MIN_THRESHOLD <= 100, "DEFAULT_MIN_THRESHOLD must be 0-100"
            assert 0 <= cls.DEFAULT_MAX_THRESHOLD <= 100, "DEFAULT_MAX_THRESHOLD must be 0-100"
            assert cls.DEFAULT_MIN_THRESHOLD <= cls.DEFAULT_MAX_THRESHOLD, "MIN_THRESHOLD must be <= MAX_THRESHOLD"
            assert cls.DEFAULT_LIMIT > 0, "DEFAULT_LIMIT must be positive"
            assert cls.MAX_LIMIT >= cls.DEFAULT_LIMIT, "MAX_LIMIT must be >= DEFAULT_LIMIT"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False


# Validate configuration on import
if not Config.validate_config():
    raise ValueError("Invalid configuration settings")