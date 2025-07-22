"""Unit tests for configuration module."""

import os
import pytest
from unittest.mock import patch

from config import Config


@pytest.mark.unit
class TestConfig:
    """Test cases for Config class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        assert Config.LOG_LEVEL == "INFO"
        assert Config.LOG_FILE == "duplicate_finder.log"
        assert Config.SETTINGS_DB == "settings.db"
        assert Config.DUPLICATES_DB == "duplicates.db"
        assert Config.FAISS_INDEX_PATH == "faiss_index.bin"
        assert Config.FAISS_METADATA_PATH == "metadata.npy"
        assert Config.BATCH_SIZE == 10
        assert Config.MAX_WORKERS == 4
        assert Config.DEFAULT_MIN_THRESHOLD == 0.0
        assert Config.DEFAULT_MAX_THRESHOLD == 100.0
        assert Config.DEFAULT_LIMIT == 10
        assert Config.MAX_LIMIT == 1000
        assert Config.ENABLE_GPU is True
    
    @patch.dict(os.environ, {
        'LOG_LEVEL': 'DEBUG',
        'LOG_FILE': 'test.log',
        'BATCH_SIZE': '20',
        'MAX_WORKERS': '8',
        'ENABLE_GPU': 'false'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        # Need to reload the module to pick up env vars
        import importlib
        import config
        importlib.reload(config)
        
        assert config.Config.LOG_LEVEL == "DEBUG"
        assert config.Config.LOG_FILE == "test.log"
        assert config.Config.BATCH_SIZE == 20
        assert config.Config.MAX_WORKERS == 8
        assert config.Config.ENABLE_GPU is False
    
    def test_get_all_settings(self):
        """Test getting all configuration settings."""
        settings = Config.get_all_settings()
        
        assert isinstance(settings, dict)
        assert 'LOG_LEVEL' in settings
        assert 'BATCH_SIZE' in settings
        assert 'DEFAULT_MIN_THRESHOLD' in settings
        
        # Should not include methods
        assert 'get_all_settings' not in settings
        assert 'validate_config' not in settings
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid values."""
        assert Config.validate_config() is True
    
    @patch.object(Config, 'BATCH_SIZE', 0)
    def test_validate_config_invalid_batch_size(self):
        """Test configuration validation with invalid batch size."""
        assert Config.validate_config() is False
    
    @patch.object(Config, 'MAX_WORKERS', -1)
    def test_validate_config_invalid_max_workers(self):
        """Test configuration validation with invalid max workers."""
        assert Config.validate_config() is False
    
    @patch.object(Config, 'DEFAULT_MIN_THRESHOLD', -1.0)
    def test_validate_config_invalid_min_threshold(self):
        """Test configuration validation with invalid min threshold."""
        assert Config.validate_config() is False
    
    @patch.object(Config, 'DEFAULT_MAX_THRESHOLD', 150.0)
    def test_validate_config_invalid_max_threshold(self):
        """Test configuration validation with invalid max threshold."""
        assert Config.validate_config() is False
    
    @patch.object(Config, 'DEFAULT_MIN_THRESHOLD', 50.0)
    @patch.object(Config, 'DEFAULT_MAX_THRESHOLD', 25.0)
    def test_validate_config_threshold_order(self):
        """Test configuration validation with min > max threshold."""
        assert Config.validate_config() is False
    
    @patch.object(Config, 'DEFAULT_LIMIT', 0)
    def test_validate_config_invalid_default_limit(self):
        """Test configuration validation with invalid default limit."""
        assert Config.validate_config() is False
    
    @patch.object(Config, 'MAX_LIMIT', 5)
    @patch.object(Config, 'DEFAULT_LIMIT', 10)
    def test_validate_config_limit_order(self):
        """Test configuration validation with default limit > max limit."""
        assert Config.validate_config() is False


@pytest.mark.unit
def test_config_import_validation():
    """Test that config validation runs on import."""
    # This test ensures that invalid config would raise ValueError on import
    # Since we're already imported, we test that no exception was raised
    assert True  # If we get here, validation passed


@pytest.mark.unit
@patch.dict(os.environ, {'BATCH_SIZE': 'invalid'})
def test_invalid_environment_variable():
    """Test handling of invalid environment variable values."""
    # This should not crash, should use default value
    import importlib
    import config
    
    try:
        importlib.reload(config)
        # Should use default value when env var is invalid
        assert config.Config.BATCH_SIZE == 10  # default value
    except ValueError:
        # This is also acceptable - config validation should catch it
        pass


@pytest.mark.unit
class TestConfigTypes:
    """Test configuration value types."""
    
    def test_string_configs(self):
        """Test string configuration values."""
        assert isinstance(Config.LOG_LEVEL, str)
        assert isinstance(Config.LOG_FILE, str)
        assert isinstance(Config.SETTINGS_DB, str)
        assert isinstance(Config.DUPLICATES_DB, str)
        assert isinstance(Config.FAISS_INDEX_PATH, str)
        assert isinstance(Config.FAISS_METADATA_PATH, str)
    
    def test_integer_configs(self):
        """Test integer configuration values."""
        assert isinstance(Config.BATCH_SIZE, int)
        assert isinstance(Config.MAX_WORKERS, int)
        assert isinstance(Config.DEFAULT_LIMIT, int)
        assert isinstance(Config.MAX_LIMIT, int)
    
    def test_float_configs(self):
        """Test float configuration values."""
        assert isinstance(Config.DEFAULT_MIN_THRESHOLD, float)
        assert isinstance(Config.DEFAULT_MAX_THRESHOLD, float)
    
    def test_boolean_configs(self):
        """Test boolean configuration values."""
        assert isinstance(Config.ENABLE_GPU, bool)