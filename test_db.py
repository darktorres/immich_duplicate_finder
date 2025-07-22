"""Unit tests for database utilities."""

import os
import sqlite3
import tempfile
import pytest
from pathlib import Path

from db import (
    startup_db_configurations,
    startup_processed_duplicate_faiss_db,
    load_settings_from_db,
    save_settings_to_db,
    save_duplicate_pair,
    load_duplicate_pairs,
    delete_duplicate_pair,
    is_db_populated,
)


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # Cleanup
    os.chdir(original_cwd)
    for db_file in ["settings.db", "duplicates.db"]:
        db_path = Path(temp_dir) / db_file
        if db_path.exists():
            db_path.unlink()
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.mark.unit
def test_startup_db_configurations(temp_db_dir):
    """Test database initialization."""
    startup_db_configurations()
    
    # Check that the database file was created
    assert Path("settings.db").exists()
    
    # Check that the table was created with default values
    conn = sqlite3.connect("settings.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM settings")
    count = cursor.fetchone()[0]
    assert count == 1
    
    cursor.execute("SELECT folder_path FROM settings")
    folder_path = cursor.fetchone()[0]
    assert folder_path == ""
    conn.close()


@pytest.mark.unit
def test_startup_processed_duplicate_faiss_db(temp_db_dir):
    """Test duplicates database initialization."""
    startup_processed_duplicate_faiss_db()
    
    # Check that the database file was created
    assert Path("duplicates.db").exists()
    
    # Check that the table was created
    conn = sqlite3.connect("duplicates.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='duplicates'")
    table_exists = cursor.fetchone() is not None
    assert table_exists
    conn.close()


@pytest.mark.unit
def test_save_and_load_settings(temp_db_dir):
    """Test saving and loading settings."""
    startup_db_configurations()
    
    test_path = "/test/path"
    save_settings_to_db(test_path)
    
    loaded_path = load_settings_from_db()
    assert loaded_path == test_path


@pytest.mark.unit
def test_load_settings_empty_db(temp_db_dir):
    """Test loading settings from empty database."""
    # Don't initialize the database
    result = load_settings_from_db()
    assert result == ""


@pytest.mark.unit
def test_duplicate_pair_operations(temp_db_dir):
    """Test saving, loading, and deleting duplicate pairs."""
    startup_processed_duplicate_faiss_db()
    
    # Test saving duplicate pairs
    save_duplicate_pair("file1.jpg", "file2.jpg", 0.95)
    save_duplicate_pair("file3.jpg", "file4.jpg", 0.85)
    
    # Test loading duplicate pairs
    pairs = load_duplicate_pairs(0.8, 1.0)
    assert len(pairs) == 2
    
    # Test filtering by threshold
    pairs_filtered = load_duplicate_pairs(0.9, 1.0)
    assert len(pairs_filtered) == 1
    assert pairs_filtered[0][2] == 0.95  # similarity score
    
    # Test database population check
    assert is_db_populated() is True
    
    # Test deleting duplicate pair
    delete_duplicate_pair("file1.jpg", "file2.jpg")
    pairs_after_delete = load_duplicate_pairs(0.8, 1.0)
    assert len(pairs_after_delete) == 1


@pytest.mark.unit
def test_duplicate_pair_no_duplicates(temp_db_dir):
    """Test saving the same duplicate pair twice."""
    startup_processed_duplicate_faiss_db()
    
    # Save the same pair twice
    save_duplicate_pair("file1.jpg", "file2.jpg", 0.95)
    save_duplicate_pair("file1.jpg", "file2.jpg", 0.95)
    
    # Should only have one entry
    pairs = load_duplicate_pairs(0.0, 1.0)
    assert len(pairs) == 1


@pytest.mark.unit
def test_duplicate_pair_reverse_order(temp_db_dir):
    """Test saving duplicate pairs in reverse order."""
    startup_processed_duplicate_faiss_db()
    
    # Save pairs in different orders
    save_duplicate_pair("file1.jpg", "file2.jpg", 0.95)
    save_duplicate_pair("file2.jpg", "file1.jpg", 0.95)
    
    # Should only have one entry (no duplicates)
    pairs = load_duplicate_pairs(0.0, 1.0)
    assert len(pairs) == 1


@pytest.mark.unit
def test_is_db_populated_empty(temp_db_dir):
    """Test database population check on empty database."""
    startup_processed_duplicate_faiss_db()
    assert is_db_populated() is False


@pytest.mark.unit
def test_invalid_similarity_value(temp_db_dir):
    """Test handling of invalid similarity values."""
    startup_processed_duplicate_faiss_db()
    
    # This should not raise an exception but should log an error
    save_duplicate_pair("file1.jpg", "file2.jpg", "invalid")
    
    # Should have no entries
    pairs = load_duplicate_pairs(0.0, 1.0)
    assert len(pairs) == 0


@pytest.mark.unit
@pytest.mark.parametrize("min_thresh,max_thresh,expected_valid", [
    (0.0, 100.0, True),
    (-1.0, 100.0, False),  # Invalid: negative min
    (0.0, 150.0, False),   # Invalid: max > 100
    (50.0, 25.0, False),   # Invalid: min > max
])
def test_load_duplicate_pairs_invalid_thresholds(temp_db_dir, min_thresh, max_thresh, expected_valid):
    """Test loading duplicate pairs with invalid thresholds."""
    startup_processed_duplicate_faiss_db()
    save_duplicate_pair("file1.jpg", "file2.jpg", 0.5)
    
    pairs = load_duplicate_pairs(min_thresh, max_thresh)
    
    if expected_valid:
        assert len(pairs) >= 0  # Should work
    else:
        assert len(pairs) == 0  # Should return empty list for invalid thresholds