import sqlite3
from typing import List, Tuple

from logger_config import logger

#############DATABASE###################


def startup_db_configurations() -> None:
    """Initializes the settings database and table."""
    conn = None
    try:
        conn = sqlite3.connect("settings.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                folder_path TEXT
            )
        """)

        # Check if the table is empty
        c.execute("SELECT COUNT(*) FROM settings")
        if c.fetchone()[0] == 0:
            # Insert default settings
            c.execute("INSERT INTO settings (folder_path) VALUES (?)", ("",))
            logger.info("Initialized settings database with default values")

        # Commit changes
        conn.commit()
        logger.debug("Settings database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Error initializing settings database: {e}")
        raise
    finally:
        if conn:
            conn.close()


def load_settings_from_db() -> str:
    """
    Loads the media folder path from the settings database.
    
    Returns:
        The folder path string, empty string if not found
    """
    conn = None
    try:
        conn = sqlite3.connect("settings.db")
        c = conn.cursor()
        c.execute("SELECT folder_path FROM settings LIMIT 1")
        settings = c.fetchone()
        result = settings[0] if settings else ""
        logger.debug(f"Loaded folder path from database: {result}")
        return result
    except sqlite3.Error as e:
        logger.error(f"Error loading settings from database: {e}")
        return ""
    finally:
        if conn:
            conn.close()


def save_settings_to_db(folder_path: str) -> None:
    """
    Saves the media folder path to the settings database.
    
    Args:
        folder_path: The folder path to save
    """
    conn = None
    try:
        conn = sqlite3.connect("settings.db")
        c = conn.cursor()
        # Assumes a single row of settings; clears existing and inserts the new path.
        c.execute("DELETE FROM settings")
        c.execute("INSERT INTO settings (folder_path) VALUES (?)", (folder_path,))
        conn.commit()
        logger.info(f"Saved folder path to database: {folder_path}")
    except sqlite3.Error as e:
        logger.error(f"Error saving settings to database: {e}")
        raise
    finally:
        if conn:
            conn.close()


####################### FAISS #############################
def startup_processed_duplicate_faiss_db() -> None:
    """Initializes the duplicates database with a corrected schema."""
    conn = None
    try:
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()
        # Schema corrected to use TEXT for file paths (vector_id1, vector_id2)
        sql = """CREATE TABLE IF NOT EXISTS duplicates(
           id INTEGER PRIMARY KEY,
           vector_id1 TEXT,
           vector_id2 TEXT,
           similarity FLOAT
        )"""
        cursor.execute(sql)
        conn.commit()
        logger.debug("Duplicates database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Error creating duplicates database/table: {e}")
        raise
    finally:
        if conn:
            conn.close()


def save_duplicate_pair(vector_id1: str, vector_id2: str, similarity: float) -> None:
    """
    Saves a pair of duplicate file paths to the database if it doesn't already exist.
    
    Args:
        vector_id1: First file path
        vector_id2: Second file path
        similarity: Similarity score between the files
    """
    try:
        similarity = float(similarity)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid similarity value: {similarity}, error: {e}")
        return
        
    conn = None
    try:
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()

        # Check if the pair already exists in either order
        cursor.execute(
            "SELECT * FROM duplicates WHERE (vector_id1 = ? AND vector_id2 = ?) OR (vector_id1 = ? AND vector_id2 = ?)",
            (vector_id1, vector_id2, vector_id2, vector_id1),
        )
        if cursor.fetchone():
            logger.debug(f"Duplicate pair already exists: {vector_id1} <-> {vector_id2}")
            return

        # If not, insert the new pair
        cursor.execute("INSERT INTO duplicates (vector_id1, vector_id2, similarity) VALUES (?, ?, ?)", (vector_id1, vector_id2, similarity))
        conn.commit()
        logger.debug(f"Saved duplicate pair: {vector_id1} <-> {vector_id2} (similarity: {similarity:.4f})")
    except sqlite3.Error as e:
        logger.error(f"Error inserting duplicate pair: {e}")
    finally:
        if conn:
            conn.close()


def delete_duplicate_pair(asset_id_1, asset_id_2):
    """Deletes a specific duplicate pair from the database."""
    conn = None
    try:
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()
        # Delete the specific duplicate entry involving the two asset IDs
        cursor.execute(
            "DELETE FROM duplicates WHERE (vector_id1 = ? AND vector_id2 = ?) OR (vector_id1 = ? AND vector_id2 = ?)",
            (asset_id_1, asset_id_2, asset_id_2, asset_id_1),
        )
        conn.commit()
        print("Deleted asset from db")
    except Exception as e:
        print(f"Error deleting duplicate entries for asset pair {asset_id_1}-{asset_id_2}:", e)
    finally:
        if conn:
            conn.close()


def load_duplicate_pairs(min_threshold: float, max_threshold: float) -> List[Tuple[str, str, float]]:
    """
    Load duplicate pairs with a similarity between the specified minimum and maximum thresholds.
    
    Args:
        min_threshold: Minimum similarity threshold
        max_threshold: Maximum similarity threshold
        
    Returns:
        List of tuples containing (vector_id1, vector_id2, similarity)
    """
    conn = None
    try:
        # Validate thresholds
        if not (0 <= min_threshold <= max_threshold <= 100):
            logger.error(f"Invalid thresholds: min={min_threshold}, max={max_threshold}")
            return []
            
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()
        # Adjust the SQL query to filter duplicates within the specified range
        cursor.execute(
            """
            SELECT vector_id1, vector_id2, similarity FROM duplicates
            WHERE similarity >= ? AND similarity <= ?
            ORDER BY similarity ASC""",
            (min_threshold, max_threshold),
        )
        duplicates = cursor.fetchall()
        
        if not duplicates:
            logger.info(f"No duplicates found within thresholds {min_threshold} and {max_threshold}")
        else:
            logger.info(f"Found {len(duplicates)} duplicate pairs within thresholds")
            
        return duplicates
    except sqlite3.Error as e:
        logger.error(f"Error loading duplicates: {e}")
        return []
    finally:
        if conn:
            conn.close()


def is_db_populated() -> bool:
    """
    Check if the 'duplicates' table in the database has any entries.
    
    Returns:
        True if database has entries, False otherwise
    """
    conn = None
    try:
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()
        # Check if there are any rows in the table
        cursor.execute("SELECT EXISTS(SELECT 1 FROM duplicates LIMIT 1)")
        exists = cursor.fetchone()[0]
        result = exists == 1
        logger.debug(f"Database populated check: {result}")
        return result
    except sqlite3.Error as e:
        logger.error(f"Error checking database population: {e}")
        return False
    finally:
        if conn:
            conn.close()
