import sqlite3

#############DATABASE###################


def startup_db_configurations():
    """Initializes the settings database and table."""
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
        # Insert default settings, setting timeout to 1500 ms
        c.execute("INSERT INTO settings (folder_path) VALUES (?)", ("",))

    # Commit changes and close the connection
    conn.commit()
    conn.close()


def load_settings_from_db():
    """Loads the media folder path from the settings database."""
    conn = sqlite3.connect("settings.db")
    c = conn.cursor()
    c.execute("SELECT folder_path FROM settings LIMIT 1")
    settings = c.fetchone()
    conn.close()
    return settings[0] if settings else ""


def save_settings_to_db(folder_path):
    """Saves the media folder path to the settings database."""
    conn = sqlite3.connect("settings.db")
    c = conn.cursor()
    # Assumes a single row of settings; clears existing and inserts the new path.
    c.execute("DELETE FROM settings")
    c.execute("INSERT INTO settings (folder_path) VALUES (?)", (folder_path,))
    conn.commit()
    conn.close()


####################### FAISS #############################
def startup_processed_duplicate_faiss_db():
    """Initializes the duplicates database with a corrected schema."""
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
    except Exception as e:
        print("Error creating database/table:", e)
    finally:
        conn.close()


def save_duplicate_pair(vector_id1, vector_id2, similarity):
    """Saves a pair of duplicate file paths to the database if it doesn't already exist."""
    similarity = float(similarity)
    try:
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()

        # Check if the pair already exists in either order
        cursor.execute(
            "SELECT * FROM duplicates WHERE (vector_id1 = ? AND vector_id2 = ?) OR (vector_id1 = ? AND vector_id2 = ?)",
            (vector_id1, vector_id2, vector_id2, vector_id1),
        )
        if cursor.fetchone():
            # print("Duplicate pair already exists.")
            return

        # If not, insert the new pair
        cursor.execute("INSERT INTO duplicates (vector_id1, vector_id2, similarity) VALUES (?, ?, ?)", (vector_id1, vector_id2, similarity))
        conn.commit()
    except Exception as e:
        print("Error inserting duplicate pair:", e)
    finally:
        conn.close()


def delete_duplicate_pair(asset_id_1, asset_id_2):
    """Deletes a specific duplicate pair from the database."""
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
        conn.close()


def load_duplicate_pairs(min_threshold, max_threshold):
    """Load duplicate pairs with a similarity between the specified minimum and maximum thresholds."""
    try:
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()
        # Adjust the SQL query to filter duplicates within the specified range
        cursor.execute(
            """
            SELECT vector_id1, vector_id2 FROM duplicates
            WHERE similarity >= ? AND similarity <= ?""",
            (min_threshold, max_threshold),
        )
        duplicates = cursor.fetchall()
        if not duplicates:
            print(f"No duplicates found within thresholds {min_threshold} and {max_threshold}")
        return duplicates
    except Exception as e:
        print("Error loading duplicates:", e)
    finally:
        if conn:
            conn.close()


def is_db_populated():
    """Check if the 'duplicates' table in the database has any entries."""
    conn = None
    try:
        conn = sqlite3.connect("duplicates.db")
        cursor = conn.cursor()
        # Check if there are any rows in the table
        cursor.execute("SELECT EXISTS(SELECT 1 FROM duplicates LIMIT 1)")
        exists = cursor.fetchone()[0]
        return exists == 1
    except Exception as e:
        print("Error checking database population:", e)
        return False
    finally:
        if conn:
            conn.close()
