#!/usr/bin/env python3
"""
Compare memory usage between original and optimized versions.
"""

import os
import sys
import time
import subprocess
import psutil


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_original_imports():
    """Test memory usage of original imports."""
    print("Testing original imports...")
    initial_memory = get_memory_usage()
    
    # Simulate original app.py imports
    import streamlit as st
    memory_after_streamlit = get_memory_usage()
    
    from db import startup_db_configurations, startup_processed_duplicate_faiss_db
    from local_media import setup_local_media
    memory_after_db = get_memory_usage()
    
    # This will trigger the heavy imports
    from imageDuplicate import extract_features
    memory_after_heavy = get_memory_usage()
    
    print(f"  Initial: {initial_memory:.2f} MB")
    print(f"  After Streamlit: {memory_after_streamlit:.2f} MB (+{memory_after_streamlit - initial_memory:.2f})")
    print(f"  After DB setup: {memory_after_db:.2f} MB (+{memory_after_db - memory_after_streamlit:.2f})")
    print(f"  After heavy imports: {memory_after_heavy:.2f} MB (+{memory_after_heavy - memory_after_db:.2f})")
    print(f"  Total increase: {memory_after_heavy - initial_memory:.2f} MB")
    
    return memory_after_heavy - initial_memory


def test_optimized_imports():
    """Test memory usage of optimized imports."""
    print("\nTesting optimized imports...")
    initial_memory = get_memory_usage()
    
    # Simulate optimized app imports (no heavy imports at startup)
    import streamlit as st
    memory_after_streamlit = get_memory_usage()
    
    from db import startup_db_configurations, startup_processed_duplicate_faiss_db
    from local_media import setup_local_media
    memory_after_db = get_memory_usage()
    
    # Heavy imports are lazy-loaded, so they don't happen at startup
    from imageDuplicate_optimized import get_model_and_transform
    memory_after_lazy = get_memory_usage()
    
    print(f"  Initial: {initial_memory:.2f} MB")
    print(f"  After Streamlit: {memory_after_streamlit:.2f} MB (+{memory_after_streamlit - initial_memory:.2f})")
    print(f"  After DB setup: {memory_after_db:.2f} MB (+{memory_after_db - memory_after_streamlit:.2f})")
    print(f"  After lazy imports: {memory_after_lazy:.2f} MB (+{memory_after_lazy - memory_after_db:.2f})")
    print(f"  Total increase: {memory_after_lazy - initial_memory:.2f} MB")
    
    # Now trigger the actual model loading
    print("\n  Triggering model loading...")
    model, transform, device = get_model_and_transform()
    memory_after_model = get_memory_usage()
    print(f"  After model loading: {memory_after_model:.2f} MB (+{memory_after_model - memory_after_lazy:.2f})")
    
    return memory_after_lazy - initial_memory, memory_after_model - memory_after_lazy


def main():
    """Main comparison function."""
    print("Memory Usage Comparison: Original vs Optimized")
    print("=" * 60)
    
    # Test in separate processes to avoid contamination
    print("Running tests in separate processes to avoid memory contamination...")
    
    # Test original version
    print("\n1. Testing Original Version:")
    result = subprocess.run([
        sys.executable, "-c", 
        """
import sys
sys.path.append('.')
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

initial = get_memory_usage()
print(f"Initial: {initial:.2f} MB")

# Heavy imports happen immediately
from imageDuplicate import model, transform, device
final = get_memory_usage()
print(f"After imports: {final:.2f} MB")
print(f"Memory increase: {final - initial:.2f} MB")
        """
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error testing original: {result.stderr}")
    
    # Test optimized version
    print("\n2. Testing Optimized Version (startup only):")
    result = subprocess.run([
        sys.executable, "-c", 
        """
import sys
sys.path.append('.')
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

initial = get_memory_usage()
print(f"Initial: {initial:.2f} MB")

# Only lightweight imports at startup
from imageDuplicate_optimized import get_model_and_transform
startup = get_memory_usage()
print(f"After startup imports: {startup:.2f} MB")
print(f"Startup memory increase: {startup - initial:.2f} MB")

# Now trigger heavy loading
model, transform, device = get_model_and_transform()
final = get_memory_usage()
print(f"After model loading: {final:.2f} MB")
print(f"Model loading increase: {final - startup:.2f} MB")
print(f"Total increase: {final - initial:.2f} MB")
        """
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error testing optimized: {result.stderr}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("- Original version loads everything at startup (high initial memory)")
    print("- Optimized version only loads heavy components when needed (low startup memory)")
    print("- This reduces startup time and allows the app to run on lower-memory systems")
    print("- Memory is only consumed when actually processing images")


if __name__ == "__main__":
    main()