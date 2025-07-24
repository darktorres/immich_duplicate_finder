#!/usr/bin/env python3
"""
Memory profiler for the duplicate finder application.
This script profiles memory usage during application startup and key operations.
"""

import os
import sys
import time
import tracemalloc
from functools import wraps
from typing import Any, Callable

import psutil
from memory_profiler import profile


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def memory_monitor(func: Callable) -> Callable:
    """Decorator to monitor memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory monitoring
        tracemalloc.start()
        initial_memory = get_memory_usage()
        
        print(f"\n{'='*60}")
        print(f"MEMORY PROFILE: {func.__name__}")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            raise
        finally:
            # Get memory statistics
            end_time = time.time()
            final_memory = get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            # Get top memory allocations
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"\nMemory usage after {func.__name__}:")
            print(f"  Final memory: {final_memory:.2f} MB")
            print(f"  Memory increase: {memory_diff:.2f} MB")
            print(f"  Peak traced memory: {peak / 1024 / 1024:.2f} MB")
            print(f"  Execution time: {end_time - start_time:.2f} seconds")
            print(f"{'='*60}\n")
        
        return result
    return wrapper


@memory_monitor
def profile_imports():
    """Profile memory usage during imports."""
    print("Profiling imports...")
    
    # Import core libraries one by one to see which ones consume the most memory
    import_steps = [
        ("os", "import os"),
        ("numpy", "import numpy as np"),
        ("PIL", "from PIL import Image"),
        ("torch", "import torch"),
        ("torchvision", "from torchvision.models import vit_b_16, ViT_B_16_Weights"),
        ("faiss", "import faiss"),
        ("streamlit", "import streamlit as st"),
    ]
    
    for name, import_stmt in import_steps:
        memory_before = get_memory_usage()
        try:
            exec(import_stmt)
            memory_after = get_memory_usage()
            print(f"  {name}: {memory_after - memory_before:.2f} MB")
        except ImportError as e:
            print(f"  {name}: Import failed - {e}")


@memory_monitor
def profile_model_loading():
    """Profile memory usage during model loading."""
    print("Profiling model loading...")
    
    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    memory_before = get_memory_usage()
    
    # Load model weights
    weights = ViT_B_16_Weights.DEFAULT
    memory_after_weights = get_memory_usage()
    print(f"  Weights loading: {memory_after_weights - memory_before:.2f} MB")
    
    # Load model
    model = vit_b_16(weights=weights)
    memory_after_model = get_memory_usage()
    print(f"  Model creation: {memory_after_model - memory_after_weights:.2f} MB")
    
    # Move to device
    model.to(device)
    memory_after_device = get_memory_usage()
    print(f"  Move to device: {memory_after_device - memory_after_model:.2f} MB")
    
    # Set to eval mode
    model.eval()
    memory_final = get_memory_usage()
    print(f"  Set eval mode: {memory_final - memory_after_device:.2f} MB")
    
    return model


@memory_monitor
def profile_faiss_operations():
    """Profile memory usage during FAISS operations."""
    print("Profiling FAISS operations...")
    
    import faiss
    import numpy as np
    
    # Check if existing index exists
    index_path = "faiss_index.bin"
    metadata_path = "metadata.npy"
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        memory_before = get_memory_usage()
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        memory_after_index = get_memory_usage()
        print(f"  FAISS index loading: {memory_after_index - memory_before:.2f} MB")
        print(f"  Index contains {index.ntotal} vectors")
        
        # Load metadata
        metadata = np.load(metadata_path, allow_pickle=True).tolist()
        memory_after_metadata = get_memory_usage()
        print(f"  Metadata loading: {memory_after_metadata - memory_after_index:.2f} MB")
        print(f"  Metadata contains {len(metadata)} entries")
        
        return index, metadata
    else:
        print("  No existing FAISS index found")
        return None, []


@memory_monitor
def profile_database_operations():
    """Profile memory usage during database operations."""
    print("Profiling database operations...")
    
    try:
        from db import startup_db_configurations, startup_processed_duplicate_faiss_db
        
        memory_before = get_memory_usage()
        startup_db_configurations()
        memory_after_config = get_memory_usage()
        print(f"  DB config startup: {memory_after_config - memory_before:.2f} MB")
        
        startup_processed_duplicate_faiss_db()
        memory_after_faiss_db = get_memory_usage()
        print(f"  FAISS DB startup: {memory_after_faiss_db - memory_after_config:.2f} MB")
        
    except Exception as e:
        print(f"  Database operations failed: {e}")


@memory_monitor
def profile_streamlit_startup():
    """Profile memory usage during Streamlit startup."""
    print("Profiling Streamlit startup...")
    
    try:
        import streamlit as st
        
        # This simulates what happens in app.py
        memory_before = get_memory_usage()
        
        # Set page config (this is usually the first Streamlit call)
        # Note: This might not work in a script context, but we'll try
        try:
            st.set_page_config(page_title="Local duplicator finder", page_icon="🖼️")
            memory_after_config = get_memory_usage()
            print(f"  Streamlit page config: {memory_after_config - memory_before:.2f} MB")
        except Exception as e:
            print(f"  Streamlit page config failed (expected in script): {e}")
            
    except Exception as e:
        print(f"  Streamlit startup failed: {e}")


def main():
    """Main profiling function."""
    print("Starting memory profiling of duplicate finder application...")
    print(f"Initial system memory: {get_memory_usage():.2f} MB")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Profile different components
    profile_imports()
    profile_model_loading()
    profile_faiss_operations()
    profile_database_operations()
    profile_streamlit_startup()
    
    print(f"\nFinal memory usage: {get_memory_usage():.2f} MB")
    print("Memory profiling complete!")


if __name__ == "__main__":
    main()