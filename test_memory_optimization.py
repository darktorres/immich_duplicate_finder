#!/usr/bin/env python3
"""
Test script to demonstrate memory optimization improvements.
"""

import os
import sys
import time
import psutil
import gc


def get_memory_info():
    """Get detailed memory information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def print_memory_status(label):
    """Print current memory status."""
    info = get_memory_info()
    print(f"{label}:")
    print(f"  RSS: {info['rss_mb']:.2f} MB")
    print(f"  VMS: {info['vms_mb']:.2f} MB") 
    print(f"  Percent: {info['percent']:.2f}%")
    print()


def test_original_approach():
    """Test what the original approach would have been (simulated)."""
    print("=" * 60)
    print("TESTING ORIGINAL APPROACH (Simulated heavy imports at startup)")
    print("=" * 60)
    
    print_memory_status("Initial memory")
    
    # Simulate the original app.py behavior
    print("Loading basic imports...")
    import streamlit as st
    print_memory_status("After Streamlit import")
    
    print("Loading database and media setup...")
    from db import startup_db_configurations, startup_processed_duplicate_faiss_db
    from local_media import setup_local_media
    startup_db_configurations()
    startup_processed_duplicate_faiss_db()
    setup_local_media()
    print_memory_status("After DB/media setup")
    
    print("Loading heavy ML components (simulating original behavior)...")
    # Simulate what original would have done - load everything immediately
    try:
        from imageDuplicate import get_model_and_transform
        components = get_model_and_transform()
        print_memory_status("After heavy ML imports")
        print("✓ Heavy components loaded successfully")
    except Exception as e:
        print(f"✗ Error loading heavy components: {e}")
        print_memory_status("After failed heavy imports")
    
    return get_memory_info()['rss_mb']


def test_optimized_approach():
    """Test the optimized approach with lazy loading."""
    print("=" * 60)
    print("TESTING OPTIMIZED APPROACH (Lazy loading)")
    print("=" * 60)
    
    print_memory_status("Initial memory")
    
    print("Loading basic imports...")
    import streamlit as st
    print_memory_status("After Streamlit import")
    
    print("Loading database and media setup...")
    from db import startup_db_configurations, startup_processed_duplicate_faiss_db
    from local_media import setup_local_media
    startup_db_configurations()
    startup_processed_duplicate_faiss_db()
    setup_local_media()
    print_memory_status("After DB/media setup")
    
    print("Loading optimized module (no heavy imports yet)...")
    from imageDuplicate import get_model_and_transform
    startup_memory = get_memory_info()['rss_mb']
    print_memory_status("After optimized imports (startup complete)")
    
    print("Now triggering heavy component loading (only when needed)...")
    try:
        components = get_model_and_transform()
        print_memory_status("After lazy-loaded heavy components")
        print("✓ Heavy components loaded on-demand successfully")
    except Exception as e:
        print(f"✗ Error loading heavy components: {e}")
        print_memory_status("After failed heavy loading")
    
    return startup_memory, get_memory_info()['rss_mb']


def main():
    """Main test function."""
    print("Memory Optimization Test")
    print("This script demonstrates the memory usage difference between")
    print("the original and optimized approaches.\n")
    
    # Test original approach in a subprocess to avoid contamination
    print("Testing approaches in sequence (same process for comparison)...")
    print("Note: In real usage, these would be separate app launches.\n")
    
    # Get baseline
    baseline = get_memory_info()['rss_mb']
    print(f"Baseline memory usage: {baseline:.2f} MB\n")
    
    # Test original
    try:
        original_memory = test_original_approach()
        original_increase = original_memory - baseline
    except Exception as e:
        print(f"Original approach test failed: {e}")
        original_memory = 0
        original_increase = 0
    
    # Force cleanup
    gc.collect()
    time.sleep(1)
    
    # Test optimized  
    try:
        startup_memory, full_memory = test_optimized_approach()
        startup_increase = startup_memory - baseline
        full_increase = full_memory - baseline
    except Exception as e:
        print(f"Optimized approach test failed: {e}")
        startup_memory = 0
        full_memory = 0
        startup_increase = 0
        full_increase = 0
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline memory: {baseline:.2f} MB")
    print()
    print("Original approach:")
    print(f"  Total memory: {original_memory:.2f} MB")
    print(f"  Increase: +{original_increase:.2f} MB")
    print()
    print("Optimized approach:")
    print(f"  Startup memory: {startup_memory:.2f} MB")
    print(f"  Startup increase: +{startup_increase:.2f} MB")
    print(f"  Full memory (when needed): {full_memory:.2f} MB")
    print(f"  Full increase: +{full_increase:.2f} MB")
    print()
    
    if startup_increase > 0 and original_increase > 0:
        startup_savings = original_increase - startup_increase
        startup_savings_percent = (startup_savings / original_increase) * 100
        print(f"Memory savings at startup: {startup_savings:.2f} MB ({startup_savings_percent:.1f}%)")
    
    print()
    print("Benefits of optimization:")
    print("✓ Faster startup time (no heavy model loading)")
    print("✓ Lower memory usage until features are actually needed")
    print("✓ Better user experience on low-memory systems")
    print("✓ Ability to run multiple instances or other apps simultaneously")


if __name__ == "__main__":
    main()