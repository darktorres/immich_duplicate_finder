#!/usr/bin/env python3
"""
Optimal Batch Size Finder

This script helps you find the optimal batch size for your system
by running performance tests and analyzing the results.

Usage:
    python find_optimal_batch_size.py
    python find_optimal_batch_size.py --quick
    python find_optimal_batch_size.py --comprehensive
"""

import argparse
import sys
import time
from test_quick_batch_optimization import QuickBatchOptimizer, get_system_recommendations
from memory_config import MEMORY_CONFIG, MemoryConfig, update_config


def print_header():
    """Print a nice header."""
    print("🚀 Optimal Batch Size Finder")
    print("=" * 50)
    print("Finding the best batch size for your system...")
    print()


def print_system_info():
    """Print system information."""
    import psutil
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"💻 System Information:")
    print(f"   Memory: {memory_gb:.1f} GB")
    print(f"   CPU Cores: {cpu_count}")
    print(f"   Current Batch Size: {MEMORY_CONFIG.batch_size}")
    print()


def run_quick_optimization():
    """Run quick optimization tests."""
    print("🧪 Running Quick Optimization Tests...")
    print("-" * 30)
    
    optimizer = QuickBatchOptimizer()
    
    # Get system recommendations
    recommendations = get_system_recommendations()
    batch_sizes = recommendations['recommended_range']
    
    print(f"System Category: {recommendations['category']}")
    print(f"Testing batch sizes: {batch_sizes}")
    print()
    
    # Run tests
    results = optimizer.test_batch_sizes(batch_sizes, num_images=25)
    
    # Find optimal
    optimal_result = optimizer.find_optimal_batch_size(results)
    
    return optimal_result, results


def run_comprehensive_optimization():
    """Run comprehensive optimization tests."""
    print("🔬 Running Comprehensive Optimization Tests...")
    print("-" * 40)
    
    from test_batch_size_optimization import BatchSizeOptimizer
    
    optimizer = BatchSizeOptimizer()
    
    try:
        # Get system recommendations
        recommendations = get_system_recommendations()
        batch_sizes = recommendations['recommended_range']
        
        # Add some additional sizes for comprehensive testing
        if max(batch_sizes) < 50:
            batch_sizes.extend([x for x in [25, 30, 40] if x not in batch_sizes])
        
        print(f"System Category: {recommendations['category']}")
        print(f"Testing batch sizes: {batch_sizes}")
        print()
        
        # Run comprehensive tests
        results = optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=40,
            iterations=3
        )
        
        # Find optimal
        optimal_result = optimizer.find_optimal_batch_size()
        
        return optimal_result, results
        
    finally:
        optimizer.cleanup()


def print_results(optimal_result, results, test_type="Quick"):
    """Print optimization results."""
    print(f"\n📊 {test_type} Test Results:")
    print("-" * 30)
    
    for result in results:
        if 'batch_size' in result:
            batch_size = result['batch_size']
            if 'avg_speed' in result:
                speed = result['avg_speed']
                memory = result.get('avg_memory_increase', 0)
                print(f"Batch {batch_size:3d}: {speed:6.2f} img/s, {memory:5.1f} MB")
            elif 'avg_images_per_second' in result:
                speed = result['avg_images_per_second']
                memory = result.get('avg_memory_increase_mb', 0)
                print(f"Batch {batch_size:3d}: {speed:6.2f} img/s, {memory:5.1f} MB")
    
    print(f"\n🏆 Optimal Batch Size: {optimal_result['optimal_batch_size']}")
    
    best = optimal_result['best_result']
    if 'avg_speed' in best:
        speed = best['avg_speed']
        memory = best.get('avg_memory_increase', 0)
    elif 'avg_images_per_second' in best:
        speed = best['avg_images_per_second']
        memory = best.get('avg_memory_increase_mb', 0)
    else:
        speed = 0
        memory = 0
    
    print(f"   Performance: {speed:.2f} images/second")
    print(f"   Memory: {memory:.1f} MB increase")
    print(f"   Score: {best.get('composite_score', 0):.3f}")


def apply_optimal_config(optimal_batch_size):
    """Apply the optimal configuration."""
    print(f"\n🔧 Configuration Update:")
    print("-" * 25)
    
    current_batch_size = MEMORY_CONFIG.batch_size
    
    if optimal_batch_size == current_batch_size:
        print(f"✅ Current batch size ({current_batch_size}) is already optimal!")
        return False
    
    print(f"Current batch size: {current_batch_size}")
    print(f"Optimal batch size: {optimal_batch_size}")
    
    # Ask user if they want to apply the change
    response = input("\nWould you like to update your configuration? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        # Create new config with optimal batch size
        new_config = MemoryConfig(
            use_cpu_only=MEMORY_CONFIG.use_cpu_only,
            model_precision=MEMORY_CONFIG.model_precision,
            batch_size=optimal_batch_size,
            faiss_index_type=MEMORY_CONFIG.faiss_index_type,
            faiss_nlist=MEMORY_CONFIG.faiss_nlist,
            max_image_size=MEMORY_CONFIG.max_image_size,
            lazy_loading=MEMORY_CONFIG.lazy_loading,
            aggressive_gc=MEMORY_CONFIG.aggressive_gc,
            clear_cache_after_batch=MEMORY_CONFIG.clear_cache_after_batch,
            max_cache_size_mb=MEMORY_CONFIG.max_cache_size_mb
        )
        
        # Update global config
        update_config(new_config)
        
        print(f"✅ Configuration updated! New batch size: {optimal_batch_size}")
        print("   Restart the application to use the new settings.")
        return True
    else:
        print("Configuration not changed.")
        return False


def print_recommendations(optimal_batch_size):
    """Print recommendations based on results."""
    print(f"\n💡 Recommendations:")
    print("-" * 20)
    
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    if optimal_batch_size <= 5:
        print("• Your system benefits from small batch sizes")
        print("• Consider enabling CPU-only mode if using GPU")
        print("• Enable aggressive garbage collection")
        if memory_gb < 8:
            print("• Consider upgrading system memory for better performance")
    
    elif optimal_batch_size <= 20:
        print("• Your system works well with medium batch sizes")
        print("• Current memory configuration seems appropriate")
        print("• Monitor memory usage during processing")
    
    else:
        print("• Your system can handle large batch sizes efficiently")
        print("• Consider disabling aggressive garbage collection")
        print("• You may benefit from larger image processing limits")
    
    print(f"\n🎯 For your system ({memory_gb:.1f} GB RAM):")
    print(f"   Recommended batch size: {optimal_batch_size}")
    
    if optimal_batch_size < 10:
        print("   Focus on memory efficiency")
    elif optimal_batch_size < 30:
        print("   Good balance of speed and memory usage")
    else:
        print("   Optimized for maximum processing speed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Find optimal batch size for image processing")
    parser.add_argument("--quick", action="store_true", help="Run quick optimization (default)")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive optimization")
    parser.add_argument("--no-apply", action="store_true", help="Don't ask to apply configuration")
    
    args = parser.parse_args()
    
    # Default to quick if no option specified
    if not args.comprehensive:
        args.quick = True
    
    print_header()
    print_system_info()
    
    try:
        if args.quick:
            optimal_result, results = run_quick_optimization()
            test_type = "Quick"
        else:
            optimal_result, results = run_comprehensive_optimization()
            test_type = "Comprehensive"
        
        if not optimal_result:
            print("❌ No optimization results available.")
            return 1
        
        optimal_batch_size = optimal_result['optimal_batch_size']
        
        print_results(optimal_result, results, test_type)
        print_recommendations(optimal_batch_size)
        
        if not args.no_apply:
            config_updated = apply_optimal_config(optimal_batch_size)
            if config_updated:
                print("\n🔄 Please restart your application to use the new batch size.")
        
        print(f"\n✅ Optimization complete!")
        print(f"   Recommended batch size: {optimal_batch_size}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Optimization cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())