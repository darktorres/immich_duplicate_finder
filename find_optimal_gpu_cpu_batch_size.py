#!/usr/bin/env python3
"""
GPU/CPU Optimal Batch Size Finder

This script tests actual PyTorch model inference on both CPU and GPU
to find the optimal device and batch size configuration.

Usage:
    python find_optimal_gpu_cpu_batch_size.py
    python find_optimal_gpu_cpu_batch_size.py --cpu-only
    python find_optimal_gpu_cpu_batch_size.py --gpu-only
"""

import argparse
import sys
import time
from test_gpu_cpu_batch_optimization import GPUCPUBatchOptimizer
from memory_config import MEMORY_CONFIG, MemoryConfig, update_config


def print_header():
    """Print a nice header."""
    print("🚀 GPU/CPU Optimal Batch Size Finder")
    print("=" * 55)
    print("Testing real PyTorch model inference performance...")
    print()


def print_system_info():
    """Print system information including GPU details."""
    import psutil
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"💻 System Information:")
    print(f"   Memory: {memory_gb:.1f} GB")
    print(f"   CPU Cores: {cpu_count}")
    print(f"   Current Batch Size: {MEMORY_CONFIG.batch_size}")
    print(f"   Current Device: {'CPU Only' if MEMORY_CONFIG.use_cpu_only else 'GPU Enabled'}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("   GPU: Not available")
    except ImportError:
        print("   GPU: PyTorch not installed")
    
    print()


def get_recommended_batch_sizes(memory_gb: float, gpu_available: bool) -> dict:
    """Get recommended batch sizes based on system specs."""
    
    if gpu_available:
        if memory_gb < 8:
            return {
                'cpu_batches': [1, 2, 4, 6, 8],
                'gpu_batches': [1, 2, 4, 6, 8, 12, 16],
                'image_count': 15,
                'category': 'Low Memory + GPU'
            }
        elif memory_gb < 16:
            return {
                'cpu_batches': [2, 4, 6, 8, 12, 16],
                'gpu_batches': [2, 4, 6, 8, 12, 16, 24, 32],
                'image_count': 20,
                'category': 'Medium Memory + GPU'
            }
        else:
            return {
                'cpu_batches': [4, 6, 8, 12, 16, 20, 24],
                'gpu_batches': [4, 6, 8, 12, 16, 20, 24, 32, 48, 64],
                'image_count': 25,
                'category': 'High Memory + GPU'
            }
    else:
        if memory_gb < 8:
            return {
                'cpu_batches': [1, 2, 4, 6, 8, 10, 12],
                'gpu_batches': [],
                'image_count': 18,
                'category': 'Low Memory CPU Only'
            }
        elif memory_gb < 16:
            return {
                'cpu_batches': [2, 4, 6, 8, 12, 16, 20, 24],
                'gpu_batches': [],
                'image_count': 22,
                'category': 'Medium Memory CPU Only'
            }
        else:
            return {
                'cpu_batches': [4, 6, 8, 12, 16, 20, 24, 32, 40, 48],
                'gpu_batches': [],
                'image_count': 30,
                'category': 'High Memory CPU Only'
            }


def run_gpu_cpu_optimization(cpu_only=False, gpu_only=False):
    """Run GPU/CPU optimization tests."""
    print("🔬 Running Real Model Inference Tests...")
    print("-" * 40)
    
    optimizer = GPUCPUBatchOptimizer()
    
    try:
        # Get system info
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available() and not cpu_only
        except ImportError:
            pass
        
        # Get recommendations
        recommendations = get_recommended_batch_sizes(memory_gb, gpu_available)
        
        print(f"System Category: {recommendations['category']}")
        
        if gpu_only and not gpu_available:
            print("❌ GPU-only mode requested but GPU not available")
            return None, None
        
        # Determine batch sizes to test
        if cpu_only:
            batch_sizes = recommendations['cpu_batches']
            print(f"Testing CPU batch sizes: {batch_sizes}")
            
            # Test only CPU
            cpu_results = optimizer.test_device_batch_sizes('cpu', batch_sizes, recommendations['image_count'])
            comparison = {
                'cpu_results': cpu_results,
                'gpu_results': [],
                'comparison': []
            }
            
        elif gpu_only:
            batch_sizes = recommendations['gpu_batches']
            print(f"Testing GPU batch sizes: {batch_sizes}")
            
            # Test only GPU
            gpu_results = optimizer.test_device_batch_sizes('cuda', batch_sizes, recommendations['image_count'])
            comparison = {
                'cpu_results': [],
                'gpu_results': gpu_results,
                'comparison': []
            }
            
        else:
            # Test both CPU and GPU
            cpu_batches = recommendations['cpu_batches']
            gpu_batches = recommendations['gpu_batches'] if gpu_available else []
            
            # Use common batch sizes for comparison (preserve order)
            if gpu_batches:
                common_batches = [b for b in cpu_batches if b in gpu_batches]
                if not common_batches:
                    # If no common batches, use the smaller range
                    common_batches = cpu_batches[:3] if len(cpu_batches) <= len(gpu_batches) else gpu_batches[:3]
            else:
                common_batches = cpu_batches
            
            print(f"Testing batch sizes: {common_batches}")
            
            comparison = optimizer.compare_cpu_gpu_performance(common_batches, recommendations['image_count'])
        
        # Find optimal configuration
        optimal_result = optimizer.find_optimal_device_batch_size(comparison)
        
        return optimal_result, comparison
        
    finally:
        optimizer.cleanup()


def print_detailed_results(comparison, test_type="Full"):
    """Print detailed optimization results."""
    print(f"\n📊 {test_type} Test Results:")
    print("-" * 35)
    
    # Print CPU results
    if comparison['cpu_results']:
        print("\n🖥️  CPU Performance:")
        print("   Batch | Speed    | CPU Mem | Inference")
        print("   ------|----------|---------|----------")
        for result in comparison['cpu_results']:
            batch = result['batch_size']
            speed = result['avg_images_per_second']
            memory = result['avg_cpu_memory']
            inf_speed = result['avg_inference_ips']
            print(f"   {batch:4d}  | {speed:6.2f}/s | {memory:5.1f}MB | {inf_speed:6.2f}/s")
    
    # Print GPU results
    if comparison['gpu_results']:
        print("\n🚀 GPU Performance:")
        print("   Batch | Speed    | CPU Mem | GPU Mem | Inference")
        print("   ------|----------|---------|---------|----------")
        for result in comparison['gpu_results']:
            batch = result['batch_size']
            speed = result['avg_images_per_second']
            cpu_mem = result['avg_cpu_memory']
            gpu_mem = result['avg_gpu_memory']
            inf_speed = result['avg_inference_ips']
            print(f"   {batch:4d}  | {speed:6.2f}/s | {cpu_mem:5.1f}MB | {gpu_mem:5.1f}MB | {inf_speed:6.2f}/s")
    
    # Print comparison
    if comparison['comparison']:
        print("\n⚡ CPU vs GPU Comparison:")
        print("   Batch | CPU Speed | GPU Speed | Speedup | Total Mem")
        print("   ------|-----------|-----------|---------|----------")
        for comp in comparison['comparison']:
            batch = comp['batch_size']
            cpu_speed = comp['cpu_speed']
            gpu_speed = comp['gpu_speed']
            speedup = comp['speedup']
            total_mem = comp['total_memory']
            print(f"   {batch:4d}  | {cpu_speed:7.2f}/s | {gpu_speed:7.2f}/s | {speedup:5.1f}x | {total_mem:6.1f}MB")


def print_recommendations(optimal_result):
    """Print detailed recommendations."""
    if not optimal_result:
        print("\n❌ No optimization results available")
        return
    
    optimal_device = optimal_result['optimal_device']
    optimal_batch_size = optimal_result['optimal_batch_size']
    reason = optimal_result['reason']
    
    print(f"\n💡 Recommendations:")
    print("-" * 20)
    
    print(f"🎯 Optimal Configuration:")
    print(f"   Device: {optimal_device.upper()}")
    print(f"   Batch Size: {optimal_batch_size}")
    print(f"   Reason: {reason}")
    
    best = optimal_result['best_result']
    
    if optimal_device == 'gpu':
        print(f"\n🚀 GPU Benefits:")
        print(f"   Speed: {best['gpu_speed']:.2f} images/second")
        print(f"   Speedup: {best['speedup']:.1f}x faster than CPU")
        print(f"   Total Memory: {best['total_memory']:.1f} MB")
        
        print(f"\n⚙️  Configuration Recommendations:")
        print(f"   • Set use_cpu_only = False")
        print(f"   • Set batch_size = {optimal_batch_size}")
        print(f"   • Enable GPU acceleration")
        if best['total_memory'] > 1000:
            print(f"   • Monitor memory usage (high usage detected)")
    
    else:
        print(f"\n🖥️  CPU Benefits:")
        print(f"   Speed: {best['cpu_speed']:.2f} images/second")
        print(f"   Memory: {best['total_memory']:.1f} MB")
        
        print(f"\n⚙️  Configuration Recommendations:")
        print(f"   • Set use_cpu_only = True")
        print(f"   • Set batch_size = {optimal_batch_size}")
        print(f"   • Focus on memory efficiency")
        if optimal_batch_size <= 4:
            print(f"   • Enable aggressive garbage collection")


def apply_optimal_config(optimal_result):
    """Apply the optimal configuration."""
    if not optimal_result:
        return False
    
    optimal_device = optimal_result['optimal_device']
    optimal_batch_size = optimal_result['optimal_batch_size']
    
    print(f"\n🔧 Configuration Update:")
    print("-" * 25)
    
    current_batch_size = MEMORY_CONFIG.batch_size
    current_cpu_only = MEMORY_CONFIG.use_cpu_only
    
    new_cpu_only = (optimal_device == 'cpu')
    
    changes = []
    if optimal_batch_size != current_batch_size:
        changes.append(f"Batch size: {current_batch_size} → {optimal_batch_size}")
    
    if new_cpu_only != current_cpu_only:
        device_change = "GPU enabled → CPU only" if new_cpu_only else "CPU only → GPU enabled"
        changes.append(f"Device: {device_change}")
    
    if not changes:
        print("✅ Current configuration is already optimal!")
        return False
    
    print("Proposed changes:")
    for change in changes:
        print(f"   • {change}")
    
    # Ask user if they want to apply the change
    response = input("\nWould you like to update your configuration? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        # Create new config with optimal settings
        new_config = MemoryConfig(
            use_cpu_only=new_cpu_only,
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
        
        print(f"✅ Configuration updated!")
        print(f"   Device: {optimal_device.upper()}")
        print(f"   Batch size: {optimal_batch_size}")
        print("   Restart the application to use the new settings.")
        return True
    else:
        print("Configuration not changed.")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Find optimal GPU/CPU batch size for real model inference")
    parser.add_argument("--cpu-only", action="store_true", help="Test only CPU performance")
    parser.add_argument("--gpu-only", action="store_true", help="Test only GPU performance")
    parser.add_argument("--no-apply", action="store_true", help="Don't ask to apply configuration")
    
    args = parser.parse_args()
    
    if args.cpu_only and args.gpu_only:
        print("❌ Cannot specify both --cpu-only and --gpu-only")
        return 1
    
    print_header()
    print_system_info()
    
    try:
        optimal_result, comparison = run_gpu_cpu_optimization(
            cpu_only=args.cpu_only,
            gpu_only=args.gpu_only
        )
        
        if not optimal_result and not comparison:
            print("❌ No optimization results available.")
            return 1
        
        # Determine test type for display
        if args.cpu_only:
            test_type = "CPU Only"
        elif args.gpu_only:
            test_type = "GPU Only"
        else:
            test_type = "CPU vs GPU"
        
        print_detailed_results(comparison, test_type)
        
        if optimal_result:
            print_recommendations(optimal_result)
            
            if not args.no_apply:
                config_updated = apply_optimal_config(optimal_result)
                if config_updated:
                    print("\n🔄 Please restart your application to use the new configuration.")
        
        print(f"\n✅ Real model optimization complete!")
        
        if optimal_result:
            print(f"   Recommended: {optimal_result['optimal_device'].upper()} with batch size {optimal_result['optimal_batch_size']}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Optimization cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())