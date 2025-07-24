#!/usr/bin/env python3
"""
Comprehensive Batch Size Testing

This script tests a wide range of batch sizes to find the optimal configuration
and understand performance scaling patterns.

Usage:
    python comprehensive_batch_test.py
    python comprehensive_batch_test.py --cpu-only
    python comprehensive_batch_test.py --gpu-only
    python comprehensive_batch_test.py --extended
"""

import argparse
import sys
import time
from test_gpu_cpu_batch_optimization import GPUCPUBatchOptimizer
from memory_config import MEMORY_CONFIG

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def get_comprehensive_batch_sizes(memory_gb: float, gpu_available: bool, extended: bool = False) -> dict:
    """Get comprehensive batch size ranges for testing."""
    
    if extended:
        # Extended testing with many more batch sizes
        if gpu_available:
            if memory_gb >= 16:
                return {
                    'cpu_batches': [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                    'gpu_batches': [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128],
                    'image_count': 30,
                    'category': 'Extended High Memory + GPU'
                }
            else:
                return {
                    'cpu_batches': [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24],
                    'gpu_batches': [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48],
                    'image_count': 25,
                    'category': 'Extended Medium Memory + GPU'
                }
        else:
            return {
                'cpu_batches': [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48],
                'gpu_batches': [],
                'image_count': 30,
                'category': 'Extended CPU Only'
            }
    
    # Standard comprehensive testing
    if gpu_available:
        if memory_gb < 8:
            return {
                'cpu_batches': [1, 2, 3, 4, 6, 8, 10, 12],
                'gpu_batches': [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24],
                'image_count': 18,
                'category': 'Comprehensive Low Memory + GPU'
            }
        elif memory_gb < 16:
            return {
                'cpu_batches': [2, 3, 4, 6, 8, 10, 12, 16, 20],
                'gpu_batches': [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40],
                'image_count': 22,
                'category': 'Comprehensive Medium Memory + GPU'
            }
        else:
            return {
                'cpu_batches': [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                'gpu_batches': [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64],
                'image_count': 25,
                'category': 'Comprehensive High Memory + GPU'
            }
    else:
        if memory_gb < 8:
            return {
                'cpu_batches': [1, 2, 3, 4, 6, 8, 10, 12, 16, 20],
                'gpu_batches': [],
                'image_count': 20,
                'category': 'Comprehensive Low Memory CPU Only'
            }
        elif memory_gb < 16:
            return {
                'cpu_batches': [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
                'gpu_batches': [],
                'image_count': 25,
                'category': 'Comprehensive Medium Memory CPU Only'
            }
        else:
            return {
                'cpu_batches': [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64],
                'gpu_batches': [],
                'image_count': 30,
                'category': 'Comprehensive High Memory CPU Only'
            }


def run_comprehensive_test(cpu_only=False, gpu_only=False, extended=False):
    """Run comprehensive batch size testing."""
    print("🔬 Comprehensive Batch Size Testing")
    print("=" * 50)
    
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
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"💻 System: {memory_gb:.1f} GB RAM")
                print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                print(f"💻 System: {memory_gb:.1f} GB RAM (CPU Only)")
        except ImportError:
            print(f"💻 System: {memory_gb:.1f} GB RAM (PyTorch not available)")
        
        # Get comprehensive batch sizes
        config = get_comprehensive_batch_sizes(memory_gb, gpu_available, extended)
        
        print(f"📋 Category: {config['category']}")
        print(f"🧪 Testing {len(config['cpu_batches'])} CPU batch sizes")
        if config['gpu_batches']:
            print(f"🚀 Testing {len(config['gpu_batches'])} GPU batch sizes")
        print()
        
        results = {}
        
        # Test CPU
        if not gpu_only:
            print("🖥️  Testing CPU Performance...")
            print(f"   Batch sizes: {config['cpu_batches']}")
            cpu_results = optimizer.test_device_batch_sizes('cpu', config['cpu_batches'], config['image_count'])
            results['cpu'] = cpu_results
        
        # Test GPU
        if gpu_available and not cpu_only:
            print("\n🚀 Testing GPU Performance...")
            print(f"   Batch sizes: {config['gpu_batches']}")
            gpu_results = optimizer.test_device_batch_sizes('cuda', config['gpu_batches'], config['image_count'])
            results['gpu'] = gpu_results
        
        return results, config
        
    finally:
        optimizer.cleanup()


def analyze_performance_scaling(results, config):
    """Analyze how performance scales with batch size."""
    print("\n📊 Performance Scaling Analysis:")
    print("=" * 40)
    
    for device, device_results in results.items():
        if not device_results:
            continue
            
        print(f"\n{device.upper()} Performance Scaling:")
        print("   Batch | Speed    | Memory   | Efficiency | Scaling")
        print("   ------|----------|----------|------------|--------")
        
        prev_speed = None
        for i, result in enumerate(device_results):
            batch = result['batch_size']
            speed = result['avg_images_per_second']
            
            if device == 'cpu':
                memory = result['avg_cpu_memory']
            else:
                memory = result['avg_cpu_memory'] + result['avg_gpu_memory']
            
            efficiency = speed / batch if batch > 0 else 0
            
            if prev_speed is not None:
                scaling = speed / prev_speed
                scaling_str = f"{scaling:.2f}x"
            else:
                scaling_str = "baseline"
            
            print(f"   {batch:4d}  | {speed:6.2f}/s | {memory:6.1f}MB | {efficiency:8.2f}   | {scaling_str}")
            prev_speed = speed
        
        # Find optimal batch size for this device
        best_result = max(device_results, key=lambda x: x['avg_images_per_second'])
        print(f"\n   🏆 Best {device.upper()}: Batch {best_result['batch_size']} at {best_result['avg_images_per_second']:.2f} img/s")


def find_performance_plateau(results):
    """Find where performance plateaus or starts declining."""
    print("\n🔍 Performance Plateau Analysis:")
    print("-" * 35)
    
    for device, device_results in results.items():
        if not device_results or len(device_results) < 3:
            continue
        
        speeds = [r['avg_images_per_second'] for r in device_results]
        batch_sizes = [r['batch_size'] for r in device_results]
        
        # Find peak performance
        peak_idx = speeds.index(max(speeds))
        peak_batch = batch_sizes[peak_idx]
        peak_speed = speeds[peak_idx]
        
        # Find where performance starts declining (if it does)
        decline_start = None
        for i in range(peak_idx + 1, len(speeds)):
            if speeds[i] < peak_speed * 0.95:  # 5% decline threshold
                decline_start = batch_sizes[i]
                break
        
        # Find diminishing returns point (where improvement is < 5%)
        diminishing_point = None
        for i in range(1, len(speeds)):
            if i > 0:
                improvement = (speeds[i] - speeds[i-1]) / speeds[i-1]
                if improvement < 0.05:  # Less than 5% improvement
                    diminishing_point = batch_sizes[i]
                    break
        
        print(f"\n{device.upper()} Analysis:")
        print(f"   Peak Performance: Batch {peak_batch} ({peak_speed:.2f} img/s)")
        
        if diminishing_point:
            print(f"   Diminishing Returns: Starts at batch {diminishing_point}")
        
        if decline_start:
            print(f"   Performance Decline: Starts at batch {decline_start}")
        else:
            print(f"   Performance Decline: Not observed in tested range")


def plot_performance_curves(results, config, save_plot=True):
    """Plot performance curves for visualization."""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️  Matplotlib not available, skipping plot generation")
        return
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot CPU results
        if 'cpu' in results and results['cpu']:
            cpu_batches = [r['batch_size'] for r in results['cpu']]
            cpu_speeds = [r['avg_images_per_second'] for r in results['cpu']]
            plt.plot(cpu_batches, cpu_speeds, 'b-o', label='CPU', linewidth=2, markersize=6)
        
        # Plot GPU results
        if 'gpu' in results and results['gpu']:
            gpu_batches = [r['batch_size'] for r in results['gpu']]
            gpu_speeds = [r['avg_images_per_second'] for r in results['gpu']]
            plt.plot(gpu_batches, gpu_speeds, 'r-o', label='GPU', linewidth=2, markersize=6)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Images per Second')
        plt.title(f'Batch Size Performance Scaling\n{config["category"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('linear')
        
        # Add annotations for peak performance
        for device, device_results in results.items():
            if device_results:
                best_result = max(device_results, key=lambda x: x['avg_images_per_second'])
                plt.annotate(f'{device.upper()} Peak\n({best_result["batch_size"]}, {best_result["avg_images_per_second"]:.1f})',
                           xy=(best_result['batch_size'], best_result['avg_images_per_second']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('batch_size_performance.png', dpi=300, bbox_inches='tight')
            print(f"\n📈 Performance plot saved as 'batch_size_performance.png'")
        
        plt.show()
        
    except ImportError:
        print("\n⚠️  Matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"\n⚠️  Error generating plot: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive batch size testing")
    parser.add_argument("--cpu-only", action="store_true", help="Test only CPU performance")
    parser.add_argument("--gpu-only", action="store_true", help="Test only GPU performance")
    parser.add_argument("--extended", action="store_true", help="Extended testing with more batch sizes")
    parser.add_argument("--no-plot", action="store_true", help="Don't generate performance plots")
    
    args = parser.parse_args()
    
    if args.cpu_only and args.gpu_only:
        print("❌ Cannot specify both --cpu-only and --gpu-only")
        return 1
    
    try:
        # Run comprehensive testing
        results, config = run_comprehensive_test(
            cpu_only=args.cpu_only,
            gpu_only=args.gpu_only,
            extended=args.extended
        )
        
        if not results:
            print("❌ No test results available.")
            return 1
        
        # Analyze results
        analyze_performance_scaling(results, config)
        find_performance_plateau(results)
        
        # Generate plots
        if not args.no_plot:
            plot_performance_curves(results, config)
        
        # Summary recommendations
        print("\n🎯 Summary Recommendations:")
        print("-" * 30)
        
        for device, device_results in results.items():
            if device_results:
                best_result = max(device_results, key=lambda x: x['avg_images_per_second'])
                
                # Find sweet spot (good performance with reasonable memory)
                sweet_spot = None
                for result in device_results:
                    speed_ratio = result['avg_images_per_second'] / best_result['avg_images_per_second']
                    if device == 'cpu':
                        memory = result['avg_cpu_memory']
                    else:
                        memory = result['avg_cpu_memory'] + result['avg_gpu_memory']
                    
                    # Sweet spot: >= 90% of peak performance with reasonable memory
                    if speed_ratio >= 0.9 and memory < 200:
                        sweet_spot = result
                        break
                
                if sweet_spot:
                    print(f"\n{device.upper()} Recommendations:")
                    print(f"   Peak Performance: Batch {best_result['batch_size']} ({best_result['avg_images_per_second']:.2f} img/s)")
                    print(f"   Sweet Spot: Batch {sweet_spot['batch_size']} ({sweet_spot['avg_images_per_second']:.2f} img/s)")
                    print(f"   Memory Usage: {memory:.1f} MB")
        
        print(f"\n✅ Comprehensive testing complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())