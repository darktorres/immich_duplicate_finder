#!/usr/bin/env python3
"""
Performance Extrapolator for Different Hardware Configurations

This tool extrapolates batch size performance for various CPU and GPU configurations
based on observed performance patterns and hardware specifications.

Usage:
    python performance_extrapolator.py
    python performance_extrapolator.py --hardware-config high-end
    python performance_extrapolator.py --custom-gpu "RTX 4090"
"""

import argparse
import sys
from typing import Dict, List, Tuple
import json


class HardwareDatabase:
    """Database of hardware specifications and performance characteristics."""
    
    def __init__(self):
        self.cpu_specs = {
            # Current baseline (your system)
            "Intel i7-12700": {
                "cores": 12,
                "base_clock": 2.1,
                "boost_clock": 4.9,
                "threads": 20,
                "tdp": 65,
                "performance_multiplier": 1.0,  # Baseline
                "memory_bandwidth": 76.8,
                "category": "mid-range"
            },
            
            # Budget CPUs
            "Intel i5-12400": {
                "cores": 6,
                "base_clock": 2.5,
                "boost_clock": 4.4,
                "threads": 12,
                "tdp": 65,
                "performance_multiplier": 0.7,
                "memory_bandwidth": 76.8,
                "category": "budget"
            },
            "AMD Ryzen 5 5600X": {
                "cores": 6,
                "base_clock": 3.7,
                "boost_clock": 4.6,
                "threads": 12,
                "tdp": 65,
                "performance_multiplier": 0.75,
                "memory_bandwidth": 51.2,
                "category": "budget"
            },
            
            # Mid-range CPUs
            "Intel i7-13700K": {
                "cores": 16,
                "base_clock": 3.4,
                "boost_clock": 5.4,
                "threads": 24,
                "tdp": 125,
                "performance_multiplier": 1.4,
                "memory_bandwidth": 89.6,
                "category": "mid-range"
            },
            "AMD Ryzen 7 7700X": {
                "cores": 8,
                "base_clock": 4.5,
                "boost_clock": 5.4,
                "threads": 16,
                "tdp": 105,
                "performance_multiplier": 1.3,
                "memory_bandwidth": 83.2,
                "category": "mid-range"
            },
            
            # High-end CPUs
            "Intel i9-13900K": {
                "cores": 24,
                "base_clock": 3.0,
                "boost_clock": 5.8,
                "threads": 32,
                "tdp": 125,
                "performance_multiplier": 1.8,
                "memory_bandwidth": 89.6,
                "category": "high-end"
            },
            "AMD Ryzen 9 7950X": {
                "cores": 16,
                "base_clock": 4.5,
                "boost_clock": 5.7,
                "threads": 32,
                "tdp": 170,
                "performance_multiplier": 1.9,
                "memory_bandwidth": 83.2,
                "category": "high-end"
            },
            
            # Workstation CPUs
            "Intel Xeon W-3275": {
                "cores": 28,
                "base_clock": 2.5,
                "boost_clock": 4.4,
                "threads": 56,
                "tdp": 205,
                "performance_multiplier": 2.2,
                "memory_bandwidth": 131.2,
                "category": "workstation"
            },
            "AMD Threadripper 7970X": {
                "cores": 32,
                "base_clock": 4.0,
                "boost_clock": 5.3,
                "threads": 64,
                "tdp": 350,
                "performance_multiplier": 2.5,
                "memory_bandwidth": 204.8,
                "category": "workstation"
            }
        }
        
        self.gpu_specs = {
            # Current baseline (your GPU)
            "RTX 2060": {
                "cuda_cores": 1920,
                "base_clock": 1365,
                "boost_clock": 1680,
                "memory": 6,
                "memory_bandwidth": 336,
                "tensor_cores": 240,
                "performance_multiplier": 1.0,  # Baseline
                "category": "mid-range",
                "generation": "Turing"
            },
            
            # Budget GPUs
            "GTX 1660 Super": {
                "cuda_cores": 1408,
                "base_clock": 1530,
                "boost_clock": 1785,
                "memory": 6,
                "memory_bandwidth": 336,
                "tensor_cores": 0,
                "performance_multiplier": 0.7,
                "category": "budget",
                "generation": "Turing"
            },
            "RTX 3060": {
                "cuda_cores": 3584,
                "base_clock": 1320,
                "boost_clock": 1777,
                "memory": 12,
                "memory_bandwidth": 360,
                "tensor_cores": 112,
                "performance_multiplier": 1.3,
                "category": "budget",
                "generation": "Ampere"
            },
            
            # Mid-range GPUs
            "RTX 3070": {
                "cuda_cores": 5888,
                "base_clock": 1500,
                "boost_clock": 1725,
                "memory": 8,
                "memory_bandwidth": 448,
                "tensor_cores": 184,
                "performance_multiplier": 1.8,
                "category": "mid-range",
                "generation": "Ampere"
            },
            "RTX 4060 Ti": {
                "cuda_cores": 4352,
                "base_clock": 2310,
                "boost_clock": 2535,
                "memory": 16,
                "memory_bandwidth": 288,
                "tensor_cores": 136,
                "performance_multiplier": 1.6,
                "category": "mid-range",
                "generation": "Ada Lovelace"
            },
            
            # High-end GPUs
            "RTX 3080": {
                "cuda_cores": 8704,
                "base_clock": 1440,
                "boost_clock": 1710,
                "memory": 10,
                "memory_bandwidth": 760,
                "tensor_cores": 272,
                "performance_multiplier": 2.4,
                "category": "high-end",
                "generation": "Ampere"
            },
            "RTX 4070 Ti": {
                "cuda_cores": 7680,
                "base_clock": 2310,
                "boost_clock": 2610,
                "memory": 12,
                "memory_bandwidth": 504,
                "tensor_cores": 240,
                "performance_multiplier": 2.2,
                "category": "high-end",
                "generation": "Ada Lovelace"
            },
            "RTX 4080": {
                "cuda_cores": 9728,
                "base_clock": 2205,
                "boost_clock": 2505,
                "memory": 16,
                "memory_bandwidth": 717,
                "tensor_cores": 304,
                "performance_multiplier": 2.8,
                "category": "high-end",
                "generation": "Ada Lovelace"
            },
            
            # Enthusiast GPUs
            "RTX 3090": {
                "cuda_cores": 10496,
                "base_clock": 1395,
                "boost_clock": 1695,
                "memory": 24,
                "memory_bandwidth": 936,
                "tensor_cores": 328,
                "performance_multiplier": 2.9,
                "category": "enthusiast",
                "generation": "Ampere"
            },
            "RTX 4090": {
                "cuda_cores": 16384,
                "base_clock": 2230,
                "boost_clock": 2520,
                "memory": 24,
                "memory_bandwidth": 1008,
                "tensor_cores": 512,
                "performance_multiplier": 3.8,
                "category": "enthusiast",
                "generation": "Ada Lovelace"
            },
            
            # Professional GPUs
            "RTX A6000": {
                "cuda_cores": 10752,
                "base_clock": 1410,
                "boost_clock": 1800,
                "memory": 48,
                "memory_bandwidth": 768,
                "tensor_cores": 336,
                "performance_multiplier": 3.2,
                "category": "professional",
                "generation": "Ampere"
            },
            "RTX 6000 Ada": {
                "cuda_cores": 18176,
                "base_clock": 915,
                "boost_clock": 2505,
                "memory": 48,
                "memory_bandwidth": 960,
                "tensor_cores": 568,
                "performance_multiplier": 4.2,
                "category": "professional",
                "generation": "Ada Lovelace"
            }
        }


class PerformanceExtrapolator:
    """Extrapolates performance for different hardware configurations."""
    
    def __init__(self):
        self.hardware_db = HardwareDatabase()
        
        # Baseline performance data from your RTX 2060 + i7-12700 system
        self.baseline_cpu_performance = {
            2: 2.80, 4: 3.69, 6: 3.88, 8: 4.16, 10: 4.21, 12: 4.18,
            16: 4.33, 20: 4.17, 24: 4.33, 28: 4.42, 32: 4.44
        }
        
        self.baseline_gpu_performance = {
            2: 6.42, 4: 7.66, 6: 9.64, 8: 9.70, 10: 11.41, 12: 11.28,
            16: 12.10, 20: 11.91, 24: 11.65, 28: 12.40, 32: 13.16,
            40: 12.39, 48: 12.22, 64: 12.30
        }
        
        self.baseline_cpu_memory = {
            2: 14.2, 4: 2.0, 6: 13.6, 8: 23.1, 10: 35.4, 12: 65.4,
            16: 44.3, 20: 69.8, 24: 121.0, 28: 8.2, 32: 0.6
        }
        
        self.baseline_gpu_memory = {
            2: 87.7, 4: 4.9, 6: 11.3, 8: 20.1, 10: 16.1, 12: 17.6,
            16: 15.0, 20: 29.5, 24: 32.2, 28: 16.5, 32: 15.6,
            40: 15.6, 48: 15.6, 64: 15.6
        }
    
    def extrapolate_cpu_performance(self, cpu_name: str) -> Dict[int, Dict]:
        """Extrapolate CPU performance for a given processor."""
        if cpu_name not in self.hardware_db.cpu_specs:
            raise ValueError(f"CPU '{cpu_name}' not found in database")
        
        cpu_spec = self.hardware_db.cpu_specs[cpu_name]
        multiplier = cpu_spec['performance_multiplier']
        
        # Performance scaling factors based on CPU characteristics
        core_factor = min(cpu_spec['cores'] / 12, 2.0)  # More cores help up to a point
        clock_factor = cpu_spec['boost_clock'] / 4.9  # Clock speed scaling
        memory_factor = cpu_spec['memory_bandwidth'] / 76.8  # Memory bandwidth scaling
        
        # Combined scaling factor
        combined_factor = multiplier * (0.4 * core_factor + 0.4 * clock_factor + 0.2 * memory_factor)
        
        results = {}
        for batch_size, baseline_perf in self.baseline_cpu_performance.items():
            # Apply diminishing returns for very high performance
            scaled_perf = baseline_perf * combined_factor
            if combined_factor > 2.0:
                scaled_perf = baseline_perf * (2.0 + (combined_factor - 2.0) * 0.7)
            
            # Memory scaling (higher performance CPUs may use more memory)
            memory_scaling = 1.0 + (combined_factor - 1.0) * 0.3
            scaled_memory = self.baseline_cpu_memory[batch_size] * memory_scaling
            
            results[batch_size] = {
                'performance': scaled_perf,
                'memory_mb': scaled_memory,
                'efficiency': scaled_perf / batch_size
            }
        
        return results
    
    def extrapolate_gpu_performance(self, gpu_name: str) -> Dict[int, Dict]:
        """Extrapolate GPU performance for a given graphics card."""
        if gpu_name not in self.hardware_db.gpu_specs:
            raise ValueError(f"GPU '{gpu_name}' not found in database")
        
        gpu_spec = self.hardware_db.gpu_specs[gpu_name]
        multiplier = gpu_spec['performance_multiplier']
        
        # Performance scaling factors based on GPU characteristics
        cuda_factor = gpu_spec['cuda_cores'] / 1920  # CUDA cores scaling
        clock_factor = gpu_spec['boost_clock'] / 1680  # Clock speed scaling
        memory_bw_factor = gpu_spec['memory_bandwidth'] / 336  # Memory bandwidth scaling
        tensor_factor = 1.0 + (gpu_spec['tensor_cores'] / 240) * 0.2  # Tensor cores bonus
        
        # Generation improvements
        generation_bonus = {
            'Turing': 1.0,
            'Ampere': 1.15,  # Architecture improvements
            'Ada Lovelace': 1.25  # Further improvements
        }.get(gpu_spec['generation'], 1.0)
        
        # Combined scaling factor
        combined_factor = multiplier * generation_bonus * (
            0.35 * cuda_factor + 
            0.25 * clock_factor + 
            0.25 * memory_bw_factor + 
            0.15 * tensor_factor
        )
        
        results = {}
        for batch_size, baseline_perf in self.baseline_gpu_performance.items():
            # GPU performance scales better with larger batch sizes
            batch_scaling = 1.0 + (batch_size - 2) * 0.02 * (combined_factor - 1.0)
            scaled_perf = baseline_perf * combined_factor * batch_scaling
            
            # Apply diminishing returns for very high performance
            if combined_factor > 3.0:
                scaled_perf = baseline_perf * (3.0 + (combined_factor - 3.0) * 0.8) * batch_scaling
            
            # Memory scaling (more powerful GPUs may use memory more efficiently)
            memory_efficiency = 1.0 - (combined_factor - 1.0) * 0.1
            memory_efficiency = max(memory_efficiency, 0.5)  # Don't go below 50%
            scaled_memory = self.baseline_gpu_memory[batch_size] * memory_efficiency
            
            # Account for larger VRAM
            if gpu_spec['memory'] > 6:
                memory_bonus = min(gpu_spec['memory'] / 6, 2.0)
                scaled_memory *= memory_bonus
            
            results[batch_size] = {
                'performance': scaled_perf,
                'memory_mb': scaled_memory,
                'efficiency': scaled_perf / batch_size,
                'speedup_vs_baseline': scaled_perf / baseline_perf
            }
        
        return results
    
    def find_optimal_batch_size(self, cpu_results: Dict, gpu_results: Dict) -> Dict:
        """Find optimal batch size for given hardware configuration."""
        
        # Find best CPU batch size
        best_cpu_batch = max(cpu_results.keys(), key=lambda b: cpu_results[b]['performance'])
        best_cpu_perf = cpu_results[best_cpu_batch]['performance']
        
        # Find best GPU batch size
        best_gpu_batch = max(gpu_results.keys(), key=lambda b: gpu_results[b]['performance'])
        best_gpu_perf = gpu_results[best_gpu_batch]['performance']
        
        # Find sweet spot (90% of peak performance with reasonable memory)
        cpu_sweet_spot = None
        gpu_sweet_spot = None
        
        for batch_size in sorted(cpu_results.keys()):
            perf_ratio = cpu_results[batch_size]['performance'] / best_cpu_perf
            if perf_ratio >= 0.9 and cpu_results[batch_size]['memory_mb'] < 100:
                cpu_sweet_spot = batch_size
                break
        
        for batch_size in sorted(gpu_results.keys()):
            perf_ratio = gpu_results[batch_size]['performance'] / best_gpu_perf
            if perf_ratio >= 0.9 and gpu_results[batch_size]['memory_mb'] < 200:
                gpu_sweet_spot = batch_size
                break
        
        return {
            'cpu_optimal': {
                'batch_size': best_cpu_batch,
                'performance': best_cpu_perf,
                'memory': cpu_results[best_cpu_batch]['memory_mb']
            },
            'cpu_sweet_spot': {
                'batch_size': cpu_sweet_spot or best_cpu_batch,
                'performance': cpu_results[cpu_sweet_spot or best_cpu_batch]['performance'],
                'memory': cpu_results[cpu_sweet_spot or best_cpu_batch]['memory_mb']
            },
            'gpu_optimal': {
                'batch_size': best_gpu_batch,
                'performance': best_gpu_perf,
                'memory': gpu_results[best_gpu_batch]['memory_mb']
            },
            'gpu_sweet_spot': {
                'batch_size': gpu_sweet_spot or best_gpu_batch,
                'performance': gpu_results[gpu_sweet_spot or best_gpu_batch]['performance'],
                'memory': gpu_results[gpu_sweet_spot or best_gpu_batch]['memory_mb']
            },
            'gpu_advantage': best_gpu_perf / best_cpu_perf
        }
    
    def generate_hardware_comparison(self, hardware_configs: List[Tuple[str, str]]) -> Dict:
        """Generate performance comparison for multiple hardware configurations."""
        
        results = {}
        
        for cpu_name, gpu_name in hardware_configs:
            try:
                cpu_results = self.extrapolate_cpu_performance(cpu_name)
                gpu_results = self.extrapolate_gpu_performance(gpu_name)
                optimal = self.find_optimal_batch_size(cpu_results, gpu_results)
                
                results[f"{cpu_name} + {gpu_name}"] = {
                    'cpu_results': cpu_results,
                    'gpu_results': gpu_results,
                    'optimal': optimal,
                    'cpu_spec': self.hardware_db.cpu_specs[cpu_name],
                    'gpu_spec': self.hardware_db.gpu_specs[gpu_name]
                }
                
            except ValueError as e:
                print(f"⚠️  Error processing {cpu_name} + {gpu_name}: {e}")
        
        return results


def print_performance_table(cpu_results: Dict, gpu_results: Dict, cpu_name: str, gpu_name: str):
    """Print a formatted performance table."""
    
    print(f"\n📊 Performance Extrapolation: {cpu_name} + {gpu_name}")
    print("=" * 80)
    
    print("\n🖥️  CPU Performance:")
    print("   Batch | Speed     | Memory   | Efficiency | vs Baseline")
    print("   ------|-----------|----------|------------|------------")
    
    for batch_size in sorted(cpu_results.keys()):
        result = cpu_results[batch_size]
        baseline_perf = 4.44 if batch_size == 32 else result['performance'] / 1.5  # Rough baseline
        vs_baseline = result['performance'] / baseline_perf
        
        print(f"   {batch_size:4d}  | {result['performance']:7.2f}/s | {result['memory_mb']:6.1f}MB | {result['efficiency']:8.2f}   | {vs_baseline:6.2f}x")
    
    print("\n🚀 GPU Performance:")
    print("   Batch | Speed     | Memory   | Efficiency | vs Baseline | Speedup")
    print("   ------|-----------|----------|------------|-------------|--------")
    
    for batch_size in sorted(gpu_results.keys()):
        result = gpu_results[batch_size]
        cpu_perf = cpu_results.get(batch_size, {}).get('performance', 1)
        gpu_speedup = result['performance'] / cpu_perf
        
        print(f"   {batch_size:4d}  | {result['performance']:7.2f}/s | {result['memory_mb']:6.1f}MB | {result['efficiency']:8.2f}   | {result['speedup_vs_baseline']:9.2f}x | {gpu_speedup:5.1f}x")


def print_hardware_comparison(comparison_results: Dict):
    """Print comparison between different hardware configurations."""
    
    print(f"\n🏆 Hardware Configuration Comparison")
    print("=" * 100)
    
    print(f"\n{'Configuration':<30} | {'CPU Peak':<12} | {'GPU Peak':<12} | {'GPU Advantage':<12} | {'Optimal Batch':<12}")
    print("-" * 100)
    
    # Sort by GPU performance
    sorted_configs = sorted(comparison_results.items(), 
                          key=lambda x: x[1]['optimal']['gpu_optimal']['performance'], 
                          reverse=True)
    
    for config_name, results in sorted_configs:
        cpu_peak = results['optimal']['cpu_optimal']['performance']
        gpu_peak = results['optimal']['gpu_optimal']['performance']
        gpu_advantage = results['optimal']['gpu_advantage']
        optimal_batch = results['optimal']['gpu_optimal']['batch_size']
        
        print(f"{config_name:<30} | {cpu_peak:9.2f}/s | {gpu_peak:9.2f}/s | {gpu_advantage:9.1f}x | {optimal_batch:9d}")


def get_predefined_configs():
    """Get predefined hardware configurations for comparison."""
    
    return {
        'budget': [
            ("Intel i5-12400", "RTX 3060"),
            ("AMD Ryzen 5 5600X", "GTX 1660 Super"),
        ],
        'mid-range': [
            ("Intel i7-12700", "RTX 2060"),  # Your current system
            ("Intel i7-13700K", "RTX 3070"),
            ("AMD Ryzen 7 7700X", "RTX 4060 Ti"),
        ],
        'high-end': [
            ("Intel i9-13900K", "RTX 4080"),
            ("AMD Ryzen 9 7950X", "RTX 3080"),
            ("Intel i7-13700K", "RTX 4070 Ti"),
        ],
        'enthusiast': [
            ("Intel i9-13900K", "RTX 4090"),
            ("AMD Ryzen 9 7950X", "RTX 3090"),
            ("AMD Threadripper 7970X", "RTX 4090"),
        ],
        'professional': [
            ("Intel Xeon W-3275", "RTX A6000"),
            ("AMD Threadripper 7970X", "RTX 6000 Ada"),
        ]
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extrapolate batch size performance for different hardware")
    parser.add_argument("--hardware-config", choices=['budget', 'mid-range', 'high-end', 'enthusiast', 'professional'], 
                       help="Predefined hardware configuration category")
    parser.add_argument("--custom-cpu", help="Custom CPU name")
    parser.add_argument("--custom-gpu", help="Custom GPU name")
    parser.add_argument("--list-hardware", action="store_true", help="List available hardware")
    parser.add_argument("--compare-all", action="store_true", help="Compare all predefined configurations")
    
    args = parser.parse_args()
    
    extrapolator = PerformanceExtrapolator()
    
    if args.list_hardware:
        print("🖥️  Available CPUs:")
        for cpu_name, spec in extrapolator.hardware_db.cpu_specs.items():
            print(f"   {cpu_name:<25} | {spec['cores']:2d} cores | {spec['boost_clock']:.1f} GHz | {spec['category']}")
        
        print(f"\n🚀 Available GPUs:")
        for gpu_name, spec in extrapolator.hardware_db.gpu_specs.items():
            print(f"   {gpu_name:<20} | {spec['cuda_cores']:5d} cores | {spec['memory']:2d}GB | {spec['category']}")
        
        return 0
    
    try:
        if args.custom_cpu and args.custom_gpu:
            # Custom hardware configuration
            cpu_results = extrapolator.extrapolate_cpu_performance(args.custom_cpu)
            gpu_results = extrapolator.extrapolate_gpu_performance(args.custom_gpu)
            
            print_performance_table(cpu_results, gpu_results, args.custom_cpu, args.custom_gpu)
            
            optimal = extrapolator.find_optimal_batch_size(cpu_results, gpu_results)
            
            print(f"\n🎯 Optimal Configuration:")
            print(f"   GPU Peak: Batch {optimal['gpu_optimal']['batch_size']} ({optimal['gpu_optimal']['performance']:.2f} img/s)")
            print(f"   GPU Sweet Spot: Batch {optimal['gpu_sweet_spot']['batch_size']} ({optimal['gpu_sweet_spot']['performance']:.2f} img/s)")
            print(f"   GPU Advantage: {optimal['gpu_advantage']:.1f}x faster than CPU")
        
        elif args.hardware_config:
            # Predefined configuration category
            configs = get_predefined_configs()
            hardware_list = configs[args.hardware_config]
            
            print(f"🔬 {args.hardware_config.title()} Hardware Configurations")
            
            comparison = extrapolator.generate_hardware_comparison(hardware_list)
            print_hardware_comparison(comparison)
            
            # Show detailed results for best configuration
            best_config = max(comparison.items(), 
                            key=lambda x: x[1]['optimal']['gpu_optimal']['performance'])
            
            config_name, results = best_config
            cpu_name = config_name.split(' + ')[0]
            gpu_name = config_name.split(' + ')[1]
            
            print(f"\n🏆 Best {args.hardware_config.title()} Configuration: {config_name}")
            print_performance_table(results['cpu_results'], results['gpu_results'], cpu_name, gpu_name)
        
        elif args.compare_all:
            # Compare all categories
            all_configs = []
            for category, configs in get_predefined_configs().items():
                all_configs.extend(configs)
            
            comparison = extrapolator.generate_hardware_comparison(all_configs)
            print_hardware_comparison(comparison)
        
        else:
            # Default: show your current system + some comparisons
            current_system = [("Intel i7-12700", "RTX 2060")]
            popular_configs = [
                ("Intel i7-13700K", "RTX 4070 Ti"),
                ("Intel i9-13900K", "RTX 4090"),
                ("AMD Ryzen 9 7950X", "RTX 4080"),
            ]
            
            all_configs = current_system + popular_configs
            comparison = extrapolator.generate_hardware_comparison(all_configs)
            
            print("🔬 Performance Extrapolation for Popular Hardware Configurations")
            print_hardware_comparison(comparison)
            
            # Show detailed results for RTX 4090 system
            rtx4090_config = next((k, v) for k, v in comparison.items() if "RTX 4090" in k)
            if rtx4090_config:
                config_name, results = rtx4090_config
                cpu_name = config_name.split(' + ')[0]
                gpu_name = config_name.split(' + ')[1]
                
                print(f"\n🚀 Detailed Analysis: {config_name}")
                print_performance_table(results['cpu_results'], results['gpu_results'], cpu_name, gpu_name)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())