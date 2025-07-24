"""
Quick batch size optimization tests for immediate feedback.

This module provides fast tests to quickly determine optimal batch sizes
without requiring extensive setup or long execution times.
"""

import gc
import time
import tempfile
import os
from typing import Dict, List
import pytest
import psutil
import numpy as np
from PIL import Image

from memory_config import MemoryConfig


class QuickBatchOptimizer:
    """Quick optimizer for finding reasonable batch sizes."""
    
    def __init__(self):
        self.baseline_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def simulate_image_processing(self, batch_size: int, num_images: int = 20) -> Dict:
        """Simulate image processing with different batch sizes."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        max_memory = start_memory
        
        # Simulate image data (smaller than real images for speed)
        processed_count = 0
        
        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            batch_size_actual = batch_end - batch_start
            
            # Simulate feature extraction (create arrays similar to image features)
            batch_features = []
            for _ in range(batch_size_actual):
                # Simulate ResNet features (768 dimensions)
                features = np.random.rand(768).astype(np.float32)
                batch_features.append(features)
                processed_count += 1
            
            # Check memory usage
            current_memory = self._get_memory_usage()
            max_memory = max(max_memory, current_memory)
            
            # Simulate cleanup
            del batch_features
            gc.collect()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        total_time = end_time - start_time
        
        return {
            'batch_size': batch_size,
            'total_time': total_time,
            'images_per_second': processed_count / max(0.001, total_time),
            'start_memory_mb': start_memory,
            'max_memory_mb': max_memory,
            'end_memory_mb': end_memory,
            'memory_increase_mb': max_memory - start_memory,
            'processed_count': processed_count,
            'memory_per_image': (max_memory - start_memory) / max(1, processed_count)
        }
    
    def test_batch_sizes(self, batch_sizes: List[int], num_images: int = 30) -> List[Dict]:
        """Test multiple batch sizes and return results."""
        results = []
        
        print(f"🧪 Testing batch sizes: {batch_sizes}")
        print(f"📊 Processing {num_images} simulated images")
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size {batch_size}...")
            
            # Run test multiple times and average
            test_results = []
            for _ in range(3):
                result = self.simulate_image_processing(batch_size, num_images)
                test_results.append(result)
                time.sleep(0.1)  # Brief pause between tests
            
            # Calculate averages
            avg_result = {
                'batch_size': batch_size,
                'avg_time': sum(r['total_time'] for r in test_results) / len(test_results),
                'avg_speed': sum(r['images_per_second'] for r in test_results) / len(test_results),
                'avg_memory_increase': sum(r['memory_increase_mb'] for r in test_results) / len(test_results),
                'avg_memory_per_image': sum(r['memory_per_image'] for r in test_results) / len(test_results),
                'max_memory_increase': max(r['memory_increase_mb'] for r in test_results),
                'min_speed': min(r['images_per_second'] for r in test_results)
            }
            
            results.append(avg_result)
        
        return results
    
    def find_optimal_batch_size(self, results: List[Dict]) -> Dict:
        """Find the optimal batch size from test results."""
        if not results:
            return {}
        
        # Score each batch size
        scored_results = []
        
        # Get ranges for normalization
        speeds = [r['avg_speed'] for r in results]
        memories = [r['avg_memory_increase'] for r in results]
        
        max_speed = max(speeds) if speeds else 1
        max_memory = max(memories) if memories else 1
        min_memory = min(m for m in memories if m > 0) if any(m > 0 for m in memories) else 1
        
        for result in results:
            # Normalize scores (0-1 range)
            speed_score = result['avg_speed'] / max_speed if max_speed > 0 else 0
            
            # Memory score: lower memory usage is better, normalize to 0-1
            memory_increase = max(result['avg_memory_increase'], 0.1)
            memory_score = min_memory / memory_increase
            memory_score = min(memory_score, 1.0)  # Cap at 1.0
            
            # Stability score (prefer consistent performance)
            stability_score = result['min_speed'] / max(result['avg_speed'], 0.1)
            stability_score = min(stability_score, 1.0)  # Cap at 1.0
            
            # Composite score (weighted) - ensure it stays 0-1
            composite_score = (
                speed_score * 0.5 +      # 50% weight on speed
                memory_score * 0.3 +     # 30% weight on memory efficiency
                stability_score * 0.2    # 20% weight on stability
            )
            
            scored_results.append({
                **result,
                'speed_score': speed_score,
                'memory_score': memory_score,
                'stability_score': stability_score,
                'composite_score': composite_score
            })
        
        # Find best result
        best_result = max(scored_results, key=lambda x: x['composite_score'])
        
        return {
            'optimal_batch_size': best_result['batch_size'],
            'best_result': best_result,
            'all_results': scored_results
        }


def get_system_recommendations() -> Dict:
    """Get batch size recommendations based on system specs."""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    if memory_gb < 4:
        return {
            'category': 'Low Memory',
            'recommended_range': [1, 2, 3, 5],
            'max_safe': 5,
            'description': 'Very conservative batch sizes for low memory systems'
        }
    elif memory_gb < 8:
        return {
            'category': 'Medium Memory',
            'recommended_range': [2, 5, 8, 10, 15],
            'max_safe': 15,
            'description': 'Balanced batch sizes for medium memory systems'
        }
    elif memory_gb < 16:
        return {
            'category': 'High Memory',
            'recommended_range': [5, 10, 15, 20, 25, 30],
            'max_safe': 30,
            'description': 'Larger batch sizes for high memory systems'
        }
    else:
        return {
            'category': 'Very High Memory',
            'recommended_range': [10, 20, 30, 50, 75, 100],
            'max_safe': 100,
            'description': 'Large batch sizes for very high memory systems'
        }


@pytest.mark.unit
class TestQuickBatchOptimization:
    """Quick tests for batch size optimization."""
    
    def setup_method(self):
        """Set up test environment."""
        self.optimizer = QuickBatchOptimizer()
    
    def test_basic_batch_processing(self):
        """Test basic batch processing simulation."""
        result = self.optimizer.simulate_image_processing(batch_size=10, num_images=20)
        
        assert result['batch_size'] == 10
        assert result['processed_count'] == 20
        assert result['total_time'] > 0
        assert result['images_per_second'] > 0
        assert result['memory_increase_mb'] >= 0
    
    def test_small_batch_sizes(self):
        """Test small batch sizes for memory-constrained systems."""
        batch_sizes = [1, 2, 5]
        results = self.optimizer.test_batch_sizes(batch_sizes, num_images=15)
        
        assert len(results) == len(batch_sizes)
        
        # Verify results are reasonable
        for result in results:
            assert result['avg_speed'] > 0
            assert result['avg_memory_increase'] >= 0
            assert result['batch_size'] in batch_sizes
    
    def test_medium_batch_sizes(self):
        """Test medium batch sizes for balanced systems."""
        batch_sizes = [5, 10, 15, 20]
        results = self.optimizer.test_batch_sizes(batch_sizes, num_images=25)
        
        assert len(results) == len(batch_sizes)
        
        # Check for reasonable performance scaling
        for i in range(1, len(results)):
            current = results[i]
            previous = results[i-1]
            
            # Speed should generally improve or stay similar with larger batches
            speed_ratio = current['avg_speed'] / previous['avg_speed']
            assert speed_ratio > 0.5, f"Significant speed degradation at batch size {current['batch_size']}"
    
    def test_optimal_batch_size_selection(self):
        """Test the optimal batch size selection algorithm."""
        batch_sizes = [2, 5, 10, 15]
        results = self.optimizer.test_batch_sizes(batch_sizes, num_images=20)
        
        optimal_result = self.optimizer.find_optimal_batch_size(results)
        
        assert 'optimal_batch_size' in optimal_result
        assert 'best_result' in optimal_result
        assert 'all_results' in optimal_result
        
        optimal_batch_size = optimal_result['optimal_batch_size']
        assert optimal_batch_size in batch_sizes
        
        # Verify scoring
        best_result = optimal_result['best_result']
        assert 'composite_score' in best_result
        assert 0 <= best_result['composite_score'] <= 1
    
    def test_system_recommendations(self):
        """Test system-based batch size recommendations."""
        recommendations = get_system_recommendations()
        
        assert 'category' in recommendations
        assert 'recommended_range' in recommendations
        assert 'max_safe' in recommendations
        assert 'description' in recommendations
        
        # Verify recommendations are reasonable
        recommended_range = recommendations['recommended_range']
        assert len(recommended_range) > 0
        assert all(isinstance(x, int) and x > 0 for x in recommended_range)
        assert max(recommended_range) <= recommendations['max_safe']
    
    def test_memory_impact_analysis(self):
        """Test analysis of memory impact for different batch sizes."""
        batch_sizes = [1, 5, 10, 20]
        results = self.optimizer.test_batch_sizes(batch_sizes, num_images=20)
        
        # Analyze memory usage patterns
        memory_increases = [r['avg_memory_increase'] for r in results]
        
        # Memory usage should generally increase with batch size, but not exponentially
        for i in range(1, len(memory_increases)):
            current_memory = memory_increases[i]
            previous_memory = memory_increases[i-1]
            
            # Allow for some variation, but flag excessive increases
            if current_memory > previous_memory * 3:
                print(f"⚠️  Large memory increase detected: batch size {batch_sizes[i]} "
                      f"uses {current_memory:.1f} MB vs {previous_memory:.1f} MB for batch size {batch_sizes[i-1]}")
    
    @pytest.mark.integration
    def test_realistic_optimization_workflow(self):
        """Test a realistic optimization workflow."""
        # Get system recommendations
        recommendations = get_system_recommendations()
        batch_sizes = recommendations['recommended_range']
        
        print(f"\n🖥️  System: {recommendations['category']}")
        print(f"📝 Testing: {batch_sizes}")
        
        # Test the recommended batch sizes
        results = self.optimizer.test_batch_sizes(batch_sizes, num_images=25)
        
        # Find optimal
        optimal_result = self.optimizer.find_optimal_batch_size(results)
        optimal_batch_size = optimal_result['optimal_batch_size']
        
        print(f"🎯 Optimal batch size: {optimal_batch_size}")
        
        # Verify the optimal batch size is within safe limits
        assert optimal_batch_size <= recommendations['max_safe']
        assert optimal_batch_size in batch_sizes
        
        # Print detailed results
        best_result = optimal_result['best_result']
        print(f"   Speed: {best_result['avg_speed']:.2f} images/second")
        print(f"   Memory: {best_result['avg_memory_increase']:.1f} MB increase")
        print(f"   Score: {best_result['composite_score']:.3f}")
        
        # Don't return a value in test methods
        assert optimal_batch_size is not None


def test_batch_size_edge_cases():
    """Test edge cases for batch size optimization."""
    optimizer = QuickBatchOptimizer()
    
    # Test very small batch size
    result_small = optimizer.simulate_image_processing(batch_size=1, num_images=5)
    assert result_small['processed_count'] == 5
    assert result_small['batch_size'] == 1
    
    # Test batch size larger than number of images
    result_large = optimizer.simulate_image_processing(batch_size=50, num_images=10)
    assert result_large['processed_count'] == 10
    assert result_large['batch_size'] == 50
    
    # Test empty case
    result_empty = optimizer.simulate_image_processing(batch_size=10, num_images=0)
    assert result_empty['processed_count'] == 0


def benchmark_current_config():
    """Benchmark the current memory configuration."""
    from memory_config import MEMORY_CONFIG
    
    optimizer = QuickBatchOptimizer()
    
    current_batch_size = MEMORY_CONFIG.batch_size
    print(f"\n📊 Benchmarking current configuration (batch size: {current_batch_size})")
    
    # Test current configuration
    result = optimizer.simulate_image_processing(current_batch_size, num_images=30)
    
    print(f"   Speed: {result['images_per_second']:.2f} images/second")
    print(f"   Memory increase: {result['memory_increase_mb']:.1f} MB")
    print(f"   Memory per image: {result['memory_per_image']:.2f} MB")
    
    # Compare with system recommendations
    recommendations = get_system_recommendations()
    recommended_sizes = recommendations['recommended_range']
    
    if current_batch_size not in recommended_sizes:
        print(f"⚠️  Current batch size ({current_batch_size}) not in recommended range: {recommended_sizes}")
        
        # Test a few recommended sizes for comparison
        comparison_sizes = recommended_sizes[:3]  # Test first 3 recommended sizes
        comparison_results = optimizer.test_batch_sizes(comparison_sizes, num_images=30)
        
        print("\n🔍 Comparison with recommended sizes:")
        for comp_result in comparison_results:
            batch_size = comp_result['batch_size']
            speed = comp_result['avg_speed']
            memory = comp_result['avg_memory_increase']
            
            speed_diff = ((speed - result['images_per_second']) / result['images_per_second']) * 100
            memory_diff = ((memory - result['memory_increase_mb']) / max(result['memory_increase_mb'], 0.1)) * 100
            
            print(f"   Batch {batch_size}: {speed:.2f} img/s ({speed_diff:+.1f}%), {memory:.1f} MB ({memory_diff:+.1f}%)")
    
    return result


if __name__ == "__main__":
    print("🚀 Quick Batch Size Optimization")
    print("=" * 40)
    
    # Show system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    print(f"💻 System: {memory_gb:.1f} GB RAM, {cpu_count} CPU cores")
    
    # Get recommendations
    recommendations = get_system_recommendations()
    print(f"📋 Category: {recommendations['category']}")
    print(f"🎯 Recommended range: {recommendations['recommended_range']}")
    
    # Run optimization
    optimizer = QuickBatchOptimizer()
    
    # Test recommended batch sizes
    batch_sizes = recommendations['recommended_range']
    results = optimizer.test_batch_sizes(batch_sizes, num_images=30)
    
    # Find optimal
    optimal_result = optimizer.find_optimal_batch_size(results)
    
    print(f"\n📊 Test Results:")
    print("-" * 25)
    for result in results:
        batch_size = result['batch_size']
        speed = result['avg_speed']
        memory = result['avg_memory_increase']
        print(f"Batch {batch_size:3d}: {speed:6.2f} img/s, {memory:5.1f} MB")
    
    print(f"\n🏆 Optimal Batch Size: {optimal_result['optimal_batch_size']}")
    best = optimal_result['best_result']
    print(f"   Performance: {best['avg_speed']:.2f} images/second")
    print(f"   Memory: {best['avg_memory_increase']:.1f} MB increase")
    print(f"   Overall Score: {best['composite_score']:.3f}")
    
    # Benchmark current config
    print("\n" + "=" * 40)
    benchmark_current_config()
    
    print("\n✅ Quick optimization complete!")