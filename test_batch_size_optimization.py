"""
Tests to find the optimal batch size for image processing operations.

This module provides comprehensive testing to determine the best batch size
for different system configurations and workloads.
"""

import gc
import os
import tempfile
import time
from typing import Dict, List, Tuple, Optional
from unittest.mock import Mock, patch
import pytest
import psutil
import numpy as np
from PIL import Image

from memory_config import MemoryConfig, MEMORY_CONFIG
from imageDuplicate import extract_features, update_faiss_index
from local_media import get_media_files


class BatchSizeOptimizer:
    """Optimizer for finding the best batch size for image processing."""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.test_images: List[str] = []
        
    def create_test_images(self, count: int = 50, temp_dir: Optional[str] = None) -> List[str]:
        """Create test images for batch size optimization."""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            
        test_images = []
        
        for i in range(count):
            # Create images with different sizes and complexities
            if i % 3 == 0:
                size = (512, 512)  # Small
            elif i % 3 == 1:
                size = (1024, 1024)  # Medium
            else:
                size = (2048, 2048)  # Large
                
            # Generate random image data
            image_data = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(image_data)
            
            image_path = os.path.join(temp_dir, f"test_image_{i:03d}.jpg")
            image.save(image_path, "JPEG", quality=85)
            test_images.append(image_path)
            
        self.test_images = test_images
        return test_images
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def test_batch_size_performance(
        self, 
        batch_sizes: List[int], 
        image_count: int = 30,
        iterations: int = 3
    ) -> List[Dict]:
        """Test different batch sizes and measure performance."""
        results = []
        
        # Create test images once
        test_images = self.create_test_images(image_count)
        
        for batch_size in batch_sizes:
            print(f"\n🧪 Testing batch size: {batch_size}")
            
            batch_results = []
            
            for iteration in range(iterations):
                print(f"  Iteration {iteration + 1}/{iterations}")
                
                # Configure memory settings
                test_config = MemoryConfig(
                    batch_size=batch_size,
                    aggressive_gc=True,
                    clear_cache_after_batch=True,
                    max_image_size=(1024, 1024)
                )
                
                # Measure performance
                result = self._run_batch_test(test_images, test_config)
                result['batch_size'] = batch_size
                result['iteration'] = iteration
                batch_results.append(result)
                
                # Clean up between iterations
                gc.collect()
                time.sleep(1)
            
            # Calculate averages for this batch size
            avg_result = self._calculate_averages(batch_results)
            results.append(avg_result)
            
        self.results = results
        return results
    
    def _run_batch_test(self, test_images: List[str], config: MemoryConfig) -> Dict:
        """Run a single batch test and measure metrics."""
        # Apply configuration
        original_config = MEMORY_CONFIG
        
        with patch('memory_config.MEMORY_CONFIG', config):
            start_time = time.time()
            start_memory = self.measure_memory_usage()
            
            # Simulate batch processing
            processed_count = 0
            max_memory_used = start_memory['rss_mb']
            
            batch_size = config.batch_size
            
            for batch_start in range(0, len(test_images), batch_size):
                batch_end = min(batch_start + batch_size, len(test_images))
                batch_files = test_images[batch_start:batch_end]
                
                batch_start_time = time.time()
                
                # Process batch (simulate feature extraction)
                for image_path in batch_files:
                    try:
                        # Load and process image
                        image = Image.open(image_path)
                        if config.max_image_size:
                            max_w, max_h = config.max_image_size
                            if image.size[0] > max_w or image.size[1] > max_h:
                                image = image.resize(config.max_image_size, Image.Resampling.LANCZOS)
                        
                        # Simulate feature extraction (without actual model)
                        features = np.random.rand(768).astype(np.float32)
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                
                # Measure memory after batch
                current_memory = self.measure_memory_usage()
                max_memory_used = max(max_memory_used, current_memory['rss_mb'])
                
                # Simulate garbage collection
                if config.aggressive_gc:
                    gc.collect()
                
                batch_time = time.time() - batch_start_time
            
            end_time = time.time()
            end_memory = self.measure_memory_usage()
            
            return {
                'total_time': end_time - start_time,
                'images_per_second': processed_count / (end_time - start_time),
                'start_memory_mb': start_memory['rss_mb'],
                'end_memory_mb': end_memory['rss_mb'],
                'max_memory_mb': max_memory_used,
                'memory_increase_mb': end_memory['rss_mb'] - start_memory['rss_mb'],
                'processed_count': processed_count,
                'memory_efficiency': processed_count / max(1, max_memory_used - start_memory['rss_mb'])
            }
    
    def _calculate_averages(self, batch_results: List[Dict]) -> Dict:
        """Calculate average metrics from multiple iterations."""
        if not batch_results:
            return {}
        
        avg_result = {'batch_size': batch_results[0]['batch_size']}
        
        numeric_keys = [
            'total_time', 'images_per_second', 'start_memory_mb', 
            'end_memory_mb', 'max_memory_mb', 'memory_increase_mb',
            'processed_count', 'memory_efficiency'
        ]
        
        for key in numeric_keys:
            values = [r[key] for r in batch_results if key in r]
            if values:
                avg_result[f'avg_{key}'] = sum(values) / len(values)
                avg_result[f'min_{key}'] = min(values)
                avg_result[f'max_{key}'] = max(values)
        
        return avg_result
    
    def find_optimal_batch_size(self) -> Dict:
        """Analyze results and find the optimal batch size."""
        if not self.results:
            return {}
        
        # Score each batch size based on multiple criteria
        scored_results = []
        
        for result in self.results:
            # Normalize metrics (higher is better for speed, lower is better for memory)
            speed_score = result.get('avg_images_per_second', 0)
            memory_score = 1000 / max(1, result.get('avg_memory_increase_mb', 1))
            efficiency_score = result.get('avg_memory_efficiency', 0)
            
            # Weighted composite score
            composite_score = (
                speed_score * 0.4 +           # 40% weight on speed
                memory_score * 0.4 +          # 40% weight on memory efficiency
                efficiency_score * 0.2        # 20% weight on overall efficiency
            )
            
            scored_results.append({
                **result,
                'speed_score': speed_score,
                'memory_score': memory_score,
                'efficiency_score': efficiency_score,
                'composite_score': composite_score
            })
        
        # Find the best batch size
        best_result = max(scored_results, key=lambda x: x['composite_score'])
        
        return {
            'optimal_batch_size': best_result['batch_size'],
            'best_result': best_result,
            'all_results': scored_results
        }
    
    def cleanup(self):
        """Clean up test images."""
        for image_path in self.test_images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"Error cleaning up {image_path}: {e}")
        
        # Clean up temp directory if empty
        if self.test_images:
            temp_dir = os.path.dirname(self.test_images[0])
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception:
                pass


@pytest.mark.slow
class TestBatchSizeOptimization:
    """Test suite for batch size optimization."""
    
    def setup_method(self):
        """Set up test environment."""
        self.optimizer = BatchSizeOptimizer()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.optimizer.cleanup()
    
    def test_small_batch_sizes(self):
        """Test small batch sizes (1-10) for memory-constrained systems."""
        batch_sizes = [1, 2, 5, 8, 10]
        results = self.optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=20,
            iterations=2
        )
        
        assert len(results) == len(batch_sizes)
        
        # Verify all batch sizes were tested
        tested_sizes = [r['batch_size'] for r in results]
        assert set(tested_sizes) == set(batch_sizes)
        
        # Check that results contain expected metrics
        for result in results:
            assert 'avg_total_time' in result
            assert 'avg_images_per_second' in result
            assert 'avg_memory_increase_mb' in result
            assert result['avg_processed_count'] > 0
    
    def test_medium_batch_sizes(self):
        """Test medium batch sizes (10-50) for balanced systems."""
        batch_sizes = [10, 15, 20, 25, 30, 40, 50]
        results = self.optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=30,
            iterations=2
        )
        
        assert len(results) == len(batch_sizes)
        
        # Verify performance trends
        for i in range(1, len(results)):
            # Generally, larger batches should be more efficient up to a point
            current = results[i]
            previous = results[i-1]
            
            # At least one metric should improve or stay similar
            speed_improved = current['avg_images_per_second'] >= previous['avg_images_per_second'] * 0.9
            memory_reasonable = current['avg_memory_increase_mb'] <= previous['avg_memory_increase_mb'] * 2
            
            assert speed_improved or memory_reasonable, f"Batch size {current['batch_size']} shows poor performance"
    
    def test_large_batch_sizes(self):
        """Test large batch sizes (50-200) for high-memory systems."""
        batch_sizes = [50, 75, 100, 150, 200]
        results = self.optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=40,
            iterations=2
        )
        
        assert len(results) == len(batch_sizes)
        
        # Check for memory usage patterns
        for result in results:
            # Large batch sizes should process more images
            assert result['avg_processed_count'] > 0
            
            # Memory usage should be reasonable (not exponential growth)
            memory_per_image = result['avg_memory_increase_mb'] / max(1, result['avg_processed_count'])
            assert memory_per_image < 50, f"Memory usage too high: {memory_per_image} MB per image"
    
    def test_find_optimal_batch_size(self):
        """Test the optimal batch size finding algorithm."""
        batch_sizes = [5, 10, 15, 20, 25]
        self.optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=25,
            iterations=2
        )
        
        optimal_result = self.optimizer.find_optimal_batch_size()
        
        assert 'optimal_batch_size' in optimal_result
        assert 'best_result' in optimal_result
        assert 'all_results' in optimal_result
        
        optimal_batch_size = optimal_result['optimal_batch_size']
        assert optimal_batch_size in batch_sizes
        
        # Verify scoring system
        best_result = optimal_result['best_result']
        assert 'composite_score' in best_result
        assert 'speed_score' in best_result
        assert 'memory_score' in best_result
        assert 'efficiency_score' in best_result
    
    def test_memory_constrained_optimization(self):
        """Test optimization for memory-constrained systems."""
        # Simulate low memory system
        batch_sizes = [1, 2, 3, 5, 8]
        
        results = self.optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=15,
            iterations=2
        )
        
        optimal_result = self.optimizer.find_optimal_batch_size()
        optimal_batch_size = optimal_result['optimal_batch_size']
        
        # For memory-constrained systems, optimal should be small
        assert optimal_batch_size <= 8, f"Optimal batch size {optimal_batch_size} too large for memory-constrained system"
        
        # Verify memory usage is reasonable
        best_result = optimal_result['best_result']
        assert best_result['avg_memory_increase_mb'] < 500, "Memory usage too high for constrained system"
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions with different batch sizes."""
        batch_sizes = [5, 10, 20, 50, 100]
        
        results = self.optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=30,
            iterations=2
        )
        
        # Check for performance regressions
        for i in range(1, len(results)):
            current = results[i]
            previous = results[i-1]
            
            # Speed shouldn't degrade significantly
            speed_ratio = current['avg_images_per_second'] / max(0.1, previous['avg_images_per_second'])
            
            # Allow some variation but flag major regressions
            if speed_ratio < 0.5:
                print(f"⚠️  Performance regression detected: batch size {current['batch_size']} "
                      f"is {(1-speed_ratio)*100:.1f}% slower than {previous['batch_size']}")
    
    @pytest.mark.integration
    def test_real_world_batch_optimization(self):
        """Test batch optimization with realistic scenarios."""
        # Test different system configurations
        system_configs = [
            {"name": "Low Memory", "batch_sizes": [1, 2, 5, 8], "image_count": 15},
            {"name": "Balanced", "batch_sizes": [5, 10, 15, 20], "image_count": 25},
            {"name": "High Memory", "batch_sizes": [10, 20, 30, 50], "image_count": 35}
        ]
        
        recommendations = {}
        
        for config in system_configs:
            print(f"\n🔧 Testing {config['name']} configuration")
            
            results = self.optimizer.test_batch_size_performance(
                batch_sizes=config['batch_sizes'],
                image_count=config['image_count'],
                iterations=2
            )
            
            optimal_result = self.optimizer.find_optimal_batch_size()
            recommendations[config['name']] = optimal_result['optimal_batch_size']
            
            print(f"   Recommended batch size: {optimal_result['optimal_batch_size']}")
        
        # Verify recommendations make sense
        assert recommendations["Low Memory"] <= recommendations["Balanced"]
        assert recommendations["Balanced"] <= recommendations["High Memory"]
        
        return recommendations


def test_batch_size_configuration_validation():
    """Test validation of batch size configurations."""
    
    # Test valid configurations
    valid_configs = [1, 5, 10, 20, 50, 100]
    for batch_size in valid_configs:
        config = MemoryConfig(batch_size=batch_size)
        assert config.batch_size == batch_size
    
    # Test edge cases
    edge_cases = [1, 1000]  # Very small and very large
    for batch_size in edge_cases:
        config = MemoryConfig(batch_size=batch_size)
        assert config.batch_size == batch_size


def test_batch_processing_memory_impact():
    """Test memory impact of different batch sizes."""
    
    def simulate_batch_processing(batch_size: int, total_items: int = 100) -> Dict:
        """Simulate batch processing and measure memory."""
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Simulate processing batches
        max_memory = start_memory
        
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = list(range(batch_start, batch_end))
            
            # Simulate memory usage during batch processing
            # Create some temporary data structures
            temp_data = [np.random.rand(100) for _ in batch_items]
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, current_memory)
            
            # Simulate cleanup
            del temp_data
            gc.collect()
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            'start_memory': start_memory,
            'max_memory': max_memory,
            'end_memory': end_memory,
            'peak_increase': max_memory - start_memory
        }
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10, 20, 50]
    results = {}
    
    for batch_size in batch_sizes:
        results[batch_size] = simulate_batch_processing(batch_size)
    
    # Verify memory usage patterns
    for batch_size, result in results.items():
        assert result['peak_increase'] >= 0, f"Memory should increase during processing"
        assert result['peak_increase'] < 1000, f"Memory increase too high: {result['peak_increase']} MB"
    
    return results


if __name__ == "__main__":
    # Run optimization when script is executed directly
    print("🚀 Starting Batch Size Optimization Tests")
    print("=" * 50)
    
    optimizer = BatchSizeOptimizer()
    
    try:
        # Test different ranges based on system memory
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System Memory: {system_memory_gb:.1f} GB")
        
        if system_memory_gb < 8:
            batch_sizes = [1, 2, 5, 8, 10]
            image_count = 20
        elif system_memory_gb < 16:
            batch_sizes = [5, 10, 15, 20, 25, 30]
            image_count = 30
        else:
            batch_sizes = [10, 20, 30, 50, 75, 100]
            image_count = 40
        
        print(f"Testing batch sizes: {batch_sizes}")
        print(f"Using {image_count} test images")
        
        # Run optimization
        results = optimizer.test_batch_size_performance(
            batch_sizes=batch_sizes,
            image_count=image_count,
            iterations=3
        )
        
        # Find optimal
        optimal_result = optimizer.find_optimal_batch_size()
        
        print("\n📊 Results Summary:")
        print("-" * 30)
        
        for result in results:
            batch_size = result['batch_size']
            speed = result['avg_images_per_second']
            memory = result['avg_memory_increase_mb']
            
            print(f"Batch Size {batch_size:3d}: {speed:.2f} img/s, {memory:.1f} MB memory")
        
        print(f"\n🎯 Optimal Batch Size: {optimal_result['optimal_batch_size']}")
        
        best = optimal_result['best_result']
        print(f"   Speed: {best['avg_images_per_second']:.2f} images/second")
        print(f"   Memory: {best['avg_memory_increase_mb']:.1f} MB increase")
        print(f"   Score: {best['composite_score']:.2f}")
        
    finally:
        optimizer.cleanup()
        print("\n✅ Optimization complete!")