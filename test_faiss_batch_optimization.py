"""
FAISS-specific batch size optimization tests.

This module tests batch sizes specifically for FAISS index operations
and feature extraction to find optimal performance for the actual workload.
"""

import gc
import os
import tempfile
import time
from typing import Dict, List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock
import pytest
import psutil
import numpy as np
from PIL import Image

from memory_config import MemoryConfig


class FAISSBatchOptimizer:
    """Optimizer specifically for FAISS operations."""

    def __init__(self):
        self.temp_dir = None
        self.test_images = []
        self.baseline_memory = self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def create_test_images(self, count: int = 20) -> List[str]:
        """Create test images for FAISS testing."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()

        test_images = []

        for i in range(count):
            # Create varied image sizes
            if i % 4 == 0:
                size = (256, 256)
            elif i % 4 == 1:
                size = (512, 512)
            elif i % 4 == 2:
                size = (1024, 1024)
            else:
                size = (800, 600)

            # Create realistic image data
            image_data = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(image_data)

            image_path = os.path.join(self.temp_dir, f"test_img_{i:03d}.jpg")
            image.save(image_path, "JPEG", quality=85)
            test_images.append(image_path)

        self.test_images = test_images
        return test_images

    def simulate_feature_extraction_batch(
        self, image_paths: List[str], config: MemoryConfig
    ) -> Dict:
        """Simulate feature extraction for a batch of images."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        max_memory = start_memory

        features_list = []
        processed_count = 0

        try:
            for image_path in image_paths:
                # Simulate image loading and preprocessing
                image = Image.open(image_path)

                # Apply image size limits if configured
                if config.max_image_size:
                    max_w, max_h = config.max_image_size
                    if image.size[0] > max_w or image.size[1] > max_h:
                        image = image.resize(
                            config.max_image_size, Image.Resampling.LANCZOS
                        )

                # Simulate feature extraction (ViT-B/16 produces 768-dim features)
                features = np.random.rand(768).astype(np.float32)
                features_list.append(features)
                processed_count += 1

                # Check memory usage
                current_memory = self._get_memory_usage()
                max_memory = max(max_memory, current_memory)

            # Simulate FAISS operations
            if features_list:
                # Convert to numpy array (as done in real code)
                features_array = np.array(features_list, dtype=np.float32)

                # Simulate FAISS index operations
                # This mimics the memory usage of actual FAISS operations
                temp_index_data = features_array.copy()

                current_memory = self._get_memory_usage()
                max_memory = max(max_memory, current_memory)

                # Cleanup
                del temp_index_data
                del features_array

        except Exception as e:
            print(f"Error in batch processing: {e}")

        finally:
            # Cleanup
            del features_list
            if config.aggressive_gc:
                gc.collect()

        end_time = time.time()
        end_memory = self._get_memory_usage()

        return {
            "processing_time": end_time - start_time,
            "images_per_second": processed_count / max(0.001, end_time - start_time),
            "start_memory_mb": start_memory,
            "max_memory_mb": max_memory,
            "end_memory_mb": end_memory,
            "memory_increase_mb": max_memory - start_memory,
            "processed_count": processed_count,
            "memory_per_image": (max_memory - start_memory) / max(1, processed_count),
        }

    def test_faiss_batch_sizes(
        self, batch_sizes: List[int], total_images: int = 30
    ) -> List[Dict]:
        """Test FAISS operations with different batch sizes."""
        # Create test images
        test_images = self.create_test_images(total_images)
        results = []

        print(f"🔬 Testing FAISS batch sizes: {batch_sizes}")
        print(f"📸 Using {len(test_images)} test images")

        for batch_size in batch_sizes:
            print(f"   Testing batch size {batch_size}...")

            # Test configuration
            config = MemoryConfig(
                batch_size=batch_size,
                aggressive_gc=True,
                clear_cache_after_batch=True,
                max_image_size=(1024, 1024),
            )

            # Run multiple iterations
            batch_results = []

            for iteration in range(3):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                max_memory = start_memory
                total_processed = 0

                # Process in batches
                for batch_start in range(0, len(test_images), batch_size):
                    batch_end = min(batch_start + batch_size, len(test_images))
                    batch_images = test_images[batch_start:batch_end]

                    batch_result = self.simulate_feature_extraction_batch(
                        batch_images, config
                    )

                    max_memory = max(max_memory, batch_result["max_memory_mb"])
                    total_processed += batch_result["processed_count"]

                end_time = time.time()
                end_memory = self._get_memory_usage()

                iteration_result = {
                    "batch_size": batch_size,
                    "total_time": end_time - start_time,
                    "images_per_second": total_processed
                    / max(0.001, end_time - start_time),
                    "max_memory_increase": max_memory - start_memory,
                    "end_memory_increase": end_memory - start_memory,
                    "processed_count": total_processed,
                }

                batch_results.append(iteration_result)

                # Brief pause between iterations
                time.sleep(0.2)

            # Calculate averages
            avg_result = {
                "batch_size": batch_size,
                "avg_time": sum(r["total_time"] for r in batch_results)
                / len(batch_results),
                "avg_speed": sum(r["images_per_second"] for r in batch_results)
                / len(batch_results),
                "avg_max_memory": sum(r["max_memory_increase"] for r in batch_results)
                / len(batch_results),
                "avg_end_memory": sum(r["end_memory_increase"] for r in batch_results)
                / len(batch_results),
                "max_memory_peak": max(r["max_memory_increase"] for r in batch_results),
                "min_speed": min(r["images_per_second"] for r in batch_results),
                "memory_efficiency": sum(r["processed_count"] for r in batch_results)
                / sum(
                    r["max_memory_increase"]
                    for r in batch_results
                    if r["max_memory_increase"] > 0
                ),
            }

            results.append(avg_result)

        return results

    def test_faiss_search_batch_sizes(
        self, batch_sizes: List[int], num_vectors: int = 100
    ) -> List[Dict]:
        """Test FAISS search operations with different batch sizes."""
        results = []

        # Create mock FAISS index data
        dimension = 768
        mock_vectors = np.random.rand(num_vectors, dimension).astype(np.float32)

        print(f"🔍 Testing FAISS search batch sizes: {batch_sizes}")
        print(f"📊 Using {num_vectors} vectors for search simulation")

        for batch_size in batch_sizes:
            print(f"   Testing search batch size {batch_size}...")

            batch_results = []

            for iteration in range(3):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                max_memory = start_memory
                searches_performed = 0

                # Simulate search operations in batches
                for batch_start in range(0, num_vectors, batch_size):
                    batch_end = min(batch_start + batch_size, num_vectors)
                    batch_vectors = mock_vectors[batch_start:batch_end]

                    # Simulate FAISS search operations
                    for vector in batch_vectors:
                        # Simulate distance calculations
                        distances = np.linalg.norm(mock_vectors - vector, axis=1)
                        # Simulate finding nearest neighbors
                        nearest_indices = np.argsort(distances)[:5]
                        searches_performed += 1

                        # Check memory
                        current_memory = self._get_memory_usage()
                        max_memory = max(max_memory, current_memory)

                    # Cleanup after batch
                    del batch_vectors
                    gc.collect()

                end_time = time.time()
                end_memory = self._get_memory_usage()

                iteration_result = {
                    "batch_size": batch_size,
                    "total_time": end_time - start_time,
                    "searches_per_second": searches_performed
                    / max(0.001, end_time - start_time),
                    "max_memory_increase": max_memory - start_memory,
                    "end_memory_increase": end_memory - start_memory,
                    "searches_performed": searches_performed,
                }

                batch_results.append(iteration_result)
                time.sleep(0.1)

            # Calculate averages
            avg_result = {
                "batch_size": batch_size,
                "avg_time": sum(r["total_time"] for r in batch_results)
                / len(batch_results),
                "avg_search_speed": sum(r["searches_per_second"] for r in batch_results)
                / len(batch_results),
                "avg_max_memory": sum(r["max_memory_increase"] for r in batch_results)
                / len(batch_results),
                "max_memory_peak": max(r["max_memory_increase"] for r in batch_results),
                "min_search_speed": min(
                    r["searches_per_second"] for r in batch_results
                ),
            }

            results.append(avg_result)

        return results

    def find_optimal_faiss_batch_size(
        self, extraction_results: List[Dict], search_results: List[Dict]
    ) -> Dict:
        """Find optimal batch size considering both extraction and search performance."""

        # Combine results by batch size
        combined_results = {}

        for result in extraction_results:
            batch_size = result["batch_size"]
            combined_results[batch_size] = {
                "batch_size": batch_size,
                "extraction_speed": result["avg_speed"],
                "extraction_memory": result["avg_max_memory"],
                "extraction_efficiency": result.get("memory_efficiency", 0),
            }

        for result in search_results:
            batch_size = result["batch_size"]
            if batch_size in combined_results:
                combined_results[batch_size].update(
                    {
                        "search_speed": result["avg_search_speed"],
                        "search_memory": result["avg_max_memory"],
                    }
                )

        # Score each batch size
        scored_results = []

        for batch_size, metrics in combined_results.items():
            # Normalize scores
            extraction_speed_score = metrics.get("extraction_speed", 0) / max(
                1, max(m.get("extraction_speed", 1) for m in combined_results.values())
            )
            search_speed_score = metrics.get("search_speed", 0) / max(
                1, max(m.get("search_speed", 1) for m in combined_results.values())
            )

            # Memory efficiency (lower memory usage is better)
            extraction_memory = metrics.get("extraction_memory", 1)
            search_memory = metrics.get("search_memory", 1)
            total_memory = extraction_memory + search_memory

            memory_score = 100 / max(1, total_memory)  # Inverse relationship

            # Composite score
            composite_score = (
                extraction_speed_score * 0.4  # 40% extraction speed
                + search_speed_score * 0.3  # 30% search speed
                + memory_score * 0.3  # 30% memory efficiency
            )

            scored_result = {
                **metrics,
                "extraction_speed_score": extraction_speed_score,
                "search_speed_score": search_speed_score,
                "memory_score": memory_score,
                "composite_score": composite_score,
            }

            scored_results.append(scored_result)

        # Find best result
        if scored_results:
            best_result = max(scored_results, key=lambda x: x["composite_score"])

            return {
                "optimal_batch_size": best_result["batch_size"],
                "best_result": best_result,
                "all_results": scored_results,
            }

        return {}

    def cleanup(self):
        """Clean up test files."""
        for image_path in self.test_images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception:
                pass

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except Exception:
                pass


@pytest.mark.slow
class TestFAISSBatchOptimization:
    """Test suite for FAISS-specific batch optimization."""

    def setup_method(self):
        """Set up test environment."""
        self.optimizer = FAISSBatchOptimizer()

    def teardown_method(self):
        """Clean up after tests."""
        self.optimizer.cleanup()

    def test_faiss_feature_extraction_batches(self):
        """Test feature extraction with different batch sizes."""
        batch_sizes = [1, 5, 10, 15]
        results = self.optimizer.test_faiss_batch_sizes(batch_sizes, total_images=20)

        assert len(results) == len(batch_sizes)

        for result in results:
            assert result["avg_speed"] > 0
            assert result["avg_max_memory"] >= 0
            assert result["batch_size"] in batch_sizes

    def test_faiss_search_batches(self):
        """Test FAISS search operations with different batch sizes."""
        batch_sizes = [5, 10, 20, 50]
        results = self.optimizer.test_faiss_search_batch_sizes(
            batch_sizes, num_vectors=50
        )

        assert len(results) == len(batch_sizes)

        for result in results:
            assert result["avg_search_speed"] > 0
            assert result["avg_max_memory"] >= 0
            assert result["batch_size"] in batch_sizes

    def test_combined_faiss_optimization(self):
        """Test combined optimization for both extraction and search."""
        batch_sizes = [5, 10, 15, 20]

        # Test both operations
        extraction_results = self.optimizer.test_faiss_batch_sizes(
            batch_sizes, total_images=25
        )
        search_results = self.optimizer.test_faiss_search_batch_sizes(
            batch_sizes, num_vectors=60
        )

        # Find optimal
        optimal_result = self.optimizer.find_optimal_faiss_batch_size(
            extraction_results, search_results
        )

        assert "optimal_batch_size" in optimal_result
        assert "best_result" in optimal_result
        assert optimal_result["optimal_batch_size"] in batch_sizes

    def test_memory_constrained_faiss_optimization(self):
        """Test FAISS optimization for memory-constrained systems."""
        # Use smaller batch sizes for memory-constrained testing
        batch_sizes = [1, 2, 5, 8]

        extraction_results = self.optimizer.test_faiss_batch_sizes(
            batch_sizes, total_images=15
        )
        search_results = self.optimizer.test_faiss_search_batch_sizes(
            batch_sizes, num_vectors=30
        )

        optimal_result = self.optimizer.find_optimal_faiss_batch_size(
            extraction_results, search_results
        )

        # For memory-constrained systems, optimal should be small
        assert optimal_result["optimal_batch_size"] <= 10

        # Verify memory usage is reasonable
        best_result = optimal_result["best_result"]
        total_memory = best_result.get("extraction_memory", 0) + best_result.get(
            "search_memory", 0
        )
        assert total_memory < 200, f"Total memory usage too high: {total_memory} MB"

    @pytest.mark.integration
    def test_realistic_faiss_workflow(self):
        """Test a realistic FAISS workflow optimization."""
        # Get system memory to determine appropriate batch sizes
        system_memory_gb = psutil.virtual_memory().total / (1024**3)

        if system_memory_gb < 8:
            batch_sizes = [2, 5, 8, 10]
            image_count = 20
            vector_count = 40
        elif system_memory_gb < 16:
            batch_sizes = [5, 10, 15, 20, 25]
            image_count = 30
            vector_count = 60
        else:
            batch_sizes = [10, 20, 30, 50]
            image_count = 40
            vector_count = 80

        print(f"\n🖥️  System Memory: {system_memory_gb:.1f} GB")
        print(f"🧪 Testing batch sizes: {batch_sizes}")

        # Test both operations
        extraction_results = self.optimizer.test_faiss_batch_sizes(
            batch_sizes, total_images=image_count
        )
        search_results = self.optimizer.test_faiss_search_batch_sizes(
            batch_sizes, num_vectors=vector_count
        )

        # Find optimal
        optimal_result = self.optimizer.find_optimal_faiss_batch_size(
            extraction_results, search_results
        )

        optimal_batch_size = optimal_result["optimal_batch_size"]
        best_result = optimal_result["best_result"]

        print(f"\n🎯 Optimal FAISS batch size: {optimal_batch_size}")
        print(
            f"   Extraction speed: {best_result.get('extraction_speed', 0):.2f} img/s"
        )
        print(f"   Search speed: {best_result.get('search_speed', 0):.2f} searches/s")
        print(
            f"   Total memory: {best_result.get('extraction_memory', 0) + best_result.get('search_memory', 0):.1f} MB"
        )
        print(f"   Overall score: {best_result['composite_score']:.3f}")

        return optimal_batch_size


def test_faiss_batch_size_validation():
    """Test validation of FAISS-specific batch size configurations."""

    # Test that batch sizes work with FAISS operations
    valid_batch_sizes = [1, 5, 10, 20, 50, 100]

    for batch_size in valid_batch_sizes:
        config = MemoryConfig(batch_size=batch_size)

        # Simulate basic FAISS operations
        features = np.random.rand(batch_size, 768).astype(np.float32)

        # Should not raise exceptions
        assert features.shape[0] == batch_size
        assert features.shape[1] == 768


def test_faiss_memory_scaling():
    """Test how FAISS memory usage scales with batch size."""

    def estimate_faiss_memory(batch_size: int, feature_dim: int = 768) -> float:
        """Estimate memory usage for FAISS operations."""
        # Feature storage: batch_size * feature_dim * 4 bytes (float32)
        feature_memory = batch_size * feature_dim * 4 / (1024 * 1024)  # MB

        # Index overhead (estimated)
        index_overhead = batch_size * 0.1  # MB

        # Processing overhead
        processing_overhead = batch_size * 0.05  # MB

        return feature_memory + index_overhead + processing_overhead

    batch_sizes = [1, 5, 10, 20, 50, 100]
    memory_estimates = []

    for batch_size in batch_sizes:
        memory_mb = estimate_faiss_memory(batch_size)
        memory_estimates.append(memory_mb)

        print(f"Batch size {batch_size:3d}: ~{memory_mb:.1f} MB estimated")

    # Verify scaling is reasonable (should be roughly linear)
    for i in range(1, len(batch_sizes)):
        current_batch = batch_sizes[i]
        previous_batch = batch_sizes[i - 1]

        current_memory = memory_estimates[i]
        previous_memory = memory_estimates[i - 1]

        # Memory should scale roughly linearly with batch size
        expected_ratio = current_batch / previous_batch
        actual_ratio = current_memory / previous_memory

        # Allow some variation but should be roughly proportional
        assert (
            0.5 < actual_ratio / expected_ratio < 2.0
        ), f"Memory scaling issue at batch size {current_batch}"


if __name__ == "__main__":
    print("🔬 FAISS Batch Size Optimization")
    print("=" * 40)

    # Show system info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💻 System Memory: {memory_gb:.1f} GB")

    optimizer = FAISSBatchOptimizer()

    try:
        # Determine test parameters based on system memory
        if memory_gb < 8:
            batch_sizes = [1, 2, 5, 8, 10]
            image_count = 20
            vector_count = 40
        elif memory_gb < 16:
            batch_sizes = [5, 10, 15, 20, 25]
            image_count = 30
            vector_count = 60
        else:
            batch_sizes = [10, 20, 30, 50, 75]
            image_count = 40
            vector_count = 80

        print(f"🧪 Testing batch sizes: {batch_sizes}")

        # Test feature extraction
        print("\n📸 Testing Feature Extraction...")
        extraction_results = optimizer.test_faiss_batch_sizes(
            batch_sizes, total_images=image_count
        )

        # Test search operations
        print("\n🔍 Testing Search Operations...")
        search_results = optimizer.test_faiss_search_batch_sizes(
            batch_sizes, num_vectors=vector_count
        )

        # Find optimal
        optimal_result = optimizer.find_optimal_faiss_batch_size(
            extraction_results, search_results
        )

        print(f"\n📊 Feature Extraction Results:")
        print("-" * 35)
        for result in extraction_results:
            batch_size = result["batch_size"]
            speed = result["avg_speed"]
            memory = result["avg_max_memory"]
            print(f"Batch {batch_size:3d}: {speed:6.2f} img/s, {memory:5.1f} MB")

        print(f"\n🔍 Search Results:")
        print("-" * 25)
        for result in search_results:
            batch_size = result["batch_size"]
            speed = result["avg_search_speed"]
            memory = result["avg_max_memory"]
            print(f"Batch {batch_size:3d}: {speed:6.2f} search/s, {memory:5.1f} MB")

        print(f"\n🏆 Optimal FAISS Batch Size: {optimal_result['optimal_batch_size']}")
        best = optimal_result["best_result"]
        print(f"   Extraction: {best.get('extraction_speed', 0):.2f} images/second")
        print(f"   Search: {best.get('search_speed', 0):.2f} searches/second")
        print(
            f"   Memory: {best.get('extraction_memory', 0) + best.get('search_memory', 0):.1f} MB total"
        )
        print(f"   Score: {best['composite_score']:.3f}")

    finally:
        optimizer.cleanup()
        print("\n✅ FAISS optimization complete!")
