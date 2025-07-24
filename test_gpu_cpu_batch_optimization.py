"""
GPU and CPU batch size optimization tests.

This module tests actual PyTorch model inference with different batch sizes
on both CPU and GPU to find optimal configurations for real workloads.
"""

import gc
import os
import tempfile
import time
from typing import Dict, List, Tuple, Optional
import pytest
import psutil
import numpy as np
from PIL import Image

from memory_config import MemoryConfig


class GPUCPUBatchOptimizer:
    """Optimizer for testing actual GPU and CPU batch sizes."""
    
    def __init__(self):
        self.baseline_memory = self._get_memory_usage()
        self.temp_dir = None
        self.test_images = []
        self._torch_components = None
        self._model_components = None
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        gpu_memory = 0
        gpu_available = False
        
        try:
            if self._torch_components is None:
                self._load_torch_components()
            
            torch = self._torch_components['torch']
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_available = True
        except Exception:
            pass
        
        return {
            'cpu_mb': cpu_memory,
            'gpu_mb': gpu_memory,
            'gpu_available': gpu_available
        }
    
    def _load_torch_components(self):
        """Load PyTorch components lazily."""
        if self._torch_components is None:
            try:
                import torch
                from torchvision.models import ViT_B_16_Weights, vit_b_16
                from torchvision.transforms import Compose, Resize, ToTensor, Normalize
                
                self._torch_components = {
                    'torch': torch,
                    'ViT_B_16_Weights': ViT_B_16_Weights,
                    'vit_b_16': vit_b_16,
                    'Compose': Compose,
                    'Resize': Resize,
                    'ToTensor': ToTensor,
                    'Normalize': Normalize
                }
                print("✅ PyTorch components loaded successfully")
            except ImportError as e:
                print(f"❌ Failed to load PyTorch: {e}")
                self._torch_components = {}
    
    def _load_model_components(self, device: str):
        """Load model components for the specified device."""
        if self._model_components is None or self._model_components.get('device') != device:
            self._load_torch_components()
            
            if not self._torch_components:
                return None
            
            try:
                torch = self._torch_components['torch']
                ViT_B_16_Weights = self._torch_components['ViT_B_16_Weights']
                vit_b_16 = self._torch_components['vit_b_16']
                Compose = self._torch_components['Compose']
                Resize = self._torch_components['Resize']
                ToTensor = self._torch_components['ToTensor']
                Normalize = self._torch_components['Normalize']
                
                # Set device
                if device == 'cuda' and not torch.cuda.is_available():
                    print("⚠️  CUDA requested but not available, falling back to CPU")
                    device = 'cpu'
                
                device_obj = torch.device(device)
                
                # Load model
                weights = ViT_B_16_Weights.DEFAULT
                model = vit_b_16(weights=weights)
                model.to(device_obj)
                model.eval()
                
                # Create transform
                transform = Compose([
                    Resize((224, 224)),  # ViT input size
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                self._model_components = {
                    'model': model,
                    'transform': transform,
                    'device': device,
                    'device_obj': device_obj,
                    'torch': torch
                }
                
                print(f"✅ Model loaded on {device.upper()}")
                return self._model_components
                
            except Exception as e:
                print(f"❌ Failed to load model on {device}: {e}")
                return None
        
        return self._model_components
    
    def create_test_images(self, count: int = 20) -> List[str]:
        """Create test images for real model testing."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        
        test_images = []
        
        for i in range(count):
            # Create varied image sizes (realistic for photo collections)
            sizes = [(640, 480), (1024, 768), (1920, 1080), (2048, 1536)]
            size = sizes[i % len(sizes)]
            
            # Create realistic image data
            image_data = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(image_data)
            
            image_path = os.path.join(self.temp_dir, f"test_img_{i:03d}.jpg")
            image.save(image_path, "JPEG", quality=85)
            test_images.append(image_path)
        
        self.test_images = test_images
        return test_images
    
    def test_real_batch_processing(
        self, 
        batch_size: int, 
        device: str, 
        image_paths: List[str],
        config: MemoryConfig
    ) -> Dict:
        """Test real model inference with actual batch processing."""
        
        # Load model components
        components = self._load_model_components(device)
        if not components:
            return {
                'error': f'Failed to load model on {device}',
                'batch_size': batch_size,
                'device': device
            }
        
        model = components['model']
        transform = components['transform']
        device_obj = components['device_obj']
        torch = components['torch']
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        max_cpu_memory = start_memory['cpu_mb']
        max_gpu_memory = start_memory['gpu_mb']
        
        processed_count = 0
        total_inference_time = 0
        
        try:
            # Process images in batches
            for batch_start in range(0, len(image_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(image_paths))
                batch_paths = image_paths[batch_start:batch_end]
                
                # Load and preprocess batch
                batch_tensors = []
                for image_path in batch_paths:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        
                        # Apply image size limits if configured
                        if config.max_image_size:
                            max_w, max_h = config.max_image_size
                            if image.size[0] > max_w or image.size[1] > max_h:
                                image = image.resize(config.max_image_size, Image.Resampling.LANCZOS)
                        
                        tensor = transform(image)
                        batch_tensors.append(tensor)
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        continue
                
                if batch_tensors:
                    # Create batch tensor
                    batch_tensor = torch.stack(batch_tensors).to(device_obj)
                    
                    # Measure inference time
                    inference_start = time.time()
                    
                    with torch.no_grad():
                        features = model(batch_tensor)
                    
                    # Synchronize GPU operations
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time
                    
                    # Check memory usage
                    current_memory = self._get_memory_usage()
                    max_cpu_memory = max(max_cpu_memory, current_memory['cpu_mb'])
                    max_gpu_memory = max(max_gpu_memory, current_memory['gpu_mb'])
                    
                    # Cleanup
                    del batch_tensor, features
                    
                    # Clear cache if configured
                    if config.clear_cache_after_batch:
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()
                
                del batch_tensors
                
                # Force garbage collection if configured
                if config.aggressive_gc:
                    gc.collect()
        
        except Exception as e:
            return {
                'error': str(e),
                'batch_size': batch_size,
                'device': device,
                'processed_count': processed_count
            }
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        total_time = end_time - start_time
        
        return {
            'batch_size': batch_size,
            'device': device,
            'total_time': total_time,
            'inference_time': total_inference_time,
            'images_per_second': processed_count / max(0.001, total_time),
            'inference_images_per_second': processed_count / max(0.001, total_inference_time),
            'start_cpu_memory': start_memory['cpu_mb'],
            'max_cpu_memory': max_cpu_memory,
            'cpu_memory_increase': max_cpu_memory - start_memory['cpu_mb'],
            'start_gpu_memory': start_memory['gpu_mb'],
            'max_gpu_memory': max_gpu_memory,
            'gpu_memory_increase': max_gpu_memory - start_memory['gpu_mb'],
            'processed_count': processed_count,
            'gpu_available': start_memory['gpu_available']
        }
    
    def test_device_batch_sizes(
        self, 
        device: str, 
        batch_sizes: List[int], 
        image_count: int = 20
    ) -> List[Dict]:
        """Test different batch sizes on a specific device."""
        
        # Create test images
        test_images = self.create_test_images(image_count)
        results = []
        
        print(f"\n🔬 Testing {device.upper()} batch sizes: {batch_sizes}")
        print(f"📸 Using {len(test_images)} test images")
        
        for batch_size in batch_sizes:
            print(f"   Testing batch size {batch_size} on {device.upper()}...")
            
            # Test configuration
            config = MemoryConfig(
                batch_size=batch_size,
                use_cpu_only=(device == 'cpu'),
                aggressive_gc=True,
                clear_cache_after_batch=True,
                max_image_size=(1024, 1024)
            )
            
            # Run multiple iterations
            batch_results = []
            
            for iteration in range(2):  # Fewer iterations for real model testing
                result = self.test_real_batch_processing(batch_size, device, test_images, config)
                
                if 'error' not in result:
                    batch_results.append(result)
                else:
                    print(f"   ⚠️  Error in iteration {iteration + 1}: {result['error']}")
                
                # Brief pause between iterations
                time.sleep(0.5)
            
            if batch_results:
                # Calculate averages
                avg_result = {
                    'batch_size': batch_size,
                    'device': device,
                    'avg_total_time': sum(r['total_time'] for r in batch_results) / len(batch_results),
                    'avg_inference_time': sum(r['inference_time'] for r in batch_results) / len(batch_results),
                    'avg_images_per_second': sum(r['images_per_second'] for r in batch_results) / len(batch_results),
                    'avg_inference_ips': sum(r['inference_images_per_second'] for r in batch_results) / len(batch_results),
                    'avg_cpu_memory': sum(r['cpu_memory_increase'] for r in batch_results) / len(batch_results),
                    'avg_gpu_memory': sum(r['gpu_memory_increase'] for r in batch_results) / len(batch_results),
                    'max_cpu_memory': max(r['cpu_memory_increase'] for r in batch_results),
                    'max_gpu_memory': max(r['gpu_memory_increase'] for r in batch_results),
                    'min_images_per_second': min(r['images_per_second'] for r in batch_results),
                    'processed_count': batch_results[0]['processed_count'],
                    'gpu_available': batch_results[0]['gpu_available']
                }
                
                results.append(avg_result)
            else:
                print(f"   ❌ All iterations failed for batch size {batch_size}")
        
        return results
    
    def compare_cpu_gpu_performance(self, batch_sizes: List[int], image_count: int = 25) -> Dict:
        """Compare CPU vs GPU performance for different batch sizes."""
        
        results = {
            'cpu_results': [],
            'gpu_results': [],
            'comparison': []
        }
        
        # Test CPU
        print("🖥️  Testing CPU performance...")
        cpu_results = self.test_device_batch_sizes('cpu', batch_sizes, image_count)
        results['cpu_results'] = cpu_results
        
        # Test GPU if available
        if self._torch_components is None:
            self._load_torch_components()
        
        if self._torch_components and self._torch_components['torch'].cuda.is_available():
            print("🚀 Testing GPU performance...")
            gpu_results = self.test_device_batch_sizes('cuda', batch_sizes, image_count)
            results['gpu_results'] = gpu_results
            
            # Create comparison
            for batch_size in batch_sizes:
                cpu_result = next((r for r in cpu_results if r['batch_size'] == batch_size), None)
                gpu_result = next((r for r in gpu_results if r['batch_size'] == batch_size), None)
                
                if cpu_result and gpu_result:
                    speedup = gpu_result['avg_images_per_second'] / max(cpu_result['avg_images_per_second'], 0.1)
                    gpu_memory_overhead = gpu_result['avg_gpu_memory']
                    
                    results['comparison'].append({
                        'batch_size': batch_size,
                        'cpu_speed': cpu_result['avg_images_per_second'],
                        'gpu_speed': gpu_result['avg_images_per_second'],
                        'speedup': speedup,
                        'cpu_memory': cpu_result['avg_cpu_memory'],
                        'gpu_memory': gpu_memory_overhead,
                        'total_memory': cpu_result['avg_cpu_memory'] + gpu_memory_overhead
                    })
        else:
            print("⚠️  GPU not available, skipping GPU tests")
        
        return results
    
    def find_optimal_device_batch_size(self, comparison_results: Dict) -> Dict:
        """Find optimal batch size considering both CPU and GPU performance."""
        
        if not comparison_results['comparison']:
            # Only CPU results available
            cpu_results = comparison_results['cpu_results']
            if cpu_results:
                best_cpu = max(cpu_results, key=lambda x: x['avg_images_per_second'])
                return {
                    'optimal_device': 'cpu',
                    'optimal_batch_size': best_cpu['batch_size'],
                    'best_result': best_cpu,
                    'reason': 'GPU not available'
                }
            return {}
        
        # Get system memory to adjust scoring
        import psutil
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Score each configuration
        scored_results = []
        
        for comp in comparison_results['comparison']:
            batch_size = comp['batch_size']
            
            # Performance scores (normalized)
            max_cpu_speed = max(c['cpu_speed'] for c in comparison_results['comparison'])
            max_gpu_speed = max(c['gpu_speed'] for c in comparison_results['comparison'])
            
            cpu_speed_score = comp['cpu_speed'] / max_cpu_speed
            gpu_speed_score = comp['gpu_speed'] / max_gpu_speed
            
            # Memory efficiency scores (lower memory usage is better)
            max_total_memory = max(c['total_memory'] for c in comparison_results['comparison'])
            memory_efficiency = 1.0 - (comp['total_memory'] / max_total_memory)
            
            # GPU advantage score
            gpu_advantage = max(0, comp['speedup'] - 1)  # How much faster GPU is
            
            # Adjust scoring based on system memory
            if system_memory_gb >= 16:
                # High memory system - prioritize performance
                cpu_score = cpu_speed_score * 0.8 + memory_efficiency * 0.2
                gpu_score = gpu_speed_score * 0.6 + gpu_advantage * 0.3 + memory_efficiency * 0.1
            elif system_memory_gb >= 8:
                # Medium memory system - balance performance and memory
                cpu_score = cpu_speed_score * 0.7 + memory_efficiency * 0.3
                gpu_score = gpu_speed_score * 0.5 + gpu_advantage * 0.3 + memory_efficiency * 0.2
            else:
                # Low memory system - prioritize memory efficiency
                cpu_score = cpu_speed_score * 0.5 + memory_efficiency * 0.5
                gpu_score = gpu_speed_score * 0.4 + gpu_advantage * 0.2 + memory_efficiency * 0.4
            
            # Additional bonus for significant GPU speedup
            if comp['speedup'] >= 2.0:
                gpu_score += 0.2  # Bonus for 2x+ speedup
            
            # Memory penalty for very high usage (>1GB total)
            if comp['total_memory'] > 1000:
                memory_penalty = min(0.3, (comp['total_memory'] - 1000) / 2000)
                cpu_score -= memory_penalty
                gpu_score -= memory_penalty
            
            scored_results.append({
                'batch_size': batch_size,
                'cpu_score': cpu_score,
                'gpu_score': gpu_score,
                'cpu_speed': comp['cpu_speed'],
                'gpu_speed': comp['gpu_speed'],
                'speedup': comp['speedup'],
                'total_memory': comp['total_memory'],
                'recommended_device': 'gpu' if gpu_score > cpu_score else 'cpu',
                'performance_gain': max(gpu_score, cpu_score),
                'system_memory_gb': system_memory_gb
            })
        
        # Find best overall configuration
        best_config = max(scored_results, key=lambda x: x['performance_gain'])
        
        # Generate reason based on choice
        if best_config['recommended_device'] == 'gpu':
            if best_config['speedup'] >= 2.0:
                reason = f"Significant performance gain: {best_config['speedup']:.1f}x speedup"
            else:
                reason = f"Better performance: {best_config['speedup']:.1f}x speedup"
        else:
            if best_config['total_memory'] < 100:
                reason = "Better memory efficiency"
            else:
                reason = "More stable performance"
        
        return {
            'optimal_device': best_config['recommended_device'],
            'optimal_batch_size': best_config['batch_size'],
            'best_result': best_config,
            'all_results': scored_results,
            'reason': reason
        }
    
    def cleanup(self):
        """Clean up test files and GPU memory."""
        # Clean up test images
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
        
        # Clear GPU memory
        if self._torch_components and 'torch' in self._torch_components:
            torch = self._torch_components['torch']
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clear model cache
        self._model_components = None
        gc.collect()


@pytest.mark.slow
class TestGPUCPUBatchOptimization:
    """Test suite for GPU and CPU batch optimization."""
    
    def setup_method(self):
        """Set up test environment."""
        self.optimizer = GPUCPUBatchOptimizer()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.optimizer.cleanup()
    
    def test_cpu_batch_sizes(self):
        """Test CPU batch sizes with real model inference."""
        batch_sizes = [1, 2, 4, 8]
        results = self.optimizer.test_device_batch_sizes('cpu', batch_sizes, image_count=12)
        
        assert len(results) > 0  # At least some results should succeed
        
        for result in results:
            assert result['device'] == 'cpu'
            assert result['avg_images_per_second'] > 0
            assert result['processed_count'] > 0
    
    @pytest.mark.skipif(not hasattr(__import__('torch', fromlist=['cuda']), 'cuda') or 
                       not __import__('torch').cuda.is_available(), 
                       reason="CUDA not available")
    def test_gpu_batch_sizes(self):
        """Test GPU batch sizes with real model inference."""
        batch_sizes = [1, 4, 8, 16]
        results = self.optimizer.test_device_batch_sizes('cuda', batch_sizes, image_count=12)
        
        assert len(results) > 0  # At least some results should succeed
        
        for result in results:
            assert result['device'] == 'cuda'
            assert result['avg_images_per_second'] > 0
            assert result['processed_count'] > 0
            assert result['gpu_available'] is True
    
    def test_cpu_gpu_comparison(self):
        """Test comparison between CPU and GPU performance."""
        batch_sizes = [1, 2, 4]
        comparison = self.optimizer.compare_cpu_gpu_performance(batch_sizes, image_count=10)
        
        assert 'cpu_results' in comparison
        assert len(comparison['cpu_results']) > 0
        
        # GPU results depend on availability
        if comparison['gpu_results']:
            assert 'comparison' in comparison
            assert len(comparison['comparison']) > 0
    
    def test_optimal_device_selection(self):
        """Test optimal device and batch size selection."""
        batch_sizes = [1, 2, 4]
        comparison = self.optimizer.compare_cpu_gpu_performance(batch_sizes, image_count=8)
        
        optimal = self.optimizer.find_optimal_device_batch_size(comparison)
        
        assert 'optimal_device' in optimal
        assert 'optimal_batch_size' in optimal
        assert optimal['optimal_device'] in ['cpu', 'gpu']
        assert optimal['optimal_batch_size'] in batch_sizes


def test_pytorch_availability():
    """Test PyTorch availability and GPU detection."""
    optimizer = GPUCPUBatchOptimizer()
    optimizer._load_torch_components()
    
    if optimizer._torch_components:
        torch = optimizer._torch_components['torch']
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("PyTorch not available")


if __name__ == "__main__":
    print("🔬 GPU/CPU Batch Size Optimization")
    print("=" * 45)
    
    # Test PyTorch availability
    test_pytorch_availability()
    
    optimizer = GPUCPUBatchOptimizer()
    
    try:
        # System info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"\n💻 System Memory: {memory_gb:.1f} GB")
        
        # Determine test parameters
        if memory_gb < 8:
            batch_sizes = [1, 2, 4]
            image_count = 10
        else:
            batch_sizes = [1, 2, 4, 8, 16]
            image_count = 15
        
        print(f"🧪 Testing batch sizes: {batch_sizes}")
        
        # Run comparison
        comparison = optimizer.compare_cpu_gpu_performance(batch_sizes, image_count)
        
        # Print CPU results
        print(f"\n🖥️  CPU Results:")
        print("-" * 20)
        for result in comparison['cpu_results']:
            batch_size = result['batch_size']
            speed = result['avg_images_per_second']
            memory = result['avg_cpu_memory']
            print(f"Batch {batch_size:2d}: {speed:6.2f} img/s, {memory:5.1f} MB CPU")
        
        # Print GPU results if available
        if comparison['gpu_results']:
            print(f"\n🚀 GPU Results:")
            print("-" * 20)
            for result in comparison['gpu_results']:
                batch_size = result['batch_size']
                speed = result['avg_images_per_second']
                cpu_memory = result['avg_cpu_memory']
                gpu_memory = result['avg_gpu_memory']
                print(f"Batch {batch_size:2d}: {speed:6.2f} img/s, {cpu_memory:5.1f} MB CPU, {gpu_memory:5.1f} MB GPU")
            
            # Print comparison
            print(f"\n⚡ Performance Comparison:")
            print("-" * 30)
            for comp in comparison['comparison']:
                batch_size = comp['batch_size']
                speedup = comp['speedup']
                total_memory = comp['total_memory']
                print(f"Batch {batch_size:2d}: {speedup:4.1f}x speedup, {total_memory:5.1f} MB total")
        
        # Find optimal configuration
        optimal = optimizer.find_optimal_device_batch_size(comparison)
        
        if optimal:
            print(f"\n🏆 Optimal Configuration:")
            print(f"   Device: {optimal['optimal_device'].upper()}")
            print(f"   Batch Size: {optimal['optimal_batch_size']}")
            print(f"   Reason: {optimal['reason']}")
            
            best = optimal['best_result']
            if optimal['optimal_device'] == 'gpu':
                print(f"   Speed: {best['gpu_speed']:.2f} images/second")
                print(f"   Speedup: {best['speedup']:.1f}x over CPU")
            else:
                print(f"   Speed: {best['cpu_speed']:.2f} images/second")
            print(f"   Memory: {best['total_memory']:.1f} MB total")
        
    finally:
        optimizer.cleanup()
        print("\n✅ GPU/CPU optimization complete!")