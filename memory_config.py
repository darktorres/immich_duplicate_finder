"""
Memory optimization configuration for the duplicate finder application.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryConfig:
    """Configuration for memory optimization settings."""
    
    # Model settings
    use_cpu_only: bool = False  # Force CPU usage even if GPU is available
    model_precision: str = "float32"  # Options: "float32", "float16" (if supported)
    batch_size: int = 10  # Number of images to process before garbage collection
    
    # FAISS settings
    faiss_index_type: str = "IndexFlatL2"  # Options: "IndexFlatL2", "IndexIVFFlat" (more memory efficient)
    faiss_nlist: int = 100  # Number of clusters for IVF index (if used)
    
    # Processing settings
    max_image_size: Optional[tuple] = (1024, 1024)  # Resize large images to save memory
    lazy_loading: bool = True  # Load heavy components only when needed
    aggressive_gc: bool = True  # Force garbage collection frequently
    
    # Cache settings
    clear_cache_after_batch: bool = True  # Clear model cache after processing batches
    max_cache_size_mb: int = 500  # Maximum cache size in MB
    
    @classmethod
    def get_low_memory_config(cls) -> 'MemoryConfig':
        """Get configuration optimized for low memory systems (< 8GB RAM)."""
        return cls(
            use_cpu_only=True,
            model_precision="float32",
            batch_size=5,
            faiss_index_type="IndexIVFFlat",
            faiss_nlist=50,
            max_image_size=(512, 512),
            lazy_loading=True,
            aggressive_gc=True,
            clear_cache_after_batch=True,
            max_cache_size_mb=200
        )
    
    @classmethod
    def get_balanced_config(cls) -> 'MemoryConfig':
        """Get balanced configuration for medium memory systems (8-16GB RAM)."""
        return cls(
            use_cpu_only=False,
            model_precision="float32",
            batch_size=10,
            faiss_index_type="IndexFlatL2",
            max_image_size=(1024, 1024),
            lazy_loading=True,
            aggressive_gc=True,
            clear_cache_after_batch=True,
            max_cache_size_mb=500
        )
    
    @classmethod
    def get_high_performance_config(cls) -> 'MemoryConfig':
        """Get configuration optimized for high memory systems (> 16GB RAM)."""
        return cls(
            use_cpu_only=False,
            model_precision="float32",
            batch_size=20,
            faiss_index_type="IndexFlatL2",
            max_image_size=None,  # No resizing
            lazy_loading=False,  # Load everything at startup for speed
            aggressive_gc=False,
            clear_cache_after_batch=False,
            max_cache_size_mb=1000
        )
    
    def apply_environment_settings(self):
        """Apply memory-related environment settings."""
        if self.use_cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Set OpenMP thread limit to prevent excessive CPU usage
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "4"
        
        # PyTorch memory settings
        if not self.use_cpu_only:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def detect_system_memory() -> int:
    """Detect system memory in GB."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        return 8  # Default assumption


def get_recommended_config() -> MemoryConfig:
    """Get recommended configuration based on system memory."""
    system_memory_gb = detect_system_memory()
    
    if system_memory_gb < 8:
        print(f"Detected {system_memory_gb}GB RAM - Using low memory configuration")
        return MemoryConfig.get_low_memory_config()
    elif system_memory_gb < 16:
        print(f"Detected {system_memory_gb}GB RAM - Using balanced configuration")
        return MemoryConfig.get_balanced_config()
    else:
        print(f"Detected {system_memory_gb}GB RAM - Using high performance configuration")
        return MemoryConfig.get_high_performance_config()


# Global configuration instance
MEMORY_CONFIG = get_recommended_config()
MEMORY_CONFIG.apply_environment_settings()


def update_config(config: MemoryConfig):
    """Update the global memory configuration."""
    global MEMORY_CONFIG
    MEMORY_CONFIG = config
    config.apply_environment_settings()


if __name__ == "__main__":
    # Test configuration detection
    print("Memory Configuration Test")
    print("=" * 40)
    
    config = get_recommended_config()
    print(f"Recommended configuration:")
    print(f"  Use CPU only: {config.use_cpu_only}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max image size: {config.max_image_size}")
    print(f"  Lazy loading: {config.lazy_loading}")
    print(f"  Aggressive GC: {config.aggressive_gc}")
    print(f"  FAISS index type: {config.faiss_index_type}")