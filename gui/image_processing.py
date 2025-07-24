"""
Memory-optimized image processing functions for GUI without heavy imports at startup.
"""

import os
import gc
from typing import List, Optional, Callable

import numpy as np
from PIL import Image

from db import save_duplicate_pair
from local_media import load_image
from logger_config import logger
from memory_config import MEMORY_CONFIG

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global cache for lazy-loaded components
_component_cache = {}


def get_torch_components():
    """Lazy load PyTorch components."""
    if 'torch_components' not in _component_cache:
        logger.info("Loading PyTorch components (lazy loading)...")
        
        import torch
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        from torchvision.transforms import Compose
        
        # Device setup with memory configuration
        if MEMORY_CONFIG.use_cpu_only:
            device = torch.device("cpu")
            logger.info("Using CPU (forced by memory configuration)")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {gpu_name}")
            else:
                logger.info("Using CPU for PyTorch operations")
        
        _component_cache['torch_components'] = {
            'torch': torch,
            'ViT_B_16_Weights': ViT_B_16_Weights,
            'vit_b_16': vit_b_16,
            'Compose': Compose,
            'device': device
        }
    
    return _component_cache['torch_components']


def get_faiss():
    """Lazy load FAISS."""
    if 'faiss' not in _component_cache:
        logger.info("Loading FAISS (lazy loading)...")
        import faiss
        _component_cache['faiss'] = faiss
    return _component_cache['faiss']


def get_model_and_transform():
    """Get model and transform with lazy loading and caching."""
    if 'model_components' not in _component_cache:
        logger.info("Loading Vision Transformer model (lazy loading)...")
        
        torch_components = get_torch_components()
        torch = torch_components['torch']
        ViT_B_16_Weights = torch_components['ViT_B_16_Weights']
        vit_b_16 = torch_components['vit_b_16']
        Compose = torch_components['Compose']
        device = torch_components['device']
        
        # Load model components
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.to(device)
        model.eval()
        
        # Create transform with RGB conversion
        def convert_image_to_rgb(image: Image.Image) -> Image.Image:
            if image.mode != "RGB":
                logger.debug(f"Converting image from {image.mode} to RGB")
                return image.convert("RGB")
            return image
        
        vit_transforms = weights.transforms()
        transform = Compose([convert_image_to_rgb, vit_transforms])
        
        _component_cache['model_components'] = {
            'model': model,
            'transform': transform,
            'device': device,
            'torch': torch
        }
        
        logger.info("Model loaded successfully")
    
    return _component_cache['model_components']


def is_model_loaded() -> bool:
    """Check if the model is currently loaded in memory."""
    return 'model_components' in _component_cache


def clear_model_cache():
    """Clear the model cache to free memory."""
    if 'model_components' in _component_cache:
        logger.info("Clearing model cache...")
        
        # Clear GPU cache if available
        if 'model_components' in _component_cache:
            components = _component_cache['model_components']
            if components['device'].type == 'cuda':
                components['torch'].cuda.empty_cache()
        
        # Remove from cache
        del _component_cache['model_components']
        
        # Force garbage collection
        gc.collect()
        logger.info("Model cache cleared")


def extract_features(images: List[Image.Image]) -> np.ndarray:
    """Extract features from a batch of images using ViT model with memory optimization."""
    if not images:
        return np.array([])
    try:
        # Now load heavy components (only when actually needed)
        components = get_model_and_transform()
        model = components['model']
        transform = components['transform']
        device = components['device']
        torch = components['torch']

        # Process image
        input_tensors = [transform(image) for image in images]
        batch_tensor = torch.stack(input_tensors).to(device)

        # Extract features
        with torch.no_grad():
            features = model(batch_tensor)

        # Convert to numpy and normalize
        features_np = features.cpu().numpy()
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        features_np = features_np / norms

        # Clear GPU cache if configured
        if MEMORY_CONFIG.clear_cache_after_batch and device.type == 'cuda':
            torch.cuda.empty_cache()

        return features_np

    except Exception as e:
        logger.error(f"Error extracting features from image batch: {e}")
        return np.array([])


def calculate_faiss_index_gui(media_files: List[str], progress_callback: Optional[Callable] = None) -> bool:
    """Calculate FAISS index for GUI application with batch processing and memory optimization."""
    try:
        logger.info(f"Starting memory-optimized FAISS index calculation for {len(media_files)} files")

        # Get FAISS (lazy loaded)
        faiss = get_faiss()

        # Initialize FAISS index based on configuration
        dimension = 1000  # ViT-B/16 feature dimension
        
        if MEMORY_CONFIG.faiss_index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, MEMORY_CONFIG.faiss_nlist)
            logger.info(f"Using memory-efficient IndexIVFFlat with {MEMORY_CONFIG.faiss_nlist} clusters")
        else:
            index = faiss.IndexFlatL2(dimension)
            logger.info("Using IndexFlatL2")

        all_features_list = []
        all_metadata = []
        processed_count = 0
        batch_size = MEMORY_CONFIG.batch_size
        
        for i in range(0, len(media_files), batch_size):
            batch_end = min(i + batch_size, len(media_files))
            batch_paths = media_files[i:batch_end]
            
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch_paths)} files)")
            
            batch_images = []
            valid_paths_in_batch = []
            
            for j, file_path in enumerate(batch_paths):
                file_index = i + j
                
                if progress_callback:
                    progress = int((file_index / len(media_files)) * 90)
                    progress_callback(progress, f"Loading {os.path.basename(file_path)} ({file_index + 1}/{len(media_files)})")

                image = load_image(file_path)
                if image:
                    # Apply image size limit if configured
                    if MEMORY_CONFIG.max_image_size:
                        max_width, max_height = MEMORY_CONFIG.max_image_size
                        if image.size[0] > max_width or image.size[1] > max_height:
                            image = image.resize((max_width, max_height), Image.Resampling.LANCZOS)
                    
                    batch_images.append(image)
                    valid_paths_in_batch.append(file_path)
                else:
                    logger.warning(f"Skipping file that could not be loaded: {file_path}")

            if not batch_images:
                continue

            # Extract features for the whole batch
            features_batch = extract_features(batch_images)
            if features_batch.size > 0:
                all_features_list.append(features_batch)
                all_metadata.extend(valid_paths_in_batch)
                processed_count += len(features_batch)
            
            # Force garbage collection after each batch
            if MEMORY_CONFIG.aggressive_gc:
                gc.collect()

        if not all_features_list:
            error_msg = "No valid features extracted from any files"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(100, error_msg)
            return False

        if progress_callback:
            progress_callback(95, f"Building final index with {len(all_metadata)} entries...")

        features_array = np.vstack(all_features_list).astype("float32")
        
        if MEMORY_CONFIG.faiss_index_type == "IndexIVFFlat":
            logger.info("Training IVF index...")
            index.train(features_array)
        
        index.add(features_array)

        faiss.write_index(index, "faiss_index.bin")
        np.save("metadata.npy", np.array(all_metadata))

        success_msg = f"FAISS index created successfully with {index.ntotal} entries from {processed_count} files"
        logger.info(success_msg)
        if progress_callback:
            progress_callback(100, success_msg)
            
        return True

    except Exception as e:
        error_msg = f"Error creating FAISS index: {e}"
        logger.error(error_msg, exc_info=True)
        if progress_callback:
            progress_callback(100, error_msg)
        return False


def generate_duplicate_db_gui(progress_callback: Optional[Callable] = None) -> bool:
    """Generate duplicate database for GUI application with memory optimization."""
    try:
        if progress_callback:
            progress_callback(10, "Loading FAISS index...")

        # Get FAISS (lazy loaded)
        faiss = get_faiss()

        # Load FAISS index and metadata
        if not os.path.exists("faiss_index.bin") or not os.path.exists("metadata.npy"):
            logger.error("FAISS index or metadata not found")
            return False

        index = faiss.read_index("faiss_index.bin")
        metadata = np.load("metadata.npy")

        if progress_callback:
            progress_callback(30, "Searching for duplicates...")

        # Search for similar images in batches
        batch_size = MEMORY_CONFIG.batch_size * 2  # Larger batches for search
        duplicate_count = 0
        total_vectors = index.ntotal
        
        for batch_start in range(0, total_vectors, batch_size):
            batch_end = min(batch_start + batch_size, total_vectors)
            
            if progress_callback:
                progress = 30 + int((batch_start / total_vectors) * 60)
                progress_callback(progress, f"Processing similarities {batch_start + 1}-{batch_end}/{total_vectors}")
            
            # Get batch of vectors
            batch_vectors = np.array([index.reconstruct(i) for i in range(batch_start, batch_end)])
            
            # Search for neighbors
            k = min(10, total_vectors)  # Number of nearest neighbors
            distances, indices = index.search(batch_vectors.astype('float32'), k)
            
            # Process results
            for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                actual_i = batch_start + i
                
                for j, (distance, idx) in enumerate(zip(dist_row, idx_row)):
                    if actual_i < idx:
                        similarity_score = max(0, 100 - (distance * 100))
                        save_duplicate_pair(metadata[actual_i], metadata[idx], similarity_score)
                        duplicate_count += 1
            
            # Garbage collection after each batch
            if MEMORY_CONFIG.aggressive_gc:
                gc.collect()

        if progress_callback:
            progress_callback(100, f"Found {duplicate_count} duplicate pairs")

        logger.info(f"Generated duplicate database with {duplicate_count} pairs")
        return True

    except Exception as e:
        logger.error(f"Error generating duplicate database: {e}")
        return False


def get_memory_usage_info() -> dict:
    """Get current memory usage information."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'model_loaded': is_model_loaded(),
            'config': MEMORY_CONFIG
        }
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {}
