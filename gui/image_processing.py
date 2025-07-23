"""
Image processing functions for GUI without Streamlit dependencies.
"""

import os
import time
from typing import List, Tuple

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import Compose

from db import is_db_populated, load_duplicate_pairs, save_duplicate_pair
from local_media import get_file_info, load_image
from logger_config import logger

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- GPU / DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Using GPU: {gpu_name}")
else:
    logger.info("Using CPU for PyTorch operations")

# Load vit_b_16 with pretrained weights
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.to(device)  # Move model to the selected device
model.eval()  # Set model to evaluation mode


def convert_image_to_rgb(image: Image.Image) -> Image.Image:
    """
    Converts a PIL Image to RGB format if it's not already.
    This handles RGBA, P (palette), and L (grayscale) modes.

    Args:
        image: PIL Image object

    Returns:
        PIL Image in RGB format
    """
    if image.mode != "RGB":
        logger.debug(f"Converting image from {image.mode} to RGB")
        return image.convert("RGB")
    return image


def extract_features(image_path: str) -> np.ndarray:
    """
    Extract features from an image using ViT model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Feature vector as numpy array
    """
    try:
        # Load and preprocess image
        image = load_image(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return np.array([])
            
        image = convert_image_to_rgb(image)
        
        # Apply transforms
        preprocess = weights.transforms()
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(input_tensor)
            
        # Convert to numpy and normalize
        features_np = features.cpu().numpy().flatten()
        norm = np.linalg.norm(features_np)
        if norm > 0:
            features_np = features_np / norm
        else:
            logger.warning(f"Zero norm features for {image_path}")
            return np.array([])
        
        return features_np
        
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return np.array([])


def calculate_faiss_index_gui(media_files: List[str], progress_callback=None) -> bool:
    """
    Calculate FAISS index for GUI application.
    
    Args:
        media_files: List of image file paths
        progress_callback: Optional callback function for progress updates
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Starting FAISS index calculation for {len(media_files)} files")
        
        # Initialize FAISS index
        dimension = 1000  # ViT-B/16 feature dimension (1000 for classification head)
        index = faiss.IndexFlatL2(dimension)
        
        # Store metadata
        metadata = []
        features_list = []
        processed_count = 0
        
        for i, file_path in enumerate(media_files):
            if progress_callback:
                progress = int((i / len(media_files)) * 90)  # Reserve 10% for saving
                progress_callback(progress, f"Processing {os.path.basename(file_path)} ({i+1}/{len(media_files)})")
                
            # Extract features
            features = extract_features(file_path)
            if features.size > 0:
                if features.size != dimension:
                    logger.warning(f"Feature dimension mismatch for {file_path}: expected {dimension}, got {features.size}")
                    continue
                features_list.append(features)
                metadata.append(file_path)
                processed_count += 1
            else:
                logger.warning(f"Skipping file with no features: {file_path}")
                
        if not features_list:
            error_msg = "No valid features extracted from any files"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(100, error_msg)
            return False
            
        if progress_callback:
            progress_callback(95, f"Saving index with {len(features_list)} entries...")
            
        # Convert to numpy array and add to index
        features_array = np.array(features_list).astype('float32')
        index.add(features_array)
        
        # Save index and metadata
        faiss.write_index(index, "faiss_index.bin")
        np.save("metadata.npy", np.array(metadata))
        
        success_msg = f"FAISS index created successfully with {index.ntotal} entries from {processed_count} files"
        logger.info(success_msg)
        if progress_callback:
            progress_callback(100, success_msg)
        return True
        
    except Exception as e:
        error_msg = f"Error creating FAISS index: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if progress_callback:
            progress_callback(100, error_msg)
        return False


def generate_duplicate_db_gui(progress_callback=None) -> bool:
    """
    Generate duplicate database for GUI application.
    
    Args:
        progress_callback: Optional callback function for progress updates
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if progress_callback:
            progress_callback(10, "Loading FAISS index...")
            
        # Load FAISS index and metadata
        if not os.path.exists("faiss_index.bin") or not os.path.exists("metadata.npy"):
            logger.error("FAISS index or metadata not found")
            return False
            
        index = faiss.read_index("faiss_index.bin")
        metadata = np.load("metadata.npy")
        
        if progress_callback:
            progress_callback(30, "Searching for duplicates...")
            
        # Search for similar images
        k = min(10, index.ntotal)  # Number of nearest neighbors
        distances, indices = index.search(index.reconstruct_n(0, index.ntotal), k)
        
        duplicate_count = 0
        total_comparisons = len(distances)
        
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            if progress_callback:
                progress = 30 + int((i / total_comparisons) * 60)
                progress_callback(progress, f"Processing similarities {i+1}/{total_comparisons}")
                
            for j, (distance, idx) in enumerate(zip(dist_row, idx_row)):
                if i < idx and distance < 0.5:  # Threshold for similarity
                    similarity_score = max(0, 100 - (distance * 100))
                    save_duplicate_pair(metadata[i], metadata[idx], similarity_score)
                    duplicate_count += 1
                    
        if progress_callback:
            progress_callback(100, f"Found {duplicate_count} duplicate pairs")
            
        logger.info(f"Generated duplicate database with {duplicate_count} pairs")
        return True
        
    except Exception as e:
        logger.error(f"Error generating duplicate database: {e}")
        return False