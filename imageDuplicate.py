import os
import time
import gc
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from db import is_db_populated, load_duplicate_pairs, save_duplicate_pair
from local_media import get_file_info, load_image
from logger_config import logger
from utility import display_asset_column
from memory_config import MEMORY_CONFIG

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global cache for lazy-loaded components
_component_cache = {}

def get_torch_components():
    """Lazy load PyTorch components."""
    if 'torch_components' not in _component_cache:
        logger.info("Loading PyTorch components (memory optimized)...")
        
        import torch
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        from torchvision.transforms import Compose
        
        # Device setup with memory configuration
        if MEMORY_CONFIG.use_cpu_only:
            device = torch.device("cpu")
            logger.info("Using CPU (memory optimization)")
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
        logger.info("Loading FAISS (memory optimized)...")
        import faiss
        _component_cache['faiss'] = faiss
    return _component_cache['faiss']

def get_model_and_transform():
    """Get model and transform with lazy loading and caching."""
    if 'model_components' not in _component_cache:
        logger.info("Loading Vision Transformer model (memory optimized)...")
        
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




# Global variables for paths
index_path = "faiss_index.bin"
metadata_path = "metadata.npy"


def extract_features(images: List[Image.Image]) -> np.ndarray:
    """Extract features from a batch of images using a pretrained model with memory optimization."""
    if not images:
        return np.array([])
    try:
        # Apply image size limit if configured
        processed_images = []
        if MEMORY_CONFIG.max_image_size:
            max_width, max_height = MEMORY_CONFIG.max_image_size
            for image in images:
                if image.size[0] > max_width or image.size[1] > max_height:
                    processed_images.append(image.resize((max_width, max_height), Image.Resampling.LANCZOS))
                    logger.debug(f"Resized image to {MEMORY_CONFIG.max_image_size}")
                else:
                    processed_images.append(image)
        else:
            processed_images = images

        # Now load heavy components (only when actually needed)
        components = get_model_and_transform()
        model, transform, device, torch = (
            components["model"],
            components["transform"],
            components["device"],
            components["torch"],
        )

        image_tensors = [transform(img) for img in processed_images]
        batch_tensor = torch.stack(image_tensors).to(device)

        with torch.no_grad():
            features = model(batch_tensor)

        features_np = features.cpu().numpy()

        # Normalize each feature vector in the batch
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        features_np = features_np / norms

        # Clear GPU cache if configured
        if MEMORY_CONFIG.clear_cache_after_batch and device.type == "cuda":
            torch.cuda.empty_cache()

        logger.debug(f"Extracted features for batch of {len(images)} images, shape: {features_np.shape}")
        return features_np

    except Exception as e:
        logger.error(f"Error extracting features from image batch: {e}")
        raise


def init_or_load_faiss_index() -> Tuple[Optional[object], List[str]]:
    """Initialize or load the FAISS index and metadata with memory optimization."""
    try:
        faiss = get_faiss()  # Lazy load FAISS
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # logger.info("Loading existing FAISS index and metadata")
            index = faiss.read_index(index_path)
            metadata = np.load(metadata_path, allow_pickle=True).tolist()
            # logger.info(f"Loaded FAISS index with {len(metadata)} entries")
        else:
            logger.info("No existing FAISS index found, will create new one")
            index = None
            metadata = []
        return index, metadata
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None, []


def save_faiss_index_and_metadata(index: object, metadata: List[str]) -> None:
    """Save the FAISS index and metadata to disk with memory optimization."""
    try:
        faiss = get_faiss()  # Lazy load FAISS
        faiss.write_index(index, index_path)
        np.save(metadata_path, np.array(metadata, dtype=object))
        logger.debug(f"Saved FAISS index with {len(metadata)} entries")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise


def calculateFaissIndex(media_files):
    """Calculate FAISS index with batch processing and memory optimization."""
    # Initialize session state variables
    if "message" not in st.session_state:
        st.session_state["message"] = ""
    if "progress" not in st.session_state:
        st.session_state["progress"] = 0
    if "stop_index" not in st.session_state:
        st.session_state["stop_index"] = False

    # Set up the UI components
    progress_bar = st.progress(st.session_state["progress"])
    stop_button = st.button("Stop Index Processing")
    message_placeholder = st.empty()

    if stop_button:
        st.session_state["stop_index"] = True
        st.session_state["calculate_faiss"] = False

    index, existing_metadata = init_or_load_faiss_index()
    files_to_process = sorted([f for f in media_files if f not in existing_metadata])
    total_files_to_process = len(files_to_process)

    if not files_to_process:
        st.success("FAISS index is already up to date.")
        st.session_state["calculate_faiss"] = False
        return

    processed_files = 0
    error_files = 0
    total_time = 0

    all_new_features_list = []
    all_new_metadata = []

    batch_size = MEMORY_CONFIG.batch_size
    st.info(f"⚡ Processing {total_files_to_process} new files in batches of {batch_size} for memory optimization")

    for i in range(0, total_files_to_process, batch_size):
        if st.session_state["stop_index"]:
            st.session_state["message"] = "Processing stopped by user."
            message_placeholder.text(st.session_state["message"])
            break

        batch_paths = files_to_process[i : i + batch_size]
        batch_images = []
        valid_paths_in_batch = []

        start_time = time.time()

        for file_path in batch_paths:
            image = load_image(file_path)
            if image:
                batch_images.append(image)
                valid_paths_in_batch.append(file_path)
            else:
                error_files += 1

        if batch_images:
            try:
                # Process the whole batch
                features_batch = extract_features(batch_images)
                all_new_features_list.append(features_batch)
                all_new_metadata.extend(valid_paths_in_batch)
                processed_files += len(batch_images)
            except Exception as e:
                logger.error(f"Error processing batch starting with {batch_paths[0]}: {e}")
                error_files += len(batch_images)

        total_time += time.time() - start_time
        processed_count = processed_files + error_files

        # Update progress
        progress_percentage = (i + len(batch_paths)) / total_files_to_process
        st.session_state["progress"] = progress_percentage
        progress_bar.progress(progress_percentage)

        estimated_time_remaining = (total_time / processed_count) * (total_files_to_process - processed_count) if processed_count > 0 else 0
        estimated_time_remaining_min = int(estimated_time_remaining / 60)

        st.session_state["message"] = (
            f"Processing file {i + len(batch_paths)}/{total_files_to_process} - "
            f"(Processed: {processed_files}, Errors: {error_files}). "
            f"Estimated time remaining: {estimated_time_remaining_min} minutes."
        )
        message_placeholder.text(st.session_state["message"])

        # Force garbage collection after each batch
        if MEMORY_CONFIG.aggressive_gc:
            gc.collect()

        # Clear GPU cache if using CUDA
        if "model_components" in _component_cache:
            components = _component_cache["model_components"]
            if components["device"].type == "cuda" and MEMORY_CONFIG.clear_cache_after_batch:
                components["torch"].cuda.empty_cache()

    # After the loop, update index and save ONCE
    if not st.session_state["stop_index"] and all_new_features_list:
        try:
            st.session_state["message"] = "Finalizing index..."
            message_placeholder.text(st.session_state["message"])
            features_array = np.vstack(all_new_features_list)
            faiss = get_faiss()

            if index is None:
                dimension = features_array.shape[1]
                # Use memory-efficient index type if configured
                if MEMORY_CONFIG.faiss_index_type == "IndexIVFFlat":
                    quantizer = faiss.IndexFlatL2(dimension)
                    index = faiss.IndexIVFFlat(quantizer, dimension, MEMORY_CONFIG.faiss_nlist)
                    logger.info(f"Creating memory-efficient IndexIVFFlat with {MEMORY_CONFIG.faiss_nlist} clusters")
                else:
                    index = faiss.IndexFlatL2(dimension)
                    logger.info("Creating new FAISS index")

            index.add(features_array)
            final_metadata = existing_metadata + all_new_metadata
            save_faiss_index_and_metadata(index, final_metadata)

            st.session_state["message"] = f"Processing complete! Added {processed_files} new files to the index."
            message_placeholder.text(st.session_state["message"])

        except Exception as e:
            st.session_state["message"] = f"Error finalizing index: {e}"
            message_placeholder.text(st.session_state["message"])
            logger.error(f"Error finalizing FAISS index: {e}")

    st.session_state["stop_index"] = False
    progress_bar.progress(1.0)
    st.session_state["calculate_faiss"] = False


def generate_db_duplicate():
    """Generate duplicate database with memory optimization."""
    st.write("Database initialization with memory optimization")
    index, metadata = init_or_load_faiss_index()
    if not index or not metadata:
        st.write("FAISS index or metadata not available.")
        return

    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False

    if st.button("Stop Finding Duplicates"):
        st.session_state["stop_requested"] = True
        st.session_state["generate_db_duplicate"] = False

    num_vectors = index.ntotal
    message_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Process in batches to manage memory
    batch_size = MEMORY_CONFIG.batch_size * 2  # Larger batches for search
    st.info(f"💾 Processing in batches of {batch_size} for memory efficiency")
    
    for batch_start in range(0, num_vectors, batch_size):
        if st.session_state["stop_requested"]:
            message_placeholder.text("Processing was stopped by the user.")
            progress_bar.empty()
            st.session_state["stop_requested"] = False
            return None
            
        batch_end = min(batch_start + batch_size, num_vectors)
        
        for i in range(batch_start, batch_end):
            progress = (i + 1) / num_vectors
            message_placeholder.text(f"Finding duplicates: processing vector {i + 1} of {num_vectors}")
            progress_bar.progress(progress)

            query_vector = np.array([index.reconstruct(i)])
            distances, indices = index.search(query_vector, 2)

            for j in range(1, indices.shape[1]):
                idx1, idx2 = i, indices[0][j]
                if idx1 != idx2:
                    sorted_pair = (min(idx1, idx2), max(idx1, idx2))
                    if sorted_pair[0] < len(metadata) and sorted_pair[1] < len(metadata):
                        save_duplicate_pair(metadata[sorted_pair[0]], metadata[sorted_pair[1]], distances[0][j])
                    else:
                        st.error(f"Metadata index out of range: {sorted_pair}")
        
        # Force garbage collection after each batch
        if MEMORY_CONFIG.aggressive_gc:
            gc.collect()

    message_placeholder.text(f"Finished processing {num_vectors} vectors.")
    progress_bar.empty()
    st.session_state["stop_requested"] = False


def show_duplicate_photos_faiss(limit, min_threshold, max_threshold):
    """Show duplicate photos with memory optimization."""
    if not is_db_populated():
        st.write("The database does not contain any duplicate entries. Please generate/update the database.")
        return

    duplicates = load_duplicate_pairs(min_threshold, max_threshold)

    if duplicates:
        # Memory-aware result limiting
        max_safe_limit = min(limit, 50)  # Cap at 50 for memory safety
        if limit > max_safe_limit:
            st.warning(f"⚠️ Limiting results to {max_safe_limit} pairs for memory optimization (requested: {limit})")
            
        st.write(f"Found {len(duplicates)} duplicate pairs with FAISS code within threshold {min_threshold} < x < {max_threshold}:")
        st.info("⚡ Memory optimized display - images loaded efficiently")
        
        progress_bar = st.progress(0)
        num_duplicates_to_show = min(len(duplicates), max_safe_limit)

        for i, dup_pair in enumerate(duplicates[:num_duplicates_to_show]):
            try:
                if st.session_state.get("stop_requested", False):
                    st.write("Processing was stopped by the user.")
                    st.session_state["stop_requested"] = False
                    st.session_state["generate_db_duplicate"] = False
                    break

                file_path_1, file_path_2, similarity = dup_pair
                st.subheader(f"Pair {i + 1} - Similarity Score: {similarity:.4f}")

                progress = (i + 1) / num_duplicates_to_show
                progress_bar.progress(progress)

                image1 = load_image(file_path_1)
                image2 = load_image(file_path_2)
                asset1_info = get_file_info(file_path_1)
                asset2_info = get_file_info(file_path_2)

                if image1 is not None and image2 is not None:
                    # Apply image size limits for memory optimization
                    if MEMORY_CONFIG.max_image_size:
                        max_width, max_height = MEMORY_CONFIG.max_image_size
                        if image1.size[0] > max_width or image1.size[1] > max_height:
                            image1 = image1.resize((max_width, max_height), Image.Resampling.LANCZOS)
                        if image2.size[0] > max_width or image2.size[1] > max_height:
                            image2 = image2.resize((max_width, max_height), Image.Resampling.LANCZOS)
                    
                    # Convert PIL images to numpy arrays
                    image1 = np.array(image1)
                    image2 = np.array(image2)
                    
                    image_comparison(
                        img1=image1,
                        img2=image2,
                        label1=os.path.basename(file_path_1),
                        label2=os.path.basename(file_path_2),
                        width=700,
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=False,
                    )

                    col1, col2 = st.columns(2)
                    display_asset_column(col1, asset1_info, asset2_info, file_path_1, file_path_2)
                    display_asset_column(col2, asset2_info, asset1_info, file_path_2, file_path_1)
                else:
                    st.write(f"Missing information for one or both assets: {file_path_1}, {file_path_2}")

                st.markdown("---")
                
                # Force garbage collection periodically
                if MEMORY_CONFIG.aggressive_gc and i % 5 == 0:
                    gc.collect()
                    
            except Exception as e:
                st.write(f"Error processing duplicate pair: {str(e)}")
                logger.error(f"Error processing duplicate pair {i + 1}: {e}")
                
        progress_bar.progress(100)
    else:
        st.write("No duplicates found.")
