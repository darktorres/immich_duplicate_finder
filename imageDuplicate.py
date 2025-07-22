import os
import time

import faiss
import numpy as np
import streamlit as st
import torch
from streamlit_image_comparison import image_comparison
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import Compose

from db import is_db_populated, load_duplicate_pairs, save_duplicate_pair
from local_media import get_file_info, load_image
from utility import display_asset_column

# Set the environment variable to allow multiple OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- GPU / DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# --- FAISS GPU SETUP ---
res = None
if device.type == "cuda":
    try:
        res = faiss.StandardGpuResources()
        print("FAISS GPU support enabled.")
    except AttributeError:
        print("Warning: faiss-gpu not installed. Falling back to CPU for FAISS.")
        print("Install it with: pip install faiss-gpu-cuXX (e.g., faiss-gpu-cu12 for CUDA 12.1)")
        res = None
# --- END SETUP ---

# Load vit_b_16 with pretrained weights
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
model.to(device)  # Move model to the selected device
model.eval()  # Set model to evaluation mode


def convert_image_to_rgb(image):
    """
    Converts a PIL Image to RGB format if it's not already.
    This handles RGBA, P (palette), and L (grayscale) modes.
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


vit_transforms = weights.transforms()

transform = Compose(
    [
        convert_image_to_rgb,
        vit_transforms,
    ]
)

# Global variables for paths
index_path = "faiss_index.bin"
metadata_path = "metadata.npy"


def extract_features(image):
    """Extract features from an image using a pretrained model."""
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        features = model(image_tensor)
    return features.cpu().numpy().flatten()  # Move features to CPU before converting to numpy


def init_or_load_faiss_index():
    """Initialize or load the FAISS index and metadata, ensuring index is ready for use."""
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        cpu_index = faiss.read_index(index_path)
        if res:
            print("Moving FAISS index to GPU...")
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            index = cpu_index
        metadata = np.load(metadata_path, allow_pickle=True).tolist()
    else:
        index = None
        metadata = []
    return index, metadata


def save_faiss_index_and_metadata(index, metadata):
    """Save the FAISS index and metadata to disk."""
    if res and hasattr(index, "getDevice"):  # Check if it is a GPU index
        print("Moving FAISS index to CPU for saving...")
        cpu_index = faiss.index_gpu_to_cpu(index)
    else:
        cpu_index = index

    faiss.write_index(cpu_index, index_path)
    np.save(metadata_path, np.array(metadata, dtype=object))


def update_faiss_index(file_path):
    """Update the FAISS index and metadata with a new image and its path."""
    index, existing_metadata = init_or_load_faiss_index()

    if file_path in existing_metadata:
        return "skipped"

    image = load_image(file_path)
    if image is None:
        return "error"

    features = extract_features(image)

    if index is None:
        # Initialize the FAISS index with the correct dimension if it's the first time
        dimension = features.shape[0]
        cpu_index = faiss.IndexFlatL2(dimension)
        if res:
            print("Creating new FAISS index on GPU.")
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            print("Creating new FAISS index on CPU.")
            index = cpu_index

    index.add(np.array([features], dtype="float32"))
    existing_metadata.append(file_path)

    save_faiss_index_and_metadata(index, existing_metadata)
    return "processed"


def calculateFaissIndex(media_files):
    # Initialize session state variables if they are not already set
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

    # Check if stop was requested and reset it if button is pressed
    if stop_button:
        st.session_state["stop_index"] = True
        st.session_state["calculate_faiss"] = False

    total_files = len(media_files)
    processed_files = 0
    skipped_files = 0
    error_files = 0
    total_time = 0

    for i, file_path in enumerate(media_files):
        if st.session_state["stop_index"]:
            st.session_state["message"] = "Processing stopped by user."
            message_placeholder.text(st.session_state["message"])
            break  # Break the loop if stop is requested

        start_time = time.time()

        status = update_faiss_index(file_path)
        if status == "processed":
            processed_files += 1
        elif status == "skipped":
            skipped_files += 1
        elif status == "error":
            error_files += 1

        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time

        # Update progress and messages
        progress_percentage = (i + 1) / total_files
        st.session_state["progress"] = progress_percentage
        progress_bar.progress(progress_percentage)
        estimated_time_remaining = (total_time / (i + 1)) * (total_files - (i + 1))
        estimated_time_remaining_min = int(estimated_time_remaining / 60)

        st.session_state["message"] = (
            f"Processing file {i + 1}/{total_files} - (Processed: {processed_files}, Skipped: {skipped_files}, Errors: {error_files}). Estimated time remaining: {estimated_time_remaining_min} minutes."
        )
        message_placeholder.text(st.session_state["message"])

    # Reset stop flag at the end of processing
    st.session_state["stop_index"] = False
    # Check if we've gone through all files (regardless of processed vs skipped)
    if not st.session_state.get("stop_index", False):
        st.session_state["message"] = "Processing complete!"
        message_placeholder.text(st.session_state["message"])
        progress_bar.progress(1.0)
        st.session_state["calculate_faiss"] = False  # Reset the flag to prevent re-running


def generate_db_duplicate():
    st.write("Database initialization")
    index, metadata = init_or_load_faiss_index()
    if not index or not metadata:
        st.write("FAISS index or metadata not available.")
        return

    # Check and update the stop mechanism in session state
    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False

    # Button to request stopping
    if st.button("Stop Finding Duplicates"):
        st.session_state["stop_requested"] = True
        st.session_state["generate_db_duplicate"] = False

    num_vectors = index.ntotal
    message_placeholder = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_vectors):
        # Check if stop has been requested
        if st.session_state["stop_requested"]:
            message_placeholder.text("Processing was stopped by the user.")
            progress_bar.empty()
            # Optionally, reset the stop flag here if you want the process to be restartable without refreshing the page
            st.session_state["stop_requested"] = False
            return None

        progress = (i + 1) / num_vectors
        message_placeholder.text(f"Finding duplicates: processing vector {i + 1} of {num_vectors}")
        progress_bar.progress(progress)

        query_vector = np.array([index.reconstruct(i)])
        distances, indices = index.search(query_vector, 2)

        for j in range(1, indices.shape[1]):
            # if distances[0][j] < threshold:
            idx1, idx2 = i, indices[0][j]
            if idx1 != idx2:
                sorted_pair = (min(idx1, idx2), max(idx1, idx2))
                # Check if the indices in sorted_pair are within the bounds of metadata
                if sorted_pair[0] < len(metadata) and sorted_pair[1] < len(metadata):
                    save_duplicate_pair(metadata[sorted_pair[0]], metadata[sorted_pair[1]], distances[0][j])
                else:
                    st.error(f"Metadata index out of range: {sorted_pair}")
                    # Optionally log more details or handle this case further

    message_placeholder.text(f"Finished processing {num_vectors} vectors.")
    progress_bar.empty()
    # Reset the stop flag after completion
    st.session_state["stop_requested"] = False


def show_duplicate_photos_faiss(limit, min_threshold, max_threshold):
    # First check if the database is populated
    if not is_db_populated():
        st.write("The database does not contain any duplicate entries. Please generate/update the database.")
        return  # Exit the function early if the database is not populated

    # Load duplicates from database
    duplicates = load_duplicate_pairs(min_threshold, max_threshold)

    if duplicates:
        st.write(f"Found {len(duplicates)} duplicate pairs with FAISS code within threshold {min_threshold} < x < {max_threshold}:")
        progress_bar = st.progress(0)
        num_duplicates_to_show = min(len(duplicates), limit)

        for i, dup_pair in enumerate(duplicates[:num_duplicates_to_show]):
            try:
                # Check if stop was requested
                if st.session_state.get("stop_requested", False):
                    st.write("Processing was stopped by the user.")
                    st.session_state["stop_requested"] = False  # Reset the flag for future operations
                    st.session_state["generate_db_duplicate"] = False
                    break  # Exit the loop

                file_path_1, file_path_2, similarity = dup_pair

                st.subheader(f"Pair {i + 1} - Similarity Score: {similarity:.4f}")

                progress = (i + 1) / num_duplicates_to_show
                progress_bar.progress(progress)

                image1 = load_image(file_path_1)
                image2 = load_image(file_path_2)
                asset1_info = get_file_info(file_path_1)
                asset2_info = get_file_info(file_path_2)

                if image1 is not None and image2 is not None:
                    # Convert PIL images to numpy arrays if necessary
                    image1 = np.array(image1)
                    image2 = np.array(image2)
                    # Proceed with image comparison
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
                    #    with col1:
                    #        st.image(image1, caption=f"Name: {asset_id_1}")
                    #    with col2:
                    #        st.image(image2, caption=f"Name: {asset_id_2}")

                    display_asset_column(col1, asset1_info, asset2_info, file_path_1, file_path_2)
                    display_asset_column(col2, asset2_info, asset1_info, file_path_2, file_path_1)
                else:
                    st.write(f"Missing information for one or both assets: {file_path_1}, {file_path_2}")

                st.markdown("---")
            except Exception as e:
                st.write(f"Error processing duplicate pair: {str(e)}")
                print(f"Error processing duplicate pair {i+1}: {e}")
        progress_bar.progress(100)
    else:
        st.write("No duplicates found.")
