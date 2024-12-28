import os
import numpy as np
import dask.array as da  # Alias for Dask arrays
from dask import delayed  # Lazy evaluation for tasks
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def load_category_dask(preprocessed_outdir, masks_dir, category, split):
    """
    Load data and segmentation masks for one category using Dask delayed.
    Args:
    - preprocessed_outdir: Path to the dataset directory (flat structure).
    - masks_dir: Path to the segmentation masks directory (category-wise subfolders).
    - category: Category name (e.g., "AnnualCrop").
    - split: Split name ("train", "val", "test").
    Returns:
    - Delayed objects for X and masks.
    """
    try:
        logging.info(f"Loading {category} {split} data...")

        # Load image data
        X_path = os.path.join(preprocessed_outdir, f"{category}_{split}.npy")
        X_data = delayed(np.load)(X_path, mmap_mode='r')

        # Load all mask files
        mask_dir = os.path.join(masks_dir, category, split)
        mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".npy")])

        # Concatenate masks lazily
        mask_data = da.concatenate(
            [da.from_delayed(delayed(np.load)(f, mmap_mode='r'), shape=(513, 513), dtype=np.float32) for f in mask_files],
            axis=0
        )

        num_samples = mask_data.shape[0]
        return da.from_delayed(X_data, shape=(num_samples, 513, 513, 3), dtype=np.float32), mask_data
    except Exception as e:
        logging.error(f"Error loading {category} {split} data: {e}")
        return da.empty((0, 513, 513, 3), dtype=np.float32), da.empty((0, 513, 513), dtype=np.float32)


def load_and_combine_dask(preprocessed_outdir, masks_dir, categories, split):
    """
    Load and combine datasets in parallel for a given split.
    Args:
    - preprocessed_outdir: Path to the dataset directory (flat structure).
    - masks_dir: Path to the segmentation masks directory (category-wise subfolders).
    - categories: List of category names.
    - split: The data split to load ('train', 'val', 'test').
    Returns:
    - Combined Dask arrays for X and masks.
    """
    delayed_results = [
        load_category_dask(preprocessed_outdir, masks_dir, category, split) for category in categories
    ]
    X_delayed, masks_delayed = zip(*delayed_results)

    # Combine delayed results into Dask arrays
    X_combined = da.concatenate(X_delayed, axis=0)
    masks_combined = da.concatenate(masks_delayed, axis=0)

    return X_combined, masks_combined

def ensure_directory_exists(path):
    """
    Ensure that the directory for the given path exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")


if __name__ == "__main__":
    # Paths and categories
    preprocessed_outdir = "../data/preprocessed_output_data"
    output_masks_dir = "../data/segmentation_masks"
    save_training_files_to = "../data/training_files"

    # Ensure the base directory exists
    ensure_directory_exists(save_training_files_to)

    categories = ["AnnualCrop", "Forest", "Residential", "Highway", "HerbaceousVegetation", "Industrial"]

    for split in ["train", "val", "test"]:
        logging.info(f"Processing {split} split...")
        
        # Load data for the split
        X_split, masks_split = load_and_combine_dask(preprocessed_outdir, output_masks_dir, categories, split)
        
        # Log chunk shapes before rechunking
        logging.info(f"{split.capitalize()} mask chunks before rechunking: {masks_split.chunks}")
        
        # Check if data exists
        if masks_split.shape[0] == 0:
            logging.error(f"No data available for {split} masks. Skipping saving for this split.")
            continue
    
        # Rechunk masks to consolidate into fewer files
        logging.info(f"Rechunking {split} masks for efficient saving...")
        masks_split = masks_split.rechunk((128, 513, 513))  # Adjust chunk size if needed
    
        # Ensure directories exist
        ensure_directory_exists(os.path.join(save_training_files_to, split))
        ensure_directory_exists(os.path.join(save_training_files_to, f"{split}_masks"))
        
        # Save data and masks
        da.to_npy_stack(os.path.join(save_training_files_to, split), X_split, axis=0)
        da.to_npy_stack(os.path.join(save_training_files_to, f"{split}_masks"), masks_split, axis=0)
    
        # Log shapes
        logging.info(f"{split.capitalize()} data shape: {X_split.shape}, {masks_split.shape}")

