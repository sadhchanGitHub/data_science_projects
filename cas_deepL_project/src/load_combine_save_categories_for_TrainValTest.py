import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_category(args):
    """
    Load data for one category and return the arrays for X and y.
    Args:
    - args: Tuple containing (preprocessed_dir, category, split)
    Returns:
    - Tuple of numpy arrays (X, y) for the given category and split.
    """
    preprocessed_dir, category, split = args
    try:
        X_data = np.load(os.path.join(preprocessed_dir, f"{category}_{split}.npy"))
        y_data = np.load(os.path.join(preprocessed_dir, f"{category}_{split}_labels.npy"))
        logging.info(f"Loaded {category} {split} data: {X_data.shape[0]} samples")
        return X_data, y_data
    except Exception as e:
        logging.error(f"Error loading {category} {split} data: {e}")
        return np.empty((0, *image_size, 3)), np.empty((0,))

def load_and_combine_parallel(preprocessed_dir, categories, split, max_workers=4):
    """
    Load and combine datasets in parallel for a given split.
    Args:
    - preprocessed_dir: Path to the dataset directory.
    - categories: List of category names.
    - split: The data split to load ('train', 'val', 'test').
    - max_workers: Number of processes to use for parallel loading.
    Returns:
    - Combined numpy arrays for X and y.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_category, [(preprocessed_dir, category, split) for category in categories]))
    
    # Combine results from all categories
    X, y = zip(*results)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and preprocess datasets for EuroSAT.")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Path to preprocessed data directory")
    parser.add_argument("--categories", type=str, nargs='+', required=True, help="List of categories to process")
    parser.add_argument("--image_size", type=int, default=128, help="Resize images to this size (square)")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of workers for parallel processing")

    args = parser.parse_args()

    preprocessed_dir = args.preprocessed_dir
    categories = args.categories
    image_size = args.image_size
    max_workers = args.max_workers

    # Load training, validation, and test data
    logging.info("Loading training data...")
    X_train, y_train = load_and_combine_parallel(preprocessed_dir, categories, "train", max_workers)
    np.save(os.path.join(preprocessed_dir, "train.npy"), X_train)
    np.save(os.path.join(preprocessed_dir, "train_labels.npy"), y_train)

    logging.info("Loading validation data...")
    X_val, y_val = load_and_combine_parallel(preprocessed_dir, categories, "val", max_workers)
    np.save(os.path.join(preprocessed_dir, "val.npy"), X_val)
    np.save(os.path.join(preprocessed_dir, "val_labels.npy"), y_val)

    logging.info("Loading test data...")
    X_test, y_test = load_and_combine_parallel(preprocessed_dir, categories, "test", max_workers)
    np.save(os.path.join(preprocessed_dir, "test.npy"), X_test)
    np.save(os.path.join(preprocessed_dir, "test_labels.npy"), y_test)

    # Print shapes of datasets
    logging.info(f"Train data shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    logging.info(f"Test data shape: {X_test.shape}, {y_test.shape}")
