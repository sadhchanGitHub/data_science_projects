import os
import numpy as np
import dask.array as da

# Function to split data incrementally
def split_data_dask(X, y, test_size, val_size):
    total_samples = X.shape[0]
    test_samples = int(total_samples * test_size)
    val_samples = int((total_samples - test_samples) * val_size)

    indices = np.random.permutation(total_samples)
    test_indices = indices[:test_samples]
    val_indices = indices[test_samples:test_samples + val_samples]
    train_indices = indices[test_samples + val_samples:]

    return (
        X[test_indices], y[test_indices],
        X[val_indices], y[val_indices],
        X[train_indices], y[train_indices]
    )

# Process and save splits incrementally
# Process and save splits incrementally
def process_and_save_category(preprocessed_outdir, category, test_size=0.2, val_size=0.2):
    file_path = os.path.join(preprocessed_outdir, f"{category}.npy")
    print(f"Processing {category}...")
    
    # Load combined `.npy` file as Dask array
    images = da.from_array(np.load(file_path, mmap_mode='r'), chunks="auto")
    labels = da.full((images.shape[0],), categories.index(category), chunks="auto")
    
    # Split data
    X_test, y_test, X_val, y_val, X_train, y_train = split_data_dask(images, labels, test_size, val_size)

    # Save each split as a single `.npy` file
    for split_name, split_data, split_labels in [
        ("train", X_train, y_train), 
        ("val", X_val, y_val), 
        ("test", X_test, y_test)
    ]:
        output_data_path = os.path.join(preprocessed_outdir, f"{category}_{split_name}.npy")
        output_label_path = os.path.join(preprocessed_outdir, f"{category}_{split_name}_labels.npy")

        # Convert Dask arrays to NumPy arrays and save
        np.save(output_data_path, split_data.compute())
        np.save(output_label_path, split_labels.compute())

        print(f"Saved {split_name} split for {category} to {output_data_path}")


# Paths
preprocessed_outdir = "../data/preprocessed_output_data"
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Process each category
for category in categories:
    process_and_save_category(preprocessed_outdir, category)
