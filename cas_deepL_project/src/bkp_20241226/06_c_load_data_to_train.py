import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2  # Ensure OpenCV is installed

# Paths
preprocessed_output_dir = "../data/preprocessed_output_data"
training_data_dir = "../data/training_data"
categories = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Residential"]

# Split to test: Change to 'train', 'val', or 'test'
split = "train"

# Number of samples to visualize
num_samples = 5

# Function to load and sample data
def load_and_sample_data(preprocessed_dir, masks_dir, category, split, num_samples):
    # Load images
    image_path = os.path.join(preprocessed_dir, f"{category}_{split}.npy")
    mask_path = os.path.join(masks_dir, f"{category}_{split}_masks_combined.npy")
    
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"Missing files for {category} ({split}). Skipping...")
        return None, None
    
    # Load image and mask data
    images = np.load(image_path)
    masks = np.load(mask_path)
    
    print(f"{category} ({split}): Loaded {images.shape[0]} images and {masks.shape[0]} masks.")
    
    # Randomly sample indices
    sample_indices = random.sample(range(images.shape[0]), num_samples)
    sampled_images = images[sample_indices]
    sampled_masks = masks[sample_indices]
    
    return sampled_images, sampled_masks, sample_indices

# Visualize samples
def visualize_samples(images, masks, indices):
    for i, idx in enumerate(indices):
        # Get image and mask
        image = images[i]
        mask = masks[i]
        
        # Normalize image for visualization
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Decode mask to class labels
        mask_class = np.argmax(mask, axis=-1)
        
        # Resize mask to match image
        mask_resized = cv2.resize(
            mask_class.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Overlay mask on image
        overlay_image = image.copy()
        overlay_image[..., 0] = (
            overlay_image[..., 0] * 0.5 + mask_resized / mask_resized.max() * 255 * 0.5
        ).astype(np.uint8)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Original Image {idx}")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_resized, cmap="jet")
        plt.title(f"Mask {idx}")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay_image)
        plt.title(f"Overlay {idx}")
        plt.axis("off")
        
        plt.show()

# Main loop for categories
for category in categories:
    print(f"Processing category: {category} ({split})...")
    sampled_images, sampled_masks, indices = load_and_sample_data(
        preprocessed_output_dir, training_data_dir, category, split, num_samples
    )
    if sampled_images is not None and sampled_masks is not None:
        visualize_samples(sampled_images, sampled_masks, indices)


# Concatenate all categories for a split
X_train, Y_train = [], []
for category in categories:
    images, masks, _ = load_and_sample_data(preprocessed_output_dir, training_data_dir, category, "train", 0)
    if images is not None and masks is not None:
        X_train.append(images)
        Y_train.append(masks)

X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(Y_train, axis=0)
print(f"Final Train Dataset: Images={X_train.shape}, Masks={Y_train.shape}")


# Prepare combined datasets for training, validation, and testing
X_train, Y_train = [], []
X_val, Y_val = [], []
X_test, Y_test = [], []

# Combine all categories for each split
for category in categories:
    # Train split
    train_images, train_masks, _ = load_and_sample_data(preprocessed_output_dir, training_data_dir, category, "train", 0)
    if train_images is not None and train_masks is not None:
        X_train.append(train_images)
        Y_train.append(train_masks)
    
    # Validation split
    val_images, val_masks, _ = load_and_sample_data(preprocessed_output_dir, training_data_dir, category, "val", 0)
    if val_images is not None and val_masks is not None:
        X_val.append(val_images)
        Y_val.append(val_masks)
    
    # Test split
    test_images, test_masks, _ = load_and_sample_data(preprocessed_output_dir, training_data_dir, category, "test", 0)
    if test_images is not None and test_masks is not None:
        X_test.append(test_images)
        Y_test.append(test_masks)

# Concatenate data across all categories
X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(Y_train, axis=0)
X_val = np.concatenate(X_val, axis=0)
Y_val = np.concatenate(Y_val, axis=0)
X_test = np.concatenate(X_test, axis=0)
Y_test = np.concatenate(Y_test, axis=0)

print(f"Final Combined Dataset Shapes:")
print(f"  Train: Images={X_train.shape}, Masks={Y_train.shape}")
print(f"  Validation: Images={X_val.shape}, Masks={Y_val.shape}")
print(f"  Test: Images={X_test.shape}, Masks={Y_test.shape}")
