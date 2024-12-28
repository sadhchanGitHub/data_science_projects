import dask.array as da
import numpy as np
import cv2
import os
import tensorflow as tf

def resize_and_encode(img, mask, target_size, num_classes):
    """
    Resize and one-hot encode a single image and its mask.
    """
    # Ensure img and mask are NumPy arrays
    img = np.array(img)
    mask = np.array(mask)

    # Convert to supported data type
    if img.dtype == 'object':
        img = img.astype(np.float32)
    if mask.dtype == 'object':
        mask = mask.astype(np.float32)

    # Resize and encode
    resized_img = cv2.resize(img, target_size)
    resized_mask = cv2.resize(mask, target_size)
    one_hot_mask = tf.keras.utils.to_categorical(resized_mask, num_classes=num_classes)
    return resized_img, one_hot_mask

def preprocess_with_dask(input_images_path, input_masks_path, output_dir, target_size=(256, 256), num_classes=2):
    """
    Preprocess images and masks using Dask for chunked processing.
    """
    # Load images and masks as Dask arrays
    images = da.from_array(np.load(input_images_path), chunks=(100, None, None, None))  # Process in chunks
    masks = da.from_array(np.load(input_masks_path), chunks=(100, None, None))          # Process in chunks

    # Lists to store results
    resized_images = []
    resized_masks = []

    # Process chunks
    for img_chunk, mask_chunk in zip(images.to_delayed(), masks.to_delayed()):
        # Compute chunk (convert delayed object to NumPy arrays)
        img_chunk = da.compute(img_chunk)[0]
        mask_chunk = da.compute(mask_chunk)[0]

        # Process each image-mask pair in the chunk
        for img, mask in zip(img_chunk, mask_chunk):
            resized_img, one_hot_mask = resize_and_encode(img, mask, target_size, num_classes)
            resized_images.append(resized_img)
            resized_masks.append(one_hot_mask)

    # Convert results to Dask arrays
    resized_images = da.from_array(np.array(resized_images), chunks=(100, None, None, None))
    resized_masks = da.from_array(np.array(resized_masks), chunks=(100, None, None, None))

    # Save results to disk
    os.makedirs(output_dir, exist_ok=True)
    da.to_npy_stack(os.path.join(output_dir, "AnnualCrop_train_resized"), resized_images)
    da.to_npy_stack(os.path.join(output_dir, "AnnualCrop_train_masks_resized"), resized_masks)

    print(f"Saved resized images and masks to {output_dir}")


# Call the function
preprocess_with_dask(
    input_images_path="../data/training_data/train/AnnualCrop_train.npy",
    input_masks_path="../data/training_data/train/AnnualCrop_train_masks_combined.npy",
    output_dir="../data/training_downsizedata_256/train",
    target_size=(256, 256)
)
