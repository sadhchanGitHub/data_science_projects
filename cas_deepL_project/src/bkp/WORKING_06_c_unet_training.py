import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from unet_model_definition import unet_mode_detect_greenery  # Import U-Net model definition
from logging_config import setup_logging  # Import your logging setup function
import os

# Initialize logging
setup_logging(log_file_path="../logs/unet_training.log")

# Dataset paths
training_data_dir = "../data/training_data"

# Categories for segmentation
categories = ["AnnualCrop"]

# Function to load data
import cv2

from tensorflow.keras.utils import to_categorical

def load_data(file_path, target_size=(512, 512), num_samples=None, is_mask=False, num_classes=2):
    """
    Load and resize data to a standard size, adding a channel dimension for masks if needed.
    Args:
        file_path (str): Path to the `.npy` file.
        target_size (tuple): Desired size (height, width) for resizing.
        num_samples (int, optional): Number of samples to load.
        is_mask (bool): Whether the data is a mask (applies one-hot encoding for masks).
        num_classes (int): Number of classes for one-hot encoding.

    Returns:
        np.ndarray: Resized data.
    """
    try:
        data = np.load(file_path)
        if num_samples:
            data = data[:num_samples]

        resized_data = []
        for sample in data:
            resized_sample = cv2.resize(sample, target_size, interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
            if is_mask:
                # Convert masks to one-hot encoding
                resized_sample = to_categorical(resized_sample, num_classes=num_classes)
            resized_data.append(resized_sample)

        resized_data = np.array(resized_data)
        logging.info(f"Loaded and resized data from {file_path} with shape {resized_data.shape}")
        return resized_data
    except Exception as e:
        logging.error(f"Error loading or resizing data from {file_path}: {e}")
        raise




def main():
    try:
        # Parameters
        # Assuming total dataset size is 200
        total_samples = 200
        num_samples = int(total_samples * 0.8)  # 80% for training
        num_val_samples = total_samples - num_samples  # 20% for validation


        # Loading training data
        x_train, y_train = [], []
        for category in categories:
            logging.info(f"Loading data for category: {category}")

            # Load images
            train_images = load_data(
                os.path.join(training_data_dir, "train", f"{category}_train.npy"),
                target_size=(512, 512),
                num_samples=num_samples,
                is_mask=False  # Not a mask
            )
            x_train.append(train_images)

            # Load masks
            train_masks = load_data(
                os.path.join(training_data_dir, "train", f"{category}_train_masks_combined.npy"),
                target_size=(512, 512),
                num_samples=num_samples,
                is_mask=True  # Add channel dimension and one-hot encode
            )
            y_train.append(train_masks)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Loading validation data
        x_val, y_val = [], []
        for category in categories:
            val_images = load_data(
                os.path.join(training_data_dir, "val", f"{category}_val.npy"),
                target_size=(512, 512),
                num_samples=num_val_samples,
                is_mask=False
            )
            x_val.append(val_images)

            val_masks = load_data(
                os.path.join(training_data_dir, "val", f"{category}_val_masks_combined.npy"),
                target_size=(512, 512),
                num_samples=num_val_samples,
                is_mask=True  # Add channel dimension and one-hot encode
            )
            y_val.append(val_masks)

        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        logging.info(f"Loaded training data: Images={x_train.shape}, Masks={y_train.shape}")
        logging.info(f"Loaded validation data: Images={x_val.shape}, Masks={y_val.shape}")

        # Validation checks
        assert x_train.shape[1:3] == (512, 512), f"Training images are not resized to 512x512 but found {x_train.shape[1:3]}"
        assert x_val.shape[1:3] == (512, 512), f"Validation images are not resized to 512x512 but found {x_val.shape[1:3]}"
        assert y_train.shape[1:3] == (512, 512), f"Training masks are not resized to 512x512 but found {y_train.shape[1:3]}"
        assert y_val.shape[1:3] == (512, 512), f"Validation masks are not resized to 512x512 but found {y_val.shape[1:3]}"

    except Exception as e:
        logging.error(f"Error during dataset loading or preprocessing: {e}")
        raise

    # Initialize U-Net model
    try:
        logging.info("Initializing the U-Net model...")
        model = unet_mode_detect_greenery(input_size=(512, 512, 3))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=logging.info)
        logging.info("U-Net model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the U-Net model: {e}")
        raise

    # Setup callbacks
    try:
        logging.info("Setting up callbacks...")
        checkpoint_path = "unet_best_model.keras"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        csv_logger = CSVLogger('training_log.csv', append=True)
        logging.info("Callbacks setup complete.")
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise

    # Train the model
    try:
        logging.info("Starting model training...")
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,  # Adjust epochs as needed
            batch_size=2,  # Adjust batch size based on memory constraints
            callbacks=[checkpoint, csv_logger],
            verbose=1
        )
        logging.info(f"Training complete. Best model saved to {checkpoint_path}.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise


# Entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
