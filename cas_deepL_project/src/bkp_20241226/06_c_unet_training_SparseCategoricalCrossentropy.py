import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from unet_model_definition import unet_mode_detect_greenery  # Import U-Net model definition
from logging_config import setup_logging  # Import your logging setup function
import cv2
import os

# Initialize logging
setup_logging(log_file_path="../logs/unet_training.log")

# Dataset paths
training_data_dir = "../data/training_data"

# Categories for segmentation
categories = ["AnnualCrop", "Forest"]

# Function to load and preprocess data
def load_and_preprocess_data(file_path, target_shape, num_classes=None, num_samples=None):
    """
    Load data, resize masks to target shape, and preprocess for sparse categorical crossentropy.
    """
    try:
        data = np.load(file_path)
        if num_samples:
            data = data[:num_samples]
        logging.info(f"Loaded data from {file_path} with shape {data.shape}")

        # Resize masks or images
        resized_data = np.zeros((data.shape[0], target_shape[0], target_shape[1]), dtype=np.uint8)
        for i, item in enumerate(data):
            resized_item = cv2.resize(item, target_shape, interpolation=cv2.INTER_NEAREST)
            resized_data[i] = resized_item

        # For masks, ensure they are within the valid range of [0, num_classes-1]
        if "masks" in file_path and num_classes:
            resized_data = np.clip(resized_data, 0, num_classes - 1)

        return resized_data
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise



def main():
    try:
        # Parameters
        num_samples = 10  # Number of training samples
        num_val_samples = 10  # Number of validation samples
        target_shape = (512, 512)  # Target shape for resizing masks
        num_classes = 2  # Number of segmentation classes

        # Loading training data
        x_train, y_train = [], []
        for category in categories:
            logging.info(f"Loading data for category: {category}")

            # Load images
            train_images = load_and_preprocess_data(
                os.path.join(training_data_dir, "train", f"{category}_train.npy"),
                target_shape,
                num_classes=None,  # Not applicable for images
                num_samples=num_samples
            )
            x_train.append(train_images)

            # Load masks
            train_masks = load_and_preprocess_data(
                os.path.join(training_data_dir, "train", f"{category}_train_masks_combined.npy"),
                #os.path.join(training_data_dir, "train", f"{category}_train_masks_cleaned.npy"),
                target_shape,
                num_classes=num_classes,
                num_samples=num_samples
            )
            y_train.append(train_masks)

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Loading validation data
        x_val, y_val = [], []
        for category in categories:
            val_images = load_and_preprocess_data(
                os.path.join(training_data_dir, "val", f"{category}_val.npy"),
                target_shape,
                num_classes=None,
                num_samples=num_val_samples
            )
            x_val.append(val_images)

            val_masks = load_and_preprocess_data(
                os.path.join(training_data_dir, "val", f"{category}_val_masks_combined.npy"),
                # os.path.join(training_data_dir, "val", f"{category}_val_masks_cleaned.npy"),
                target_shape,
                num_classes=num_classes,
                num_samples=num_val_samples
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

        model_sparsecatgent = unet_mode_detect_greenery(input_size=(512, 512, 3), num_classes=1)
        # Compile the model
        model_sparsecatgent.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"]
        )
        model_sparsecatgent.summary(print_fn=logging.info)
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
