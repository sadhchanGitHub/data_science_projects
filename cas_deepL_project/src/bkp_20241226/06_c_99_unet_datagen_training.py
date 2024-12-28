import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from unet_model_definition import unet_mode_detect_greenery  # Import U-Net model definition
from logging_config import setup_logging  # Import your logging setup function
import os
from tensorflow.keras.utils import to_categorical
import cv2


# Initialize logging
setup_logging(log_file_path="../logs/unet_training.log")

# Dataset paths
training_data_dir = "../data/training_data"
categories = ["AnnualCrop"]  # Use only one category for simplicity


def data_generator(images_path, masks_path, batch_size, target_size=(256, 256), num_classes=2):
    """
    Optimized generator to yield batches of images and masks for training.
    """
    images = np.load(images_path)
    masks = np.load(masks_path)

    while True:
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_masks = masks[i:i + batch_size]

            batch_images = np.array([cv2.resize(img, target_size) for img in batch_images])
            batch_masks = np.array([cv2.resize(mask, target_size) for mask in batch_masks])
            batch_masks = np.array([to_categorical(mask, num_classes=num_classes) for mask in batch_masks])

            yield batch_images, batch_masks


def main():
    try:
        # Parameters
        batch_size = 1
        epochs = 1
        samples_per_category = 50
        target_size = (256, 256)

        # File paths
        train_images_path = os.path.join(training_data_dir, "train", f"{categories[0]}_train.npy")
        train_masks_path = os.path.join(training_data_dir, "train", f"{categories[0]}_train_masks_combined.npy")
        val_images_path = os.path.join(training_data_dir, "val", f"{categories[0]}_val.npy")
        val_masks_path = os.path.join(training_data_dir, "val", f"{categories[0]}_val_masks_combined.npy")

        # Create generators
        train_gen = data_generator(train_images_path, train_masks_path, batch_size, target_size, num_classes=2)
        val_gen = data_generator(val_images_path, val_masks_path, batch_size, target_size, num_classes=2)

        # Initialize U-Net model
        logging.info("Initializing the U-Net model...")
        model = unet_mode_detect_greenery(input_size=(target_size[0], target_size[1], 3))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=logging.info)
        logging.info("U-Net model initialized successfully.")

        # Warm-up GPU before training
        logging.info("Warming up GPU...")
        dummy_input = np.random.random((1, target_size[0], target_size[1], 3))
        _ = model.predict(dummy_input)
        logging.info("GPU warm-up complete.")

        # Callbacks
        checkpoint_path = "unet_best_model.keras"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        csv_logger = CSVLogger('training_log.csv', append=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Start training
        logging.info("Starting model training...")
        steps_per_epoch = samples_per_category // batch_size
        validation_steps = max(1, steps_per_epoch // 5)  # Ensure at least 1 validation step

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=[checkpoint, csv_logger, early_stopping],
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
