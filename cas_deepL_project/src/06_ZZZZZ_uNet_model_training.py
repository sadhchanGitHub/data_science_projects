import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from datetime import datetime
from multiprocessing import Process
from gpu_logger import log_gpu_usage  # Import the GPU logging function

# Dataset paths
training_data_dir = "../data/training_data"

# Parameters
categories = ["AnnualCrop", "Forest", "Residential"]
batch_size = 8
target_shape = (256, 256)
num_classes = 2
epochs = 20
checkpoint_path = "unet_best_model.keras"

# Sample limits
total_samples = 6000  # Total samples to use across all categories
num_train_samples = int(total_samples * 0.8)  # 80% for training
num_val_samples = total_samples - num_train_samples  # 20% for validation

train_samples_per_category = num_train_samples // len(categories)  # Training samples per category
val_samples_per_category = num_val_samples // len(categories)  # Validation samples per category

steps_per_epoch = max(1, (len(categories) * train_samples_per_category) // batch_size)
validation_steps = max(1, (len(categories) * val_samples_per_category) // batch_size)

# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{int(datetime.now().timestamp())}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

def pad_batch(batch_images, batch_masks, batch_size):
    """Pad incomplete batches to match the batch size."""
    padding_needed = batch_size - len(batch_images)
    if padding_needed > 0:
        pad_image = np.zeros_like(batch_images[0])
        pad_mask = np.zeros_like(batch_masks[0])
        for _ in range(padding_needed):
            batch_images = np.append(batch_images, [pad_image], axis=0)
            batch_masks = np.append(batch_masks, [pad_mask], axis=0)
    return batch_images, batch_masks

def data_generator(data_dir, categories, split, batch_size, samples_per_category):
    """Generator that yields batches of images and masks for multiple categories and a specific split."""
    all_images = []
    all_masks = []

    for category in categories:
        image_file = os.path.join(data_dir, f"{category}_{split}.npy")
        mask_file = os.path.join(data_dir, f"{category}_{split}_masks_combined.npy")

        if not os.path.exists(image_file) or not os.path.exists(mask_file):
            logging.warning(f"Missing data for category '{category}' in '{split}' split. Skipping...")
            continue

        images = np.load(image_file)[:samples_per_category]
        masks = np.load(mask_file)[:samples_per_category]

        all_images.append(images)
        all_masks.append(masks)

        logging.info(f"Loaded {len(images)} samples for '{split}' split in category '{category}'.")

    if len(all_images) == 0 or len(all_masks) == 0:
        logging.error(f"No data available for '{split}' split. Exiting generator.")
        return

    all_images = np.concatenate(all_images, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    while True:
        for i in range(0, len(all_images), batch_size):
            batch_images = all_images[i : i + batch_size]
            batch_masks = all_masks[i : i + batch_size]

            # Pad incomplete batches
            batch_images, batch_masks = pad_batch(batch_images, batch_masks, batch_size)

            yield np.array(batch_images), np.array(batch_masks)

def unet_model(input_size=(256, 256, 3)):
    """U-Net model definition for binary segmentation."""
    inputs = Input(input_size)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c5)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

def main():
    try:
        # Training and Validation Generators
        train_gen = data_generator(
            data_dir=training_data_dir,
            categories=categories,
            split="train",
            batch_size=batch_size,
            samples_per_category=train_samples_per_category,
        )
        val_gen = data_generator(
            data_dir=training_data_dir,
            categories=categories,
            split="val",
            batch_size=batch_size,
            samples_per_category=val_samples_per_category,
        )

        logging.info(f"Data generators created with {steps_per_epoch} steps per epoch "
                     f"and {validation_steps} validation steps.")
    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

    try:
        # U-Net Model
        model = unet_model(input_size=(*target_shape, 3))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=logging.info)
        logging.info("U-Net model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the U-Net model: {e}")
        raise

    try:
        # Callbacks
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
        csv_logger = CSVLogger(os.path.join(log_dir, "training_log.csv"), append=True)
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

        # Train the Model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=[checkpoint, csv_logger, early_stopping],
            verbose=1,
        )
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
