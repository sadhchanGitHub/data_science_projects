import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from datetime import datetime
import time
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
sample_limit = 3000  # Maximum samples per category for training and validation

# Ensure logs are written to a unique file in the logs directory
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{int(time.time())}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file)],
)


def data_generator(training_data_dir, categories, split, batch_size, sample_limit_per_category):
    """
    Generator that yields batches of images and masks for multiple categories and a specific split.
    Skips batches where all masks are empty.
    Limits the total number of samples used per category.
    Logs skipped samples and category-specific statistics.
    """
    # Load images and masks for each category
    all_images = []
    all_masks = []


    category_skipped_samples = {category: 0 for category in categories}  # Track skipped samples per category

    for category in categories:
        image_file = os.path.join(training_data_dir, f"{category}_{split}.npy")
        mask_file = os.path.join(training_data_dir, f"{category}_{split}_masks_combined.npy")

        if not os.path.exists(image_file) or not os.path.exists(mask_file):
            logging.warning(f"Missing data for category '{category}' in '{split}' split. Skipping...")
            continue

        images = np.load(image_file)[:sample_limit_per_category]
        masks = np.load(mask_file)[:sample_limit_per_category]

        if len(images) != len(masks):
            logging.warning(
                f"Mismatch detected for category '{category}' in '{split}' split: "
                f"{len(images)} images, {len(masks)} masks. Using the minimum count."
            )
            min_samples = min(len(images), len(masks))
            images = images[:min_samples]
            masks = masks[:min_samples]

        logging.info(f"Loaded {len(images)} samples for '{split}' split in category '{category}'.")
        all_images.append(images)
        all_masks.append(masks)



    all_images = np.concatenate(all_images, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    logging.info(
        f"Total loaded data for '{split}' split: {len(all_images)} images and masks across categories {categories}."
    )
    skipped_samples = 0
    total_batches = (len(all_images) + batch_size - 1) // batch_size  # Calculate total batches

    while True:
        for i in range(0, len(all_images), batch_size):
            batch_images = []
            batch_masks = []
            for j in range(i, min(i + batch_size, len(all_images))):
                image = all_images[j]
                mask = all_masks[j]

                if np.max(mask) == 0:  # Skip empty masks
                    skipped_samples += 1
                    continue

                batch_images.append(image)
                batch_masks.append(mask)

            # if len(batch_images) > 0:
            #     logging.info(
            #         f"Generated batch {i // batch_size + 1}/{total_batches} for '{split}' split "
            #         f"with {len(batch_images)} samples (Skipped so far: {skipped_samples})."
            #     )
                yield np.array(batch_images), np.array(batch_masks)

        logging.info(
            f"End of '{split}' split for categories {categories}. Total skipped samples: {skipped_samples}"
        )
        skipped_samples = 0  # Reset for next loop


def calculate_steps(training_data_dir, categories, split, batch_size, sample_limit=None):
    """
    Calculate steps for training or validation based on available samples and batch size.
    """
    total_samples = 0
    for category in categories:
        image_file = os.path.join(training_data_dir, f"{category}_{split}.npy")
        mask_file = os.path.join(training_data_dir, f"{category}_{split}_masks_combined.npy")

        if not os.path.exists(image_file) or not os.path.exists(mask_file):
            logging.warning(f"Missing data for category '{category}' in '{split}' split. Skipping...")
            continue

        images = np.load(image_file)
        masks = np.load(mask_file)

        # Handle mismatched images and masks
        min_samples = min(len(images), len(masks))
        if sample_limit is not None:
            min_samples = min(min_samples, sample_limit)

        total_samples += min_samples

    return max(1, total_samples // batch_size)


def unet_model(input_size=(256, 256, 3)):
    """
    U-Net model definition for binary segmentation.
    """
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
        steps_per_epoch = calculate_steps(training_data_dir, categories, "train", batch_size, sample_limit)
        validation_steps = calculate_steps(training_data_dir, categories, "val", batch_size, sample_limit)

        train_gen = data_generator(training_data_dir, categories, "train", batch_size, sample_limit)
        val_gen = data_generator(training_data_dir, categories, "val", batch_size, sample_limit)

        logging.info("Data generators and steps calculated successfully.")
    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

    try:
        model = unet_model(input_size=(*target_shape, 3))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary(print_fn=logging.info)
        logging.info("U-Net model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the U-Net model: {e}")
        raise

    try:
        # Callbacks setup
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
        csv_logger = CSVLogger(os.path.join(log_dir, "training_log.csv"), append=True)
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
        callbacks = [checkpoint, csv_logger, early_stopping, lr_scheduler]

        gpu_logger = Process(target=log_gpu_usage, args=(log_dir, 30))
        gpu_logger.start()
        logging.info("GPU logging process started.")

        try:
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
        finally:
            if gpu_logger.is_alive():
                gpu_logger.terminate()
                gpu_logger.join()
                logging.info("GPU logging stopped.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
