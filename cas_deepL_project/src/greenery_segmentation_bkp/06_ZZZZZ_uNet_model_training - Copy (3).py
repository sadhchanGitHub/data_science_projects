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

# Corrected Parameters
total_samples = 9000  # Total samples across all categories

num_train_samples = int(total_samples * 0.8)  # 80% for training
num_val_samples = total_samples - num_train_samples  # 20% for validation

train_samples_per_category = num_train_samples // len(categories)  # Training samples per category
val_samples_per_category = num_val_samples // len(categories)  # Validation samples per category

# Corrected Steps
steps_per_epoch = max(1, (len(categories) * train_samples_per_category) // batch_size)
validation_steps = max(1, (len(categories) * val_samples_per_category) // batch_size)


# Ensure logs are written to a unique file in the logs directory
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{int(time.time())}.log")

# Configure logging to write to a file and suppress terminal output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Write logs to file
        # Remove StreamHandler to suppress terminal output
    ]
)



# Additional logger for skipped masks
# skipped_mask_log_file = os.path.join(log_dir, f"skipped_masks_{int(time.time())}.log")
# skipped_mask_logger = logging.getLogger("skipped_masks")
# skipped_mask_logger.setLevel(logging.INFO)
# skipped_mask_logger.addHandler(logging.FileHandler(skipped_mask_log_file))


def data_generator(training_data_dir, categories, split, batch_size, sample_limit_per_category):
    """
    Generator that yields batches of images and masks for multiple categories and a specific split.
    Skips batches where all masks are empty.
    Limits the total number of samples used per category.
    """
    # Load images and masks for each category
    all_images = []
    all_masks = []
    for category in categories:
        image_file = os.path.join(training_data_dir, f"{category}_{split}.npy")
        mask_file = os.path.join(training_data_dir, f"{category}_{split}_masks_combined.npy")

        images = np.load(image_file)[:sample_limit_per_category]
        masks = np.load(mask_file)[:sample_limit_per_category]

        assert len(images) == len(masks), f"Mismatch between number of images and masks for {category}."

        logging.info(f"Loaded {len(images)} images and masks for {split} split for {category}.")
        all_images.append(images)
        all_masks.append(masks)

    all_images = np.concatenate(all_images, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    logging.info(f"Total loaded data for {split} split: {len(all_images)} images and masks across categories {categories}.")
    skipped_samples = 0

    while True:
        for i in range(0, len(all_images), batch_size):
            batch_images = []
            batch_masks = []
            for j in range(i, min(i + batch_size, len(all_images))):
                image = all_images[j]
                mask = all_masks[j]

                if np.max(mask) == 0:
                    # skipped_mask_logger.info(f"Skipping empty mask for {category}  at index {j} in {split} split across categories.")
                    skipped_samples += 1
                    continue

                batch_images.append(image)
                batch_masks.append(mask)

            if len(batch_images) > 0:
                yield np.array(batch_images), np.array(batch_masks)

        logging.info(f"Total skipped samples for {split} split across categories {categories}: {skipped_samples}")



# Example U-Net model definition (unchanged for binary segmentation)
def unet_model(input_size=(256, 256, 3)):
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
        train_gen = data_generator(
            training_data_dir, categories, "train", batch_size, train_samples_per_category
        )
        val_gen = data_generator(
            training_data_dir, categories, "val", batch_size, val_samples_per_category
        )
        logging.info("Data generators initialized with sampling.")
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
        # Define callbacks
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
        
        training_epoch_accuracy_log = os.path.join(log_dir, f"training_epoch_accuracy_{int(time.time())}.csv")
        csv_logger = CSVLogger(training_epoch_accuracy_log, append=True)
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    
        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
    
        # Combine all callbacks
        callbacks = [checkpoint, csv_logger, early_stopping, lr_scheduler]
    
        logging.info("Callbacks setup complete.")
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise

    

    gpu_logger = Process(target=log_gpu_usage, args=(log_dir, 30))  # Log every 30 seconds
    
    gpu_logger.start()
    print("GPU logger process started.")  # Debugging message
    
    # Check if the GPU logger started successfully
    if not gpu_logger.is_alive():
        print("GPU logger process failed to start or stopped unexpectedly.")
        # Optionally handle the issue (e.g., exit or continue without logging)
        gpu_logger.terminate()  # Ensure the process is cleaned up
        gpu_logger.join()
    
    try:
        # Model training process
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
        # Stop GPU logging after training
        if gpu_logger.is_alive():
            print(f"Is GPU logger alive? {gpu_logger.is_alive()}")
            gpu_logger.terminate()
            gpu_logger.join()
            print("GPU logging stopped.")

    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
