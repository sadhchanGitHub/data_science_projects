import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Dataset paths
training_data_dir = "../data/training_data"

# Parameters
categories = ["AnnualCrop"]
batch_size = 8
target_shape = (256, 256)  # Images are already sized
num_classes = 2
epochs = 2
checkpoint_path = "unet_best_model.keras"
sample_limit = 100  # Limit the number of samples for training and validation

def data_generator(training_data_dir, category, split, batch_size, sample_limit):
    """
    Generator that yields batches of images and binary masks for a specific category and split.
    Skips batches where all masks are empty.
    Limits the total number of samples used.
    """
    image_file = os.path.join(training_data_dir, f"{category}_{split}.npy")
    mask_file = os.path.join(training_data_dir, f"{category}_{split}_masks_combined.npy")

    # Load the images and masks
    images = np.load(image_file)[:sample_limit]
    masks = np.load(mask_file)[:sample_limit]

    assert len(images) == len(masks), "Mismatch between number of images and masks."

    logging.info(f"Loaded {len(images)} images and masks for {split} split.")
    skipped_samples = 0

    while True:
        for i in range(0, len(images), batch_size):
            batch_images = []
            batch_masks = []
            for j in range(i, min(i + batch_size, len(images))):
                image = images[j]
                mask = masks[j]

                if np.max(mask) == 0:
                    logging.info(f"Skipping empty mask at index {j}")
                    skipped_samples += 1
                    continue

                batch_images.append(image)
                batch_masks.append(mask)

            if len(batch_images) > 0:
                yield np.expand_dims(np.array(batch_images), axis=-1), np.expand_dims(np.array(batch_masks), axis=-1)

        logging.info(f"Total skipped samples for {split}: {skipped_samples}")



def unet_model(input_size=(256, 256, 3)):
    """
    Define a U-Net model for binary segmentation.
    """
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = UpSampling2D((2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c5)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model



class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_epochs, total_epochs, verbose=1):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            new_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            new_lr = self.initial_lr * tf.math.exp(-0.1 * (epoch - self.warmup_epochs))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        if self.verbose:
            print(f"\nEpoch {epoch + 1}: Learning rate is {new_lr:.6f}.")



def main():
    try:
        train_gen = data_generator(training_data_dir, categories[0], "train", batch_size, sample_limit)
        val_gen = data_generator(training_data_dir, categories[0], "val", batch_size, sample_limit)

        logging.info("Data generators initialized with sampling.")
    except Exception as e:
        logging.error(f"Error during dataset preparation: {e}")
        raise

    try:
        logging.info("Initializing the U-Net model...")
        model = unet_model(input_size=(*target_shape, 3))
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary(print_fn=logging.info)
        logging.info("U-Net model initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the U-Net model: {e}")
        raise

    try:
        logging.info("Setting up callbacks...")
        initial_lr = 0.001
        warmup_epochs = 5
        lr_scheduler = WarmUpLearningRateScheduler(initial_lr=initial_lr, warmup_epochs=warmup_epochs, total_epochs=epochs)

        checkpoint_path = os.path.join(os.getcwd(), "unet_best_model.keras")
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
        logging.info(f"Resolved ModelCheckpoint filepath: {checkpoint_path}")


        csv_logger = CSVLogger("training_log.csv", append=True)
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        callbacks = [checkpoint]
        logging.info("Callbacks setup complete.")
    except Exception as e:
        logging.error(f"Error setting up callbacks: {e}")
        raise

    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=max(1, sample_limit // batch_size),
            validation_steps=max(1, sample_limit // batch_size // 2),
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error(f"Type of error object: {type(e)}")
        raise



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")

