import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import time
import os

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

timestamp = int(time.time())


#import modules
from define_eurosat_CNN_model import create_cnn_eurosat_model, train_cnn_eurosat_model, visualize_eurosat_cnn_model_training


print("\n")

print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


# Configure GPU settings (restrict TensorFlow to use only the first GPU)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
        print("GPU configured successfully.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

        

# Load train, val, and test data
training_dir = "../data/training_data"
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
#categories = ["Forest"]

def classification_data_generator(training_dir, categories, split, batch_size, samples_per_category):
    """
    Generator for loading images and labels in batches.
    """
    image_files = []
    label_files = []

    # Collect image and label file paths for all categories
    for category in categories:
        image_files += [os.path.join(training_dir, f"{category}_{split}.npy")]
        label_files += [os.path.join(training_dir, f"{category}_{split}_labels_one_hot.npy")]

    while True:
        for i in range(0, len(image_files)):
            images = np.load(image_files[i])[:samples_per_category]
            labels = np.load(label_files[i])[:samples_per_category]

            # Yield data in batches
            for j in range(0, len(images), batch_size):
                batch_images = images[j:j + batch_size]
                batch_labels = labels[j:j + batch_size]
                yield np.array(batch_images), np.array(batch_labels)


batch_size = 32


# Sample limits
total_samples = 20000  # Total samples to use across all categories
num_train_samples = int(total_samples * 0.8)  # 80% for training
num_val_samples = total_samples - num_train_samples  # 20% for validation

print(f"total_samples: {total_samples}, num_train_samples: {num_train_samples}, num_val_samples: {num_val_samples} \n ")

train_samples_per_category = num_train_samples // len(categories)  # Training samples per category
val_samples_per_category = num_val_samples // len(categories)  # Validation samples per category
print(f"train_samples_per_category: {train_samples_per_category}, val_samples_per_category: {val_samples_per_category} \n ")


# Total training and validation samples
total_train_samples = sum([train_samples_per_category for _ in categories])
total_val_samples = sum([val_samples_per_category for _ in categories])
print(f"total_train_samples: {total_train_samples}, total_val_samples: {total_val_samples} \n ")


# Steps per epoch
steps_per_epoch = max(1, total_train_samples // batch_size)
validation_steps = max(1, total_val_samples // batch_size)

print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}\n")


train_generator = classification_data_generator(training_dir, categories, split="train", batch_size=batch_size,samples_per_category=train_samples_per_category)
val_generator = classification_data_generator(training_dir, categories, split="val", batch_size=batch_size,samples_per_category=val_samples_per_category)

batch_images, batch_labels = next(train_generator)

print(f"Batch images shape: {batch_images.shape}")
print(f"Batch labels shape: {batch_labels.shape}")




# Model parameters
input_shape = (256, 256, 3)  # Shape of input images
num_classes = 6  # Number of categories

# Create and compile the model
model = create_cnn_eurosat_model(input_shape, num_classes)


# Print model summary
# model.summary()


# Train the model
history = train_cnn_eurosat_model(model, train_generator, steps_per_epoch, val_generator, validation_steps)
logging.info("Model training completed.")

# Save full model
model.save(f'../models/eurosat_cnn_model_{timestamp}.keras')
logging.info(f"Full model saved to: ../models/eurosat_cnn_model_{timestamp}.keras")
print(f"Full model saved to: ../models/eurosat_cnn_model_{timestamp}.keras")

# Save weights
model.save_weights(f'../models/eurosat_cnn_weights_{timestamp}.weights.h5')
logging.info(f"Model weights saved to: ../models/eurosat_cnn_weights_{timestamp}.weights.h5")
print(f"Model weights saved to: ../models/eurosat_cnn_weights_{timestamp}.weights.h5")


# Visualize training progress
visualize_eurosat_cnn_model_training(history, timestamp)
logging.info("Training progress visualization completed.")

print(f"Ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


    
