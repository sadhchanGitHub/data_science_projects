import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from unet_model_definition import unet_mode_detect_greenery  # in a separate file
from logging_config import setup_logging  # Import your logging setup function
from tensorflow.keras.callbacks import CSVLogger
import os

from tensorflow.keras.optimizers import Adam

model_sparsecatgent = unet_mode_detect_greenery(input_size=(513, 513, 3), num_classes=1)
# Compile the model
model_sparsecatgent.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)
model_sparsecatgent.summary()
