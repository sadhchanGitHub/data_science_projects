import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from unet_model_definition import unet_mode_detect_greenery  # in a separate file
from logging_config import setup_logging  # Import your logging setup function
from tensorflow.keras.callbacks import CSVLogger
import os

from tensorflow.keras.optimizers import Adam

# Compile the model
model = unet_mode_detect_greenery(input_size=(513, 513, 3), num_classes=1)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
