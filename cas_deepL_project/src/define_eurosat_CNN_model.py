import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.layers import Dropout
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import logging
import time
import sys

timestamp = int(time.time())


"""
# Define the CNN model
def create_cnn_eurosat_model(input_shape, num_classes):
    eurosat_cnn_model = Sequential([
        Input(shape=input_shape),  # Use Input layer explicitly
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    eurosat_cnn_model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

    return eurosat_cnn_model
"""

from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast


"""
def create_cnn_eurosat_model_with_augmentation(input_shape, num_classes):
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),
    ])
    
    eurosat_cnn_model_with_augmentation = Sequential([
        Input(shape=input_shape),
        
        # Apply data augmentation
        data_augmentation,
        
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Fully connected layers
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    
    # Compile the model
    eurosat_cnn_model_with_augmentation.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
    
    return eurosat_cnn_model_with_augmentation
"""

def create_cnn_eurosat_model_with_augmentation(input_shape, num_classes):
    # Data augmentation layer
    
    """
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),
    ])
    """
    
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),  # Adjust contrast for brightness-like effect
        RandomTranslation(0.2, 0.2),  # Correct usage for random translatio
    ])


    eurosat_cnn_model_with_augmentation = Sequential([
        Input(shape=input_shape),
        
        # Apply data augmentation
        data_augmentation,
        
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),  # Increased L2 regularization
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),  # Increased L2 regularization
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        
        # Fully connected layers
        Flatten(),
        #Dense(128, activation='relu', kernel_regularizer=l2(5e-4)),  # Increased L2 regularization
        Dense(256, activation='relu', kernel_regularizer=l2(5e-4)),  # Increased L2 regularization
        # Dropout(0.6),  # Increased Dropout from 0.5 to 0.6
        Dropout(0.7),  # Increased Dropout from 0.6 to 0.7
        Dense(num_classes, activation='softmax'),
    ])
    
    # Compile the model
    eurosat_cnn_model_with_augmentation.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
    
    return eurosat_cnn_model_with_augmentation



def create_cnn_eurosat_model(input_shape, num_classes):
    eurosat_cnn_model = Sequential([
        Input(shape=input_shape),
        #Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        #Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-4)),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        #Dropout(0.5),
        Dropout(0.7),
        Dense(num_classes, activation='softmax')
    ])

    eurosat_cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

    return eurosat_cnn_model



class CustomEarlyStopping(Callback):
    def __init__(self, patience=3):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_val_loss = np.Inf
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        print(f"Epoch {epoch + 1}: val_loss = {current_val_loss:.4f}, val_accuracy = {logs.get('val_accuracy'):.4f}")
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0  # Reset patience counter
        else:
            self.wait += 1
            print(f"No improvement in val_loss for {self.wait} consecutive epochs.")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"\nCustom Early Stopping triggered at epoch {self.stopped_epoch + 1}\n")



# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,  min_lr=1e-6 )
# Reduce learning rate by half # Wait for 3 epochs of no improvement # Minimum learning rate

# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
from tensorflow.keras.callbacks import LearningRateScheduler

def cyclical_lr(epoch, lr):
    base_lr = 1e-4
    max_lr = 5e-4
    step_size = 10
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr

lr_scheduler = LearningRateScheduler(cyclical_lr)


def train_cnn_eurosat_model(log_file, model_name, model, train_generator, steps_per_epoch, val_generator, validation_steps, batch_size=32, epochs=20):
    # Set up logging to a file
    # log_filename = f'../logs/training_log_{int(time.time())}.log'
    sys.stdout = open(log_file, 'w')  # Redirect stdout to the log file

    try:
        # Add logging for the start of training
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logging.info(f"\nTraining Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'../models/{model}_{timestamp}.keras', save_best_only=True, monitor='val_loss')
        #custom_early_stopping = CustomEarlyStopping()
        custom_early_stopping = CustomEarlyStopping(patience=3)

        
        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,  # Corrected keyword
            validation_steps=validation_steps,
            epochs=epochs,
            #callbacks=[early_stopping, model_checkpoint, custom_early_stopping]
            #callbacks = [custom_early_stopping, model_checkpoint]
            
            #learning rate is added
            callbacks = [early_stopping, model_checkpoint, lr_scheduler]

        )
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logging.info(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    finally:
        # Ensure stdout is reset even if an error occurs
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        # print(f"Training log saved to: {log_filename}")
        logging.info(f"Training log saved to: {log_file}.")#
    
    return history




# Visualize training progress
def visualize_eurosat_cnn_model_training(history, timestamp):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../logs/eurosat_cnn_training_with_augmentation_progress_{timestamp}.png')
    logging.info(f"with_augmentation Training progress visualized and saved as 'eurosat_training_progress_with_augmentation_{timestamp}.png'.")
