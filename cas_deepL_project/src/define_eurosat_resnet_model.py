import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import logging
import time
import sys

timestamp = int(time.time())


# Define input shape and number of classes
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 6  # Adjust according to your dataset

def get_data_generators(training_dir, batch_size):
    # Define the data generator with augmentation

    """
    train_datagen = ImageDataGenerator(
        #rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    """
    
    train_datagen = ImageDataGenerator(
        #rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )


    # Create the training generator
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        #class_mode='categorical',
        class_mode='sparse',
        subset='training'
    )

    # Create the validation generator
    val_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        #class_mode='categorical',
        class_mode='sparse',
        subset='validation'
    )

    return train_generator, val_generator


def eurosat_resnet_model_with_augmentation(input_shape, num_classes):
        
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    
    # Freeze the base model layers
    base_model.trainable = False

    # Define the model
    resnet_model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return resnet_model

from tensorflow.keras.regularizers import l2
def eurosat_resnet_model_with_augmentation_l2(input_shape, num_classes):
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    base_model.trainable = False

    # Define the model
    resnet_model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Added L2 regularization
        Dropout(0.5),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))  # L2 on output layer
    ])
    
    # Compile the model
    resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    
    return resnet_model

class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        self.log_file = log_file
    
    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1} - {logs}\n")
        logging.info(f"Epoch {epoch + 1} - {logs}")

def get_callbacks(log_file):
    return [
        #EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('resnet50_eurosat_best.keras', save_best_only=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        LoggingCallback(log_file)
    ]



def train_eurosat_resnet_model(log_file, model_name, model, train_generator, steps_per_epoch, val_generator, validation_steps, epochs=40):
    logging.info(f"Starting training for model without finetuning: {model_name}")
    callbacks = get_callbacks(log_file)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )
    logging.info(f"Finished training for model without finetuning: {model_name} \n")

    # Fine-tuning
    model.layers[0].trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info(f"Starting training for model WITH finetuning: {model_name}")
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=20,
        callbacks=callbacks
    )
    logging.info(f"Finished training for model WITH finetuning: {model_name} \n")

    return history, history_fine


    
# Plot training history
def visualize_eurosat_resnet_model_training(resnet_model_name, history_var, fine_tuning, timestamp):
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history_var.history['accuracy'], label='Train Accuracy')
    plt.plot(history_var.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history_var.history['loss'], label='Train Loss')
    plt.plot(history_var.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    suffix = '_fine_tuning' if fine_tuning else ''
    plt.savefig(f'../outputs/{resnet_model_name}_training_progress_{suffix}_{timestamp}.png')
    plt.savefig(f'../outputs/{resnet_model_name}_training_progress_{timestamp}.png')
    logging.info(f"saved the ../outputs/eurosat_resnet_training_with_augmentation_progress_{suffix}_{timestamp}'.png file")


