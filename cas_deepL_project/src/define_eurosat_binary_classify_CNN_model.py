import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.layers import RandomBrightness, RandomTranslation, Activation
from tensorflow.keras.layers import SpatialDropout2D
import os

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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

timestamp = int(time.time())

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


"""
def create_cnn_eurosat_binary_classify(input_shape, num_classes):
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2)
    ])
    cnn_eurosat_binary_classify = Sequential([
    Input(shape=input_shape),
    data_augmentation,
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(5e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.7),
    Dense(1, activation='sigmoid')  # Binary output
    ])

    cnn_eurosat_binary_classify.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return cnn_eurosat_binary_classify
"""

def create_cnn_eurosat_binary_classify(input_shape, num_classes):
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.1),  # Reduced from 0.2
        RandomZoom(0.1),      # Reduced from 0.2
        RandomContrast(0.1),  # Reduced from 0.2
        RandomTranslation(0.05, 0.05)  # Reduced translation
    ])

    cnn_eurosat_binary_classify = Sequential([
        Input(shape=input_shape),
        data_augmentation,
        Conv2D(32, (3, 3), kernel_regularizer=l2(5e-4)),
        BatchNormalization(),

        
        Activation('relu'),
        MaxPooling2D((2, 2)),
        # Commented out SpatialDropout2D
        # SpatialDropout2D(0.2, data_format='channels_last'),

        Conv2D(64, (3, 3), kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dropout(0.6),
        Dense(1, activation='sigmoid')  # Binary output
    ])

    cnn_eurosat_binary_classify.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    return cnn_eurosat_binary_classify


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



from tensorflow.keras.callbacks import LearningRateScheduler
"""
def cyclical_lr(epoch, lr):
    base_lr = 1e-4
    max_lr = 5e-4
    step_size = 10
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr
    lr_scheduler = LearningRateScheduler(cyclical_lr)
"""

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)



def train_cnn_eurosat_binary_classify(model_name, model, train_generator, steps_per_epoch, val_generator, validation_steps, batch_size=32, epochs=20):
    
    #set up logfile to track training epochs etc...
    tr_log_file = f'../logs/06_a_execute_{model_name}_training_epochs_log_{timestamp}.log'
    sys.stdout = open(tr_log_file, 'w')  # Redirect stdout to the log file


    try:
        # Add logging for the start of training
        logging.info(f"\n {model_name} Training Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
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
        logging.info(f"\n {model_name} Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    finally:
        # Ensure stdout is reset even if an error occurs
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        # print(f"Training log saved to: {log_filename}")
        logging.info(f" {model_name} Training log saved to: {tr_log_file}.")#
    
    return history

def main_visualizations(history, timestamp, model_name, model, val_images, val_labels):
    """
    Orchestrates all visualization tasks: metrics, confusion matrix, and misclassified samples.
    """
    try:
        # Visualize training metrics
        visualize_cnn_eurosat_binary_classify_training_metrics(history, timestamp, model_name)
        logging.info(f"{model_name} Training metrics visualization completed.")

        # Generate predictions
        preds = model.predict(val_images)
        preds = (preds > 0.5).astype("int32")  # Binary classification threshold

        # Visualize confusion matrix
        visualize_cnn_eurosat_binary_classify_training_confusionmatrix(preds, val_labels, timestamp, model_name)
        logging.info(f"{model_name} Confusion matrix visualization completed.")

        # Save misclassified samples
        cnn_eurosat_binary_classify_chk_misclassified_indices(preds, val_images, val_labels, timestamp, model_name)
        logging.info(f"{model_name} Misclassified samples visualization completed.")

        # Save true classified samples
        cnn_eurosat_binary_classify_chk_trueclassified_indices(preds, val_images, val_labels, timestamp, model_name)

        logging.info(f"{model_name} Trueclassified samples visualization completed.")

    except Exception as e:
        logging.error(f"Error during {model_name} visualizations: {e}")
        raise

def cnn_eurosat_binary_classify_chk_misclassified_indices(preds, val_images, val_labels, timestamp, model_name):
    """
    Saves misclassified images to the output directory.
    """
    output_dir = "../outputs/misclassified_samples"
    os.makedirs(output_dir, exist_ok=True)

    misclassified_indices = np.where(preds.reshape(-1) != val_labels)[0]
    
    if len(misclassified_indices) == 0:
        logging.warning("No correctly mis-classified samples found.")
        return
    
    for i, idx in enumerate(misclassified_indices[:5]):
        
        # Clear previous figures
        plt.clf()
        plt.cla()
        plt.close('all')
        
        false_fig = plt.figure(figsize=(10, 8)) 
        false_image = (val_images[idx] * 255).astype('uint8')
        plt.imshow(false_image)
        plt.title(f"Misclassified: True: {val_labels[idx]}, Pred: {preds[idx][0]}")
        plt.axis('off')
        false_fig.savefig(f"{output_dir}/misclassified_{i}.png")
        plt.close(false_fig)
    
    logging.info(f"Misclassified images saved to: {output_dir}")


def cnn_eurosat_binary_classify_chk_trueclassified_indices(preds, val_images, val_labels, timestamp, model_name):
    """
    Saves correctly classified images to the output directory.
    """
    output_dir = "../outputs/trueclassified_samples"
    os.makedirs(output_dir, exist_ok=True)

    trueclassified_indices = np.where(preds.reshape(-1) == val_labels)[0]
    
    if len(trueclassified_indices) == 0:
        logging.warning("No correctly classified samples found.")
        return

    for i, idx in enumerate(trueclassified_indices[:5]):
        
        # Clear previous figures
        plt.clf()
        plt.cla()
        plt.close('all')
  
        true_fig = plt.figure(figsize=(10, 8)) 
        true_image = (val_images[idx] * 255).astype('uint8')
        plt.imshow(true_image)
        plt.title(f"Correctly Classified: True: {val_labels[idx]}, Pred: {preds[idx][0]}")
        plt.axis('off')
        true_fig.savefig(f"{output_dir}/trueclassified_{i}.png")
        plt.close(true_fig)
    
    logging.info(f"Correctly classified images saved to: {output_dir}")



def visualize_cnn_eurosat_binary_classify_training_metrics(history, timestamp, model_name):
    # Clear previous figures
    plt.clf()
    plt.cla()
    plt.close('all')
    
    acc_loss_fig = plt.figure(figsize=(18, 6))  # Create a new figure with the correct size
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Precision and Recall plot
    if 'precision' in history.history and 'val_precision' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['precision'], label='Train Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.plot(history.history['recall'], label='Train Recall')
        plt.plot(history.history['val_recall'], label='Validation Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
    else:
        logging.warning("Precision and Recall metrics not found in training history.")
    
    plt.tight_layout()
    acc_loss_fig.savefig(f'../logs/06_a_execute_{model_name}_training_metrics_{timestamp}.png')
    plt.close(acc_loss_fig)  # Close the specific figure
    logging.info(f"Training metrics visualized and saved.")


def visualize_cnn_eurosat_binary_classify_training_confusionmatrix(preds, val_labels, timestamp, model_name):

    cm = confusion_matrix(val_labels, preds)
    class_names = ['Residential', 'HerbaceousVegetation']
    
    # Clear previous figures
    plt.clf()
    plt.cla()
    plt.close('all')
    
    cfmatrix_fig = plt.figure(figsize=(10, 8))  # Adjust size if needed
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cfmatrix_fig.savefig(f'../logs/06_a_execute_{model_name}_training_confusion_matrix_{timestamp}.png')
    plt.close(cfmatrix_fig)
    logging.info("Confusion matrix saved.")


