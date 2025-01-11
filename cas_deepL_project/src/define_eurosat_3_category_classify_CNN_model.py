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
from sklearn.metrics import classification_report, confusion_matrix

timestamp = int(time.time())

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

# The model outputs probabilities for 3 classes: 0, 1, and 2.
# Original labels (1, 4, 5) do not align with these class indices.
# Custom loss function remaps labels as follows:
# - 1 → 0 (Residential)
# - 4 → 1 (Industrial)
# - 5 → 2 (HerbaceousVegetation)
# This ensures the labels match the model's output indices, avoiding errors during training.


def custom_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.squeeze(y_true)  # Ensure correct shape
    y_true = tf.ensure_shape(y_true, [None])  # Explicitly define shape
    y_true = tf.cast(y_true, tf.int32)  # Ensure integer type
    y_true = tf.where(y_true == 1, 0, tf.where(y_true == 4, 1, 2))  # Remap labels

    # Log the remapped labels for debugging, only 5
    # tf.print("Sample remapped y_true:", y_true[:5])
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, BatchNormalization, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout, Activation, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D

def create_cnn_eurosat_3_category_classify(input_shape, num_classes):
    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.1),
        RandomZoom(0.2),
        RandomContrast(0.2),
        RandomBrightness(0.2),
        RandomTranslation(0.1, 0.1)
    ])
    
    model = Sequential([
        Input(shape=input_shape),
        data_augmentation,

        # Block 1
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(5e-4)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=custom_sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    logging.info(f"learning_rate=1e-3")
    return model

   

"""
OOM out of memory issues , hence switched to simpke model above

def create_cnn_eurosat_3_category_classify(input_shape, num_classes):
    logging.info("Simplified data augmentation and reduced to 4 convolutional layers with global pooling to prevent overfitting.")

    # Data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.regularizers import l2

    # CNN model
    cnn_eurosat_3_category_classify = Sequential([
        Input(shape=input_shape),
        data_augmentation,  # Data augmentation
        
        # First block
        Conv2D(32, (3, 3), kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        
        # Second block
        Conv2D(64, (3, 3), kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),

        # Third block
        Conv2D(128, (3, 3), kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),

        # Fourth block
        Conv2D(256, (3, 3), kernel_regularizer=l2(5e-4)),
        BatchNormalization(),
        Activation('relu'),
        
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Prevent overfitting
        Dense(num_classes, activation='softmax')  # 3-category classification
    ])
    
    # Compile the model
    cnn_eurosat_3_category_classify.compile(
        optimizer='adam',
        loss=custom_sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return cnn_eurosat_3_category_classify

"""

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

def cyclical_lr(epoch, lr):
    base_lr = 1e-6
    max_lr = 1e-4
    step_size = 10
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))

def lr_warmup(epoch, lr):
    if epoch < 5:
        return lr * (epoch + 1) / 5
    return lr

"""
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)
"""



import logging
import traceback  # For detailed error tracebacks

def train_cnn_eurosat_3_category_classify(model_name, model, train_generator, steps_per_epoch, val_generator, validation_steps, batch_size, epochs=20):
    # Set up logfile to track training
    tr_log_file = f'../logs/07_b_execute_{model_name}_training_epochs_log_{timestamp}.log'
    sys.stdout = open(tr_log_file, 'w')  # Redirect stdout to the log file
    
    history = None
    
    try:
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'../models/{model}_{timestamp}.keras', save_best_only=True, monitor='val_loss')
        #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        
        from tensorflow.keras.callbacks import LearningRateScheduler
        
        # Add the cyclical learning rate scheduler
        lr_scheduler = LearningRateScheduler(cyclical_lr)

        # Log the start of training
        logging.info(f"\n{model_name} Training Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        """
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=[early_stopping, lr_scheduler]  # Updated scheduler
        )
        """

        history = model.fit(train_generator, epochs=5, validation_data=val_generator)

        # Check predictions after training
        # predictions = model.predict(train_images[:10])
        # predicted_classes = np.argmax(predictions, axis=1)
        # print("Predicted classes:", predicted_classes)
        # print("True labels:", train_labels[:10])

        logging.info(f"\n{model_name} Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error(traceback.format_exc())  # Log the full stack trace

    finally:
        # Ensure stdout is reset even if an error occurs
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        logging.info(f"{model_name} Training log saved to: {tr_log_file}.\n")
        
    return history




def main_visualizations(history, timestamp, model_name, model):
    """
    Orchestrates all visualization tasks: metrics, confusion matrix, and misclassified samples.
    """
    try:
        if history is None:  # Corrected syntax for checking history
            logging.info(f"No history available; skipping visualizations.")
            return
        
        # Visualize training metrics
        visualize_cnn_eurosat_3_category_classify_training_metrics(history, timestamp, model_name)
        logging.info(f"{model_name} Training metrics visualization completed.")

        """
        # Get predictions and predicted classes
        predictions = model.predict(val_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Compare predictions with remapped validation labels
        remapped_val_labels = np.where(val_labels == 1, 0, np.where(val_labels == 4, 1, 2))
        print("Predicted classes:", predicted_classes[:10])
        print("Remapped validation labels:", remapped_val_labels[:10])

        
        # Visualize confusion matrix
        visualize_cnn_eurosat_3_category_classify_training_confusionmatrix(predicted_classes, val_images, val_labels, timestamp, model_name)
        logging.info(f"{model_name} Confusion matrix visualization completed.")
        
        # Save misclassified samples
        cnn_eurosat_3_category_classify_chk_misclassified_indices(predicted_classes, val_images, val_labels, timestamp, model_name)
        logging.info(f"{model_name} Misclassified samples visualization completed.")
        
        # Save true classified samples
        cnn_eurosat_3_category_classify_chk_trueclassified_indices(predicted_classes, val_images, val_labels, timestamp, model_name)
        logging.info(f"{model_name} Trueclassified samples visualization completed.")
        """
    except Exception as e:
        logging.error(f"Error during {model_name} visualizations: {e}")
        raise
        
def cnn_eurosat_3_category_classify_chk_misclassified_indices(predicted_classes, val_images, val_labels, timestamp, model_name):
    """
    Saves misclassified images to the output directory.
    """
    try:
        output_dir = "../outputs/misclassified_samples"
        os.makedirs(output_dir, exist_ok=True)
    
        # Get misclassified indices
        misclassified_indices = np.where(predicted_classes != val_labels)[0]
        
        if len(misclassified_indices) == 0:
            logging.warning("No misclassified samples found.")
            return
        
        for i, idx in enumerate(misclassified_indices[:5]):
            plt.clf()
            plt.cla()
            plt.close('all')
            
            false_fig = plt.figure(figsize=(10, 8)) 
            false_image = (val_images[idx] * 255).astype('uint8')
            plt.imshow(false_image)
            plt.title(f"Misclassified: True: {val_labels[idx]}, Pred: {predicted_classes[idx]}")
            plt.axis('off')
            false_fig.savefig(f"{output_dir}/misclassified_3category_classify_{i}.png")
            plt.close(false_fig)
        
        logging.info(f"Misclassified images saved to: {output_dir}")
    
    except Exception as e:
        logging.error(f"Error while saving Misclassified classified images: {e}")


def cnn_eurosat_3_category_classify_chk_trueclassified_indices(predicted_classes, val_images, val_labels, timestamp, model_name):
    """
    Saves correctly classified images to the output directory.
    """

    try:
        output_dir = "../outputs/trueclassified_samples"
        os.makedirs(output_dir, exist_ok=True)

        # Get true classified indices
        trueclassified_indices = np.where(predicted_classes == val_labels)[0]
        
        if len(trueclassified_indices) == 0:
            logging.warning("No correctly classified samples found.")
            return
    
        for i, idx in enumerate(trueclassified_indices[:5]):
            plt.clf()
            plt.cla()
            plt.close('all')
            
            true_fig = plt.figure(figsize=(10, 8)) 
            true_image = (val_images[idx] * 255).astype('uint8')
            plt.imshow(true_image)
            plt.title(f"Correctly Classified: True: {val_labels[idx]}, Pred: {predicted_classes[idx]}")
            plt.axis('off')
            true_fig.savefig(f"{output_dir}/trueclassified_3category_classify_{i}.png")
            plt.close(true_fig)
        
        logging.info(f"Correctly classified images saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error while saving Correctly classified images: {e}")


def visualize_cnn_eurosat_3_category_classify_training_metrics(history, timestamp, model_name):
    
    try: 
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
        acc_loss_fig.savefig(f'../logs/07_b_execute_{model_name}_training_metrics_{timestamp}.png')
        plt.close(acc_loss_fig)  # Close the specific figure
        logging.info(f"Training metrics visualized and saved.")
    
    except Exception as e:
        logging.error(f"Error while saving Training metrics: {e}")
        

def visualize_cnn_eurosat_3_category_classify_training_confusionmatrix(predicted_classes, val_images, val_labels, timestamp, model_name):
    try:
        # Remap validation labels
        remapped_labels = np.where(val_labels == 1, 0, np.where(val_labels == 4, 1, 2))
        
        # Generate and log classification report
        clf_report = classification_report(remapped_labels, predicted_classes, target_names=['Residential', 'Industrial', 'HerbaceousVegetation'])
        logging.info("\n" + clf_report)

        # Generate confusion matrix with remapped labels
        cm = confusion_matrix(remapped_labels, predicted_classes)
        class_names = ['Residential', 'Industrial', 'HerbaceousVegetation']
        
        # Clear previous figures
        plt.clf()
        plt.cla()
        plt.close('all')
        
        cfmatrix_fig = plt.figure(figsize=(10, 8))  # Adjust size if needed
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cfmatrix_fig.savefig(f'../logs/07_b_execute_{model_name}_training_confusion_matrix_{timestamp}.png')
        plt.close(cfmatrix_fig)
        logging.info("Confusion matrix saved.")

    except Exception as e:
        logging.error(f"Error while saving confusion matrix: {e}")

