import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import logging
import time
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

timestamp = int(time.time())

# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
#log_file = os.path.join(log_dir, f"eurosat_cnn_evaluation_{int(datetime.now().timestamp())}.log")
log_file = os.path.join(log_dir, f"evaluate_cnn_eurosat_model_with_augmentation_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)


def load_data(data_dir, categories, split):
    """
    Load test data from .npy files for all categories.
    """
    test_images = []
    test_labels = []
    for i, category in enumerate(categories):
        test_images.append(np.load(os.path.join(data_dir, f"{category}_{split}.npy")))
        test_labels.append(np.load(os.path.join(data_dir, f"{category}_{split}_labels_one_hot.npy")))
    
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    logging.info(f"Test images shape: {test_images.shape}")
    logging.info(f"Test labels shape: {test_labels.shape} \n")

    return test_images, test_labels

def evaluate_model(model_path, data_dir, categories, input_shape):
    """
    Evaluate a trained model on the test set.
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}\n")
    logging.info(f"Model loaded from {model_path}\n")
    
    # Load test data
    test_images, test_labels = load_data(data_dir, categories, split="test")
    print(f"Test data shape: {test_images.shape}, Test labels shape: {test_labels.shape}")
    logging.info(f"Test data shape: {test_images.shape}, Test labels shape: {test_labels.shape}")
    
    # Normalize the images (if not already normalized)
    # test_images = test_images / 255.0  # Ensure pixel values are in the range [0, 1]
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}\n")
    logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}\n")
    
    # Predict on test data
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # Generate classification report
    print("Classification Report:\n")
    clf_report = classification_report(true_classes, predicted_classes, target_names=categories)
    print(clf_report)
    logging.info(f"Classification Report'")
    logging.info(clf_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"../logs/eurosat_cnn_with_augmentation_confusion_matrix_{timestamp}.png")
    print("Confusion matrix saved as PNG.")
    #plt.show()
    logging.info(f"Confusion matrix saved as 'eurosat_cnn_with_augmentation_confusion_matrix_{timestamp}.png'. \n")



# Configuration
data_dir = "../data/training_data"
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
#model_path = "../models/eurosat_cnn_model_1736155383.keras" 
model_path = "../models/create_cnn_eurosat_model_with_augmentation_1736189259.keras" 

input_shape = (256, 256, 3)

# Run evaluation
evaluate_model(model_path, data_dir, categories, input_shape)
