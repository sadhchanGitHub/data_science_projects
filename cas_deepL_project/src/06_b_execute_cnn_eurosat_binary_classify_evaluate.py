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



def load_data(data_dir, categories, split):
    """
    Load test data from .npy files for all categories.
    """
    test_images = []
    test_labels = []

    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    test_images = np.load(os.path.join(training_dir_remapped, f"binary_{split}_images.npy"))
    test_labels = np.load(os.path.join(training_dir_remapped, f"binary_{split}_labels.npy"))
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
    test_images, test_labels = load_data(data_dir, split="test")
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
    plt.savefig(f"../outputs/06_b_{model_name}_evaluation_confusion_matrix_{timestamp}.png")
    logging.info(f"Confusion matrix saved as '06_b_{model_name}_evaluation_confusion_matrix_{timestamp}.png'. \n")

def sanity_check_testdata(training_dir_remapped):
    test_images = np.load(os.path.join(training_dir_remapped, "binary_test_images.npy"))
    test_labels = np.load(os.path.join(training_dir_remapped, "binary_test_labels.npy"))
    logging.info(f"train_images shape: {test_images.shape}")
    logging.info(f"train_labels shape: {test_labels.shape}")
    logging.info(f"Sample labels: {test_labels[:5]}")



def main():
    try:
        
        # Logging configuration
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"06_b_execute_cnn_eurosat_binary_classify_evaluate_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        
        logging.info("\n")
        logging.info("called via 06_b_execute_cnn_eurosat_binary_classify_evaluate.py...\n")
        logging.info(" Script Started ...\n")
        logging.info("This script will evaluate the cnn model passed via args along with image_size \n")

        # Check if image size is passed as an argument
        if len(sys.argv) != 3:
            print("Usage: python 06_b_execute_cnn_eurosat_binary_classify_evaluate.py <image_size>")
            sys.exit(1)
    
        # Get image size from the command-line argument
        image_size = int(sys.argv[1])
        logging.info(f"image_size is {image_size} \n ")


        # Get the model path from the command-line argument
        model_name = sys.argv[2]
        # Load the model using the provided path
        model = load_model(f"../models/{model_name}")
        logging.info(f"model_name is {model_name} \n ")

        # call gpu usage
        chk_gpu_config()

        training_dir_remapped = f"../data/training_data_remapped_binary_{image_size}"
        logging.info(f"training_dir_remapped is {training_dir_remapped} \n ")
        batch_size = 32
        logging.info(f"batch_size is {batch_size} \n ")

        # Sanity check
        sanity_check_testdata(training_dir_remapped)

        #derive input_size from image_size
        input_shape = (image_size, image_size, 3)  # Shape of input images
        num_classes = 2  # Number of categories
        logging.info(f"batch_size is {input_shape} and num_classes is {num_classes} \n ")
        
        # Run evaluation
        evaluate_model(model, training_dir_remapped, input_shape)

    except Exception as e:
        logging.error(f"Error during script : {e}")
        raise


    logging.info(f"Script Ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Script failed: {e}")
    
