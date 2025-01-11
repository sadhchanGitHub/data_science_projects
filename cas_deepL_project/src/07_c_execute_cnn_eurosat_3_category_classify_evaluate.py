import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time
from datetime import datetime
import sys
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

timestamp = int(time.time())

def chk_gpu_config():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU configured successfully.")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")

def load_data(data_dir, split):
    test_images = np.load(os.path.join(data_dir, f"binary_{split}_images.npy"))
    test_labels = np.load(os.path.join(data_dir, f"binary_{split}_labels.npy"))
    logging.info(f"{split.capitalize()} images shape: {test_images.shape}")
    logging.info(f"{split.capitalize()} labels shape: {test_labels.shape}")
    return test_images, test_labels
def evaluate_model(model_path, model_name_trim, data_dir, input_shape):
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
    
    # Evaluate the model
    results = model.evaluate(test_images, test_labels, verbose=1)
    
    # Print all metrics
    metrics = model.metrics_names
    for metric_name, result in zip(metrics, results):
        print(f"{metric_name}: {result:.4f}")
        logging.info(f"{metric_name}: {result:.4f}")
    

        # Generate predictions
        from sklearn.metrics import classification_report

        predictions = model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        print(classification_report(test_labels, predicted_classes, target_names=['HerbaceousVegetation', 'Industrial', 'Residential']))


    # we now have only two categories
    class_names = ['Residential', 'HerbaceousVegetation']
    
    # Generate classification report
    print("Classification Report:\n")
    clf_report = classification_report(true_classes, predicted_classes, target_names=class_names)
    logging.info("Classification Report")
    logging.info(clf_report)
    

    
    # Clear previous figures
    plt.clf()
    plt.cla()
    plt.close('all')
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    cfmatrix_fig = plt.figure(figsize=(10, 8))  # Adjust size if needed
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cfmatrix_fig.savefig(f'../outputs/06_b_{model_name_trim}_evaluation_confusion_matrix_{timestamp}.png')
    plt.close(cfmatrix_fig)
    logging.info(f"Confusion matrix saved as '06_b_{model_name_trim}_evaluation_confusion_matrix_{timestamp}.png'. \n")


def main():
    try:
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"06_b_execute_cnn_eurosat_binary_classify_evaluate_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        logging.info("Script Started...")

        if len(sys.argv) != 3:
            print("Usage: python 06_b_execute_cnn_eurosat_binary_classify_evaluate.py <image_size> <model_name>")
            sys.exit(1)

        image_size = int(sys.argv[1])
        model_name = sys.argv[2]
        model_path = f"../models/{model_name}"

        model_name_trim = model_name.replace(".keras", "")

        chk_gpu_config()

        data_dir = f"../data/training_data_remapped_binary_{image_size}"
        evaluate_model(model_path, model_name_trim, data_dir, (image_size, image_size, 3))

    except Exception as e:
        logging.error(f"Error during script execution: {e}")
        raise

if __name__ == "__main__":
    main()
