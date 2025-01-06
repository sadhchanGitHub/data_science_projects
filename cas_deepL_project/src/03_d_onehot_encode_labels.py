import os
import numpy as np
from tensorflow.keras.utils import to_categorical

training_dir = "../data/training_data"

categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]
num_classes = len(categories)

for category in categories:
    train_labels = np.load(os.path.join(training_dir, f"{category}_train_labels.npy"))
    val_labels = np.load(os.path.join(training_dir, f"{category}_val_labels.npy"))
    test_labels = np.load(os.path.join(training_dir, f"{category}_test_labels.npy"))
    
    train_labels_one_hot = to_categorical(train_labels, num_classes)
    val_labels_one_hot = to_categorical(val_labels, num_classes)
    test_labels_one_hot = to_categorical(test_labels, num_classes)
    
    np.save(os.path.join(training_dir, f"{category}_train_labels_one_hot.npy"), train_labels_one_hot)
    np.save(os.path.join(training_dir, f"{category}_val_labels_one_hot.npy"), val_labels_one_hot)
    np.save(os.path.join(training_dir, f"{category}_test_labels_one_hot.npy"), test_labels_one_hot)
    
    print(f"One-hot encoding completed and saved for category: {category}")
