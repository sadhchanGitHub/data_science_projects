import numpy as np
import pandas as pd
import time
import sys
import os
import logging

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tier2_def_tokenizer_and_embedding_setup import prepare_tokenizer


# Check if model path is passed as an argument
if len(sys.argv) != 2:
    print("Usage: python tier2_02_execute_CNN_model_evaluation.py <model_path>")
    sys.exit(1)

# Get the model path from the command-line argument
model_name = sys.argv[1]

# Load the model using the provided path
model = load_model(f"../models/{model_name}")


timestamp = int(time.time())

# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tier2_02_execute_CNN_model_evaluation_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logging.info(" \n")
logging.info("called via tier2_02_execute_CNN_model_evaluation.py...\n")
logging.info(" Script Started ...\n")

# Load the dataset
data_path = "../data/New_York_reviews_with_no_restaurantname_in_Review.csv"
df = pd.read_csv(data_path)

# Prepare tokenizer (reuse the same logic from training)
tokenizer = prepare_tokenizer(df)

# Preprocess test data
max_length = 100  # Ensure this matches the length used during training
X = pad_sequences(tokenizer.texts_to_sequences(df['review_no_restaurant']), maxlen=max_length)
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])

# Use the same split logic (ensure random_state matches)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
logging.info(f"Test Loss: {loss}")
logging.info(f"Test Accuracy: {accuracy}")


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Predict on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
logging.info("confusion matrix is as follows:")
logging.info(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f'../outputs/tier2_02_execute_CNN_model_evaluation_confusion_matrix_{timestamp}.png')
logging.info(f"Confusion matrix saved to ../outputs/tier2_02_execute_CNN_model_evaluation_confusion_matrix_{timestamp}.png")

# Classification Report
clf_report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
logging.info("Classification Report for tier2_02_execute_CNN_model_evaluation is as follows: ")
logging.info(clf_report)

"""
# Check false negatives
print("\nFalse Negatives:")
for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
    if true_label == 1 and pred_label == 0:  # False Negative
        print(f"Review {i}: True Label = {true_label}, Predicted = {pred_label}")
        print(f"Text: {df['review_no_restaurant'].iloc[i]}")
        print("---")

"""

logging.info("script done \n")



