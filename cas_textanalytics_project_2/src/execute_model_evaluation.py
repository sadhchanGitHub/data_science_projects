import numpy as np
import pandas as pd
import time
import sys


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tokenizer_and_embedding_setup import prepare_tokenizer



timestamp = int(time.time())

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

# Load the trained model
model = load_model('../models/best_cnn_model_1736031362.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Predict on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f'../logs/confusion_matrix_{timestamp}.png')
print(f"Confusion matrix saved to ../logs/confusion_matrix_{timestamp}.png")

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Check false negatives
print("\nFalse Negatives:")
for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
    if true_label == 1 and pred_label == 0:  # False Negative
        print(f"Review {i}: True Label = {true_label}, Predicted = {pred_label}")
        print(f"Text: {df['review_no_restaurant'].iloc[i]}")
        print("---")
        
"""
for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
    if true_label != pred_label:
        print(f"Review {i}: True Label = {true_label}, Predicted = {pred_label}")
        print(f"Text: {df['review_no_restaurant'].iloc[i]}")
        print("---")
"""






