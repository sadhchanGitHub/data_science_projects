import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import logging
import time
import sys

timestamp = int(time.time())

from sklearn.model_selection import train_test_split

# Preprocess sequences and labels
def preprocess_sequences_and_labels(data, tokenizer, sentiment_column='sentiment', max_length=100, test_size=0.2, val_size=0.1, random_state=42):
    # Convert text to sequences
    X = pad_sequences(tokenizer.texts_to_sequences(data['review_no_restaurant']), maxlen=max_length)
    le = LabelEncoder()
    y = le.fit_transform(data[sentiment_column])
    
    # Split into train and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Further split training set into train and validation sets
    val_split = int(len(X_train_full) * val_size / (1 - test_size))
    X_train, X_val = X_train_full[val_split:], X_train_full[:val_split]
    y_train, y_val = y_train_full[val_split:], y_train_full[:val_split]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le



def define_model(tokenizer, embedding_matrix, embedding_dim):
    model = Sequential([
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False
        ),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.5),  # Add Dropout layer to reduce overfitting
        Dense(10, activation='relu'),
        Dropout(0.5),  # Another Dropout layer after Dense layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model



def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=20):
    # Set up logging to a file
    log_filename = f'../logs/training_log_{timestamp}.log'
    sys.stdout = open(log_filename, 'w')  # Redirect stdout to the log file

    epochs = 20
    batch_size = 32

    try:
        # Add logging for the start of training
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'../models/best_cnn_model_{int(time.time())}.keras', save_best_only=True, monitor='val_loss')

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint]
        )

        # End logging
        print(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    finally:
        # Ensure stdout is reset even if an error occurs
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print(f"Training log saved to: {log_filename}")
    
    return history

# Visualize training progress
def visualize_model_training(history, timestamp):
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
    plt.savefig(f'../logs/training_progress_{timestamp}.png')
    logging.info(f"Training progress visualized and saved as 'training_progress_{timestamp}.png'.")
