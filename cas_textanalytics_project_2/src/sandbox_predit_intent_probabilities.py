import numpy as np
import torch
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load GloVe embeddings
def load_glove_embeddings(glove_path, embedding_dim=300):
    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                if len(coefs) == embedding_dim:
                    embeddings_index[word] = coefs
            except ValueError:
                pass
    return embeddings_index

# Convert text to GloVe vector
def text_to_glove_vector(text, embeddings_index, embedding_dim=300):
    tokens = text.lower().split()
    vectors = [embeddings_index.get(token, np.zeros(embedding_dim)) for token in tokens]
    return np.mean(vectors, axis=0)

# Load your trained model
model = load_model(f"../models/tier2_01_cnn_model_1736284036.keras") 

# Predict intent and plot probabilities
def predict_and_plot(user_input):
    glove_path = "../data/glove.840B.300d.txt"  # Replace with actual GloVe path
    embeddings_index = load_glove_embeddings(glove_path)
    
    glove_vector = text_to_glove_vector(user_input, embeddings_index)
    input_tensor = np.expand_dims(glove_vector, axis=0)
    
    prediction = model.predict(input_tensor)
    intent = np.argmax(prediction)
    
    # Print the probabilities
    print(f"Input: {user_input}")
    print(f"Predicted probabilities: {prediction}")
    print(f"Predicted intent: {intent}")
    
    # Plot the probabilities
    intents = ["Intent 0", "Intent 1", "Intent 2"]
    plt.bar(intents, prediction[0])
    plt.title(f"Intent Probabilities for Input: '{user_input}'")
    plt.ylabel("Probability")
    plt.xlabel("Intents")
    plt.ylim(0, 1)
    plt.show()

# Example usage
if __name__ == "__main__":
    user_input = "Can you recommend a restaurant?"  # Replace with any input
    predict_and_plot(user_input)
