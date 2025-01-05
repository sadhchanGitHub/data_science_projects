import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import time

# visualize embeddings
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt


# Import your functions
from tokenizer_and_embedding_setup import load_glove_embeddings, create_embedding_matrix  
from tokenizer_and_embedding_setup import prepare_tokenizer, create_embedding_matrix
from embedding_validation import validate_embeddings
from embedding_visualization import visualize_embeddings
from build_CNN_model import preprocess_sequences_and_labels, define_model, train_model, visualize_model_training

# Configure GPU settings (restrict TensorFlow to use only the first GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
        print("\n")
        print("GPU configured successfully.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
        
# Load the dataset
data_path = "../data/New_York_reviews_with_no_restaurantname_in_Review.csv"
df = pd.read_csv(data_path)

timestamp = int(time.time())

# Verify dataset
print("Dataset loaded successfully.")
print("Dataset columns names as follows: ")
print(df.columns)
print("\n")

# Prepare tokenizer
tokenizer = prepare_tokenizer(df)

# Load GloVe embeddings
# glove_path = "../data/glove.6B.100d.txt"
# embedding_dim = 100

glove_path = "../data/glove.840B.300d.txt"
embedding_dim = 300

embeddings_index = load_glove_embeddings(glove_path, embedding_dim)

# Prepare embedding matrix
embedding_matrix = create_embedding_matrix(tokenizer.word_index, embeddings_index, embedding_dim)
# Validation
print(f"Embedding matrix shape: {embedding_matrix.shape}")
#print(f"Sample embedding for 'delicious': {embeddings_index.get('delicious', 'Not found')}")
logging.info("Embedding matrix created.")

# Validate embeddings
restaurant_names = df['restaurant_name'].str.lower().unique().tolist()
validation_output = validate_embeddings(tokenizer, embedding_matrix, restaurant_names)
with open(f'../logs/validation_results_{timestamp}.log', 'w') as log_file:
    log_file.write(validation_output)

# List of words to visualize
# words_to_plot = ['taste', 'ambiance', 'flavor', 'portion', 'price', 'quality', 'service', 'atmosphere']
words_to_plot = [
    'food', 'good', 'great', 'service', 'place', 'restaurant', 'delicious',
    'nice', 'menu', 'excellent', 'friendly', 'atmosphere', 'experience',
    'wine', 'lunch', 'dinner', 'breakfast', 'recommend', 'amazing',
    'pizza', 'chicken', 'salad', 'cozy', 'noisy', 'overpriced', 'cheap'
]

# Visualize embeddings
visualize_embeddings(tokenizer, embedding_matrix, words_to_plot, output_path="pca_visualization_data.npz")
logging.info("Word embeddings visualization completed.")


# Preprocess sequences and labels
X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_sequences_and_labels(
    df, tokenizer, sentiment_column='sentiment', max_length=100, test_size=0.2, val_size=0.1
)
logging.info("Sequences and labels preprocessed.")

# Define the model
model = define_model(tokenizer, embedding_matrix, embedding_dim)
logging.info("Model defined and compiled.")

# Train the model
history = train_model(model, X_train, y_train, X_val, y_val)
logging.info("Model training completed.")

# Visualize training progress
visualize_model_training(history, timestamp)
logging.info("Training progress visualization completed.")
    