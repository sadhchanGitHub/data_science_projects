import pandas as pd
import numpy as np
import tensorflow as tf
from io import StringIO

from importlib import reload
import logging
reload(logging)

import time
import os

# visualize embeddings
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt


# Import your functions
from tier2_def_tokenizer_and_embedding_setup import load_glove_embeddings, create_embedding_matrix, prepare_tokenizer
from tier2_def_embedding_validation import validate_embeddings
from tier2_def_embedding_visualization import visualize_embeddings
from tier2_def_build_CNN_model import preprocess_sequences_and_labels, define_model, train_model, visualize_model_training


timestamp = int(time.time())

# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tier2_01_execute_CNN_model_training_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logging.info(" \n ")
logging.info("called via tier2_01_execute_CNN_model_training.py...\n")
logging.info(" Script Started ...\n")


# Configure GPU settings (restrict TensorFlow to use only the first GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Optional: Enable memory growth
        logging.info("GPU configured successfully. \n")
    except RuntimeError as e:
        logging.info(f"Error configuring GPU: {e} \n")
        
# Load the dataset
data_path = "../data/New_York_reviews_with_no_restaurantname_in_Review.csv"
df = pd.read_csv(data_path)

# Verify dataset
logging.info("Dataset loaded successfully.")
logging.info("Dataset columns names as follows: ")
logging.info(df.columns)
logging.info("\n")

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
logging.info(f"Embedding matrix shape: {embedding_matrix.shape}")
#print(f"Sample embedding for 'delicious': {embeddings_index.get('delicious', 'Not found')}")
logging.info("Embedding matrix created.")

# Validate embeddings
restaurant_names = df['restaurant_name'].str.lower().unique().tolist()
validation_output = validate_embeddings(tokenizer, embedding_matrix, restaurant_names)
with open(f'../logs/tier2_01_execute_CNN_model_training_2_validation_results_{timestamp}.log', 'w') as log_file:
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
visualize_embeddings(tokenizer, embedding_matrix, words_to_plot, output_path="../outputs/pca_visualization_data.npz")
logging.info("\n Word embeddings visualization completed.\n ")


# Preprocess sequences and labels
X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_sequences_and_labels(
    df, tokenizer, sentiment_column='sentiment', max_length=100, test_size=0.2, val_size=0.1
)
logging.info("\n Sequences and labels preprocessed.\n")

# Define the model
model = define_model(tokenizer, embedding_matrix, embedding_dim)
logging.info("\n Model defined and compiled. \n")

# Capture the model summary in a StringIO buffer
summary_buffer = StringIO()
model.summary(print_fn=lambda x: summary_buffer.write(x + "\n"))
# Log the model summary
logging.info("\n Model Summary:\n" + summary_buffer.getvalue())

# Train the model
history = train_model(model, X_train, y_train, X_val, y_val)
logging.info("\n Model training completed. \n")

# Visualize training progress
visualize_model_training(history, timestamp)
logging.info("Training progress visualization completed.")
    