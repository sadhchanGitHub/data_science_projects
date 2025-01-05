import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
from tokenizer_and_embedding_setup import prepare_tokenizer, create_embedding_matrix
from embedding_validation import validate_embeddings
import tensorflow as tf


# Load dataset
df = pd.read_csv("../data/New_York_reviews_cleaned_with_spacy.csv")

# Rename the column
df.rename(columns={'sample': 'sentiment'}, inplace=True)

# Verify the change
print(df.columns)


# Preprocess the text
max_vocab = 5000
max_length = 100

# Tokenizer
print("Fitting tokenizer...")
tokenizer = Tokenizer(num_words=max_vocab, oov_token="<UNK>")
tokenizer.fit_on_texts(df['cleaned_review'])
X = tokenizer.texts_to_sequences(df['cleaned_review'])
X = pad_sequences(X, maxlen=max_length, padding='post')

# Save the tokenizer to a file
with open('../models/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

print("Tokenizer saved successfully.")

# Label encoding for sentiment
y = df['sentiment']  # Ensure 'sentiment' is the correct column for labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Configure logging
logging.basicConfig(
    filename='../logs/malformed_lines.log',  # Log file name
    level=logging.INFO,              # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def load_glove_embeddings(glove_path, embedding_dim):
    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                if len(coefs) == embedding_dim:
                    embeddings_index[word] = coefs
                else:
                    logging.info(f"Skipping malformed line {line_num}: {line.strip()}")
            except ValueError:
                logging.info(f"Skipping line {line_num}: {line.strip()}")
    return embeddings_index


# glove_path = "../data/glove.6B.100d.txt"  # Update this to the correct path

glove_path = "../data/glove.840B.300d.txt"  # Update this to the correct path

embedding_dim = 300
embeddings_index = load_glove_embeddings(glove_path, embedding_dim)

# Prepare embedding matrix
def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_matrix = create_embedding_matrix(tokenizer.word_index, embeddings_index, embedding_dim)

# Validation
print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"Sample embedding for 'delicious': {embeddings_index.get('delicious', 'Not found')}")
