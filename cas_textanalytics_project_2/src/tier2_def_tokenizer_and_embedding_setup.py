import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import logging
import time
import os
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer



timestamp = int(time.time())



def load_glove_embeddings(glove_path, embedding_dim):
    
    logging.info(f"executing load_glove_embeddings... ")

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
                logging.info(f"Skipping line {line_num} due to ValueError: {line.strip()}")
    return embeddings_index

# Prepare embedding matrix
def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    logging.info(f"starting create_embedding_matrix... ")
    print("starting create_embedding_matrix ...")
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("end create_embedding_matrix ...\n")
    logging.info(f"end create_embedding_matrix ... ")
    return embedding_matrix


def prepare_tokenizer(data, max_vocab=5000, max_length=100, save_path='../models/tier2_01_execute_CNN_tokenizer.pkl'):
    logging.info("starting prepare_tokenizer ...")

    # Get all unique restaurant names in lowercase
    ignore_words = set(data['restaurant_name'].str.lower().unique())
    
    # Initialize and fit tokenizer on preprocessed text
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<UNK>")
    tokenizer.fit_on_texts(data['review_no_restaurant'])
    
    # Save tokenizer
    with open(save_path, 'wb') as file:
        pickle.dump(tokenizer, file)
        
    logging.info(f"Tokenizer saved successfully {save_path} \n")

    
    return tokenizer







