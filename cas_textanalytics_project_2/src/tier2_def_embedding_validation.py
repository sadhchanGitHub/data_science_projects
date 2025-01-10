import numpy as np
from itertools import combinations
from io import StringIO
import random
import re
import logging


def get_domain_specific_words():
    # List of domain-specific words
    return ['taste', 'ambiance', 'flavor', 'portion', 'price', 'quality', 'service', 'atmosphere']

def generate_word_pairs(words, max_pairs=20):
    """Generate random word pairs from a given word list."""
    return random.sample(list(combinations(words, 2)), min(len(words) * (len(words) - 1) // 2, max_pairs))

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def check_missing_words(tokenizer_word_index, embedding_matrix):
    # Check for missing words
    vocab_size = len(tokenizer_word_index)
    missing_words = []
    for word, idx in tokenizer_word_index.items():
        if idx >= len(embedding_matrix) or not np.any(embedding_matrix[idx]):
            missing_words.append(word)
    return missing_words, vocab_size

import re

def filter_significant_missing_words(missing_words, ignore_words):
    """Filter out restaurant-related words using precompiled regex patterns."""
    # Precompile patterns for all ignore words
    patterns = [re.compile(rf'\b{re.escape(ignore_word)}\b', re.IGNORECASE) for ignore_word in ignore_words]
    
    # Filter missing words
    significant_missing = []
    for word in missing_words:
        if not any(pattern.search(word) for pattern in patterns):
            significant_missing.append(word)
    
    return significant_missing


def validate_embeddings(tokenizer, embedding_matrix, restaurant_names, top_n=50, max_pairs=20):
    output = StringIO()
    
    # Get top N frequent words from the tokenizer
    top_words = [word for word, _ in sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    
    # Add domain-specific words
    domain_specific_words = get_domain_specific_words()
    combined_words = list(set(top_words + domain_specific_words))
    
    # Exclude restaurant names
    combined_words = [word for word in combined_words if word not in restaurant_names]

    # Validate individual word embeddings
    output.write("\n--- Embedding Validation for Combined Words ---\n")
    for word in combined_words:
        if word in tokenizer.word_index:
            word_index = tokenizer.word_index[word]
            embedding_vector = embedding_matrix[word_index]
            output.write(f"Embedding for '{word}': {embedding_vector[:5]}...\n")
        else:
            output.write(f"'{word}' not found in tokenizer vocabulary.\n")

    # Generate random word pairs for cosine similarity
    word_pairs = generate_word_pairs(combined_words, max_pairs=max_pairs)
    
    # Validate cosine similarity for word pairs
    output.write("\n--- Cosine Similarity for Word Pairs ---\n")
    for word1, word2 in word_pairs:
        if word1 in tokenizer.word_index and word2 in tokenizer.word_index:
            vec1 = embedding_matrix[tokenizer.word_index[word1]]
            vec2 = embedding_matrix[tokenizer.word_index[word2]]
            similarity = cosine_similarity(vec1, vec2)
            output.write(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}\n")
        else:
            output.write(f"One or both words not found: '{word1}', '{word2}'\n")

    # Check missing words
    missing_words, vocab_size = check_missing_words(tokenizer.word_index, embedding_matrix)
    significant_missing = filter_significant_missing_words(missing_words, restaurant_names)
    
    significant_num_missing = len(significant_missing)
    print(f"Significant Missing words in embedding matrix (excluding restaurant names): {significant_num_missing}/{vocab_size} ({(significant_num_missing/vocab_size)*100:.2f}%)")
    print(f"Sample missing words (non-restaurant names): {significant_missing[:10]} \n")  # Print first 100 missing words
    
    output.write("\n--- Significant Missing Words ---\n")
    output.write(f"Total significant missing words: {len(significant_missing)}\n")
    output.write(f"Total significant missing words % to total: ({(significant_num_missing/vocab_size)*100:.2f}%)\n")
    output.write(f"Sample: {significant_missing[:100]}\n")
    
    return output.getvalue()
