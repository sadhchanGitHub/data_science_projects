import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from joblib import dump, load

# import functions
from tier2_def_tokenizer_and_embedding_setup import load_glove_embeddings

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime

# Function to print current timestamp
def print_timestamp(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")
    
# Load tokenizer
def load_tokenizer(tokenizer_path):
    #print_timestamp("Load tokenizer...")
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer


# Tokenize and pad input text
#print_timestamp("pre process text...")
def preprocess_text(text, tokenizer, max_length=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return padded_sequence


def predict_sentiment(text, model, tokenizer, max_length=100):
    preprocessed_text = preprocess_text(text, tokenizer, max_length)
    prediction = model.predict(preprocessed_text, verbose=0)  # Suppress output
    return 'positive' if prediction >= 0.5 else 'negative'

def find_restaurants(user_input, data, glove_path, tokenizer, sentiment_model, cuisines, embedding_dim=300, max_length=100):
    """
    Find restaurants by matching keywords and detecting cuisines, prioritizing cuisine-based filtering.
    """
    # Detect cuisine from user input
    detected_cuisine = next((cuisine for cuisine in cuisines if cuisine.lower() in user_input.lower()), None)
    
    if detected_cuisine:
        print(f"Detected Cuisine: {detected_cuisine}")
        filtered_data = data[data['review_no_restaurant'].str.contains(detected_cuisine, case=False, na=False)]
        if not filtered_data.empty:
            # Return top 10 unique restaurant names for the cuisine
            return filtered_data['restaurant_name'].drop_duplicates().head(10).tolist()
    
    # Fallback to keyword matching
    print("No cuisine detected, falling back to keyword search...")
    keywords = user_input.lower().split()
    keyword_filter = data['review_no_restaurant'].str.contains('|'.join(keywords), case=False, na=False)
    filtered_data = data[keyword_filter]
    
    if not filtered_data.empty:
        # Return top 10 unique restaurant names for keywords
        return filtered_data['restaurant_name'].drop_duplicates().head(10).tolist()
    
    # Final fallback: Provide top 10 generic restaurants
    print("No keywords or cuisines matched. Returning generic top-rated restaurants...")
    top_restaurants = data['restaurant_name'].value_counts().head(10).index.tolist()
    return top_restaurants


def chatbot_response(user_input, data, glove_path, tokenizer_path, sentiment_model_path, cuisines):
    tokenizer = load_tokenizer(tokenizer_path)
    sentiment_model = load_model(sentiment_model_path)
    
    recommendations = find_restaurants(user_input, data, glove_path, tokenizer, sentiment_model, cuisines)
    
    # Format the response
    return f"Here are some recommended restaurants: {', '.join(recommendations)}"


def initialize_params():
    data = pd.read_csv("../data/New_York_reviews_with_no_restaurantname_in_Review.csv") 
    glove_path = "../data/glove.840B.300d.txt"  
    tokenizer_path = "../models/tier2_01_execute_CNN_tokenizer.pkl"  
    sentiment_model_path = "../models/tier2_01_cnn_model_1736284036.keras"  
    
    # List of popular cuisines
    cuisines = ["Italian", "Indian", "Chinese", "Mexican", "Japanese", "Thai", 
                "Mediterranean", "French", "American", "Korean", "Vietnamese", 
                "Spanish", "Lebanese", "Greek", "Turkish"]
    
    return data, glove_path, tokenizer_path, sentiment_model_path, cuisines

if __name__ == "__main__":
  
    # Initialize parameters
    data, glove_path, tokenizer_path, sentiment_model_path, cuisines = initialize_params()
    
    # Get user input from the command-line argument
    if len(sys.argv) != 2:
        print("Usage: python tier3_leightweight_chatbot.py <user_input> \n ")
        sys.exit(1)

    user_input = sys.argv[1]
    print(f"\n user_input is {user_input} \n ")
    
    # Call the chatbot response
    print(chatbot_response(user_input, data, glove_path, tokenizer_path, sentiment_model_path, cuisines))

# Gradio interface
interface = gr.Interface(
    fn=chatbot_response, 
    inputs="text", 
    outputs="text", 
    title="Simple GloVe-Based Chatbot",
    description="Ask me about restaurants or popular dishes!"
)

if __name__ == "__main__":
    interface.launch()
