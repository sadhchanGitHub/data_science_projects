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

def find_restaurants(user_input, data, glove_path, tokenizer, sentiment_model, cuisines):
    detected_cuisine = next((cuisine for cuisine in cuisines if cuisine.lower() in user_input.lower()), None)
    
    if detected_cuisine:
        filtered_data = data[data['review_no_restaurant'].str.contains(detected_cuisine, case=False, na=False)]
        if not filtered_data.empty:
            return filtered_data['restaurant_name'].drop_duplicates().head(10).tolist(), False
    
    keywords = user_input.lower().split()
    keyword_filter = data['review_no_restaurant'].str.contains('|'.join(keywords), case=False, na=False)
    filtered_data = data[keyword_filter]
    
    if not filtered_data.empty:
        return filtered_data['restaurant_name'].drop_duplicates().head(10).tolist(), False
    
    # Fallback to top-rated options
    top_rated = data['restaurant_name'].value_counts().head(10).index.tolist()
    return top_rated, True


"""
def chatbot_response(user_input, data, glove_path, tokenizer_path, sentiment_model_path, cuisines):
    tokenizer = load_tokenizer(tokenizer_path)
    sentiment_model = load_model(sentiment_model_path)
    
    recommendations = find_restaurants(user_input, data, glove_path, tokenizer, sentiment_model, cuisines)
    
    # Format the response
    return f"Here are some recommended restaurants: {', '.join(recommendations)}"
"""

def chatbot_response(user_input, data, glove_path, tokenizer_path, sentiment_model_path, cuisines):
    tokenizer = load_tokenizer(tokenizer_path)
    sentiment_model = load_model(sentiment_model_path)
    
    # Get recommendations and fallback flag
    recommendations, fallback = find_restaurants(user_input, data, glove_path, tokenizer, sentiment_model, cuisines)
    
    # Customize response based on fallback
    if fallback:
        return (
            f"Sorry, I couldn't find restaurants specifically matching your input '{user_input}'. "
            f"Here are some top-rated options instead: "
            f"{', '.join(recommendations[:5])}"
        )
    else:
        return (
            f"Here are some restaurants matching your input '{user_input}': "
            f"{', '.join(recommendations[:5])}"
        )



def initialize_params():
    
    try:
        data = pd.read_csv("../data/New_York_reviews_with_no_restaurantname_in_Review.csv")
    except FileNotFoundError:
        print("Error: Data file not found.")
        sys.exit(1)

    try:
        glove_path = "../data/glove.840B.300d.txt"  
    except FileNotFoundError:
        print("Error: glove_path file not found.")
        sys.exit(1)
    
    try:
        tokenizer_path = "../models/tier2_01_execute_CNN_tokenizer.pkl"  
    except FileNotFoundError:
        print("Error: tokenizer_path file not found.")
        sys.exit(1)
    try:
        sentiment_model_path = "../models/tier2_01_cnn_model_1736284036.keras"  
    except FileNotFoundError:
        print("Error: sentiment_model_path file not found.")
        sys.exit(1)
    
    
    # List of popular cuisines
    cuisines = ["Italian", "Indian", "Chinese", "Mexican", "Japanese", "Thai", 
                "Mediterranean", "French", "American", "Korean", "Vietnamese", 
                "Spanish", "Lebanese", "Greek", "Turkish"]
    
    return data, glove_path, tokenizer_path, sentiment_model_path, cuisines

import gradio as gr

def main():
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the chatbot.")
    parser.add_argument("--mode", choices=["cli", "gui"], default="cli", 
                        help="Choose between 'cli' (command-line) and 'gui' (Gradio interface).")
    parser.add_argument("--input", type=str, help="User input for the chatbot (CLI mode only).")
    args = parser.parse_args()

    # Initialize parameters
    data, glove_path, tokenizer_path, sentiment_model_path, cuisines = initialize_params()

    # Gradio interface
    def gradio_chatbot(user_input):
        response = chatbot_response(user_input, data, glove_path, tokenizer_path, sentiment_model_path, cuisines)
        print(f"GUI Response: {response}")  # For debugging GUI outputs
        return response


    if args.mode == "gui":
        # Launch Gradio interface
        interface = gr.Interface(
            fn=gradio_chatbot,
            inputs="text",
            outputs="text",
            title="Restaurant Recommendation Chatbot",
            description="Ask about cuisines, restaurants, or popular dishes!"
        )
        interface.launch()

    else:
        # Command-line mode
        if args.input:
            print("Chatbot:", chatbot_response(args.input, data, glove_path, tokenizer_path, sentiment_model_path, cuisines))
        else:
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                print("Chatbot:", chatbot_response(user_input, data, glove_path, tokenizer_path, sentiment_model_path, cuisines))


if __name__ == "__main__":
    main()
