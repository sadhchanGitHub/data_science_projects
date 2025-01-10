import pandas as pd
import joblib
import os
import time
import logging

timestamp = int(time.time())

# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"06_loadModel_generate_Restaurants_list_sorted_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logging.info("called via 06_loadModel_generate_Restaurants_list_sorted.py...\n")
logging.info(" Script Started ...\n")

logging.info("This script will will load the logistic regression model, best among 3 trails and will generate /outputs/ranked_restaurants.csv sorted by positive sentiments in descending order...\n")

# Load the saved model and vectorizer
model = joblib.load("../models/logistic_regression_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer_using_LogisticRegression.pkl")

# Load the dataset for aggregation
df = pd.read_csv("../data/New_York_reviews_cleaned_with_spacy.csv")

# Transform the reviews using the saved vectorizer
X = vectorizer.transform(df['cleaned_review'])

# Predict sentiments
y_pred = model.predict(X)

# Add predictions to the dataframe
df['predicted_sentiment'] = y_pred

# Group by restaurant and calculate sentiment percentages
sentiment_summary = df.groupby('restaurant_name')['predicted_sentiment'].value_counts(normalize=True).unstack(fill_value=0)
sentiment_summary.columns = ['negative_percentage', 'positive_percentage']  # Rename for clarity

# Round percentages to 2 decimal places
sentiment_summary = sentiment_summary.round({'negative_percentage': 2, 'positive_percentage': 2})

# Add total review counts for each restaurant
review_counts = df.groupby('restaurant_name')['predicted_sentiment'].count()
sentiment_summary['total_reviews'] = review_counts

# Rank restaurants by positive sentiment percentage
restaurants_list_sorted = sentiment_summary.sort_values(by='positive_percentage', ascending=False)

# Save the ranking for further use
restaurants_list_sorted.to_csv("../outputs/restaurants_list_sorted.csv")

logging.info("sample of the best restaurants saved is as follows: ")
logging.info(restaurants_list_sorted.head())

logging.info("script done \n")

