import pandas as pd
import joblib

# Load the saved model and vectorizer
model = joblib.load("../models/logistic_regression_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

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
ranked_restaurants = sentiment_summary.sort_values(by='positive_percentage', ascending=False)

# Save the ranking for further use
ranked_restaurants.to_csv("../data/ranked_restaurants.csv")

print(ranked_restaurants.head())
