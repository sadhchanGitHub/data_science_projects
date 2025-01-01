import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
df_cleaned = pd.read_csv("../data/New_York_reviews_cleaned_with_spacy.csv")

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned text
X = vectorizer.fit_transform(df_cleaned['cleaned_review'])

# Map target labels (Positive -> 1, Negative -> 0)
y = df_cleaned['sample'].map({'Positive': 1, 'Negative': 0}).values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save TF-IDF features and labels
np.save("../data/X_train_tfidf.npy", X_train.toarray())
np.save("../data/X_test_tfidf.npy", X_test.toarray())
np.save("../data/y_train.npy", y_train)
np.save("../data/y_test.npy", y_test)

print("TF-IDF vectorization completed and files saved!")
