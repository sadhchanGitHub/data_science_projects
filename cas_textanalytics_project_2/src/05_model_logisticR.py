import dask.dataframe as dd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your dataset as a Dask DataFrame
df_train = dd.read_csv("../data/New_York_reviews_cleaned_with_spacy.csv")

# Add explicit metadata for the 'sample' column
df_train['sample'] = df_train['sample'].map({'Positive': 1, 'Negative': 0}, meta=('sample', 'int64'))

# Initialize TfidfVectorizer
logging.info("Initializing TfidfVectorizer...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_train['cleaned_review'].compute())
y = df_train['sample'].compute()

# Split the data
logging.info("Splitting the data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train Multinomial Naïve Bayes
logging.info("Training LogisticRegression...")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
logging.info("Testing the model...")
y_pred = model.predict(X_test)

# Evaluate the model
logging.info("Evaluating the model...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
