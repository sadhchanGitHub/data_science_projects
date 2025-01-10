import dask.dataframe as dd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import os
import joblib  # For saving/loading models

timestamp = int(time.time())

# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"05_b_model2_logisticRegression_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logging.info("called via 05_b_model2_logisticRegression.py...\n")
logging.info(f" Started ...\n")



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

# accuracy_report
accuracy_report = accuracy_score(y_test, y_pred)
logging.info("Accuracy_report is as follows: %.2f%%" % (accuracy_report * 100))

# classification_report
clf_report = classification_report(y_test, y_pred)
logging.info("Classification Report is as follows: ")
logging.info(clf_report)


logging.info("This is the best model among the 3 trails, lets save this as logistic_regression_model.pkl and its vectorizertfidf_vectorizer_using_LogisticRegression.pkl \n ")

joblib.dump(model, "../models/logistic_regression_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer_using_LogisticRegression.pkl")


logging.info(f"Ended \n")
