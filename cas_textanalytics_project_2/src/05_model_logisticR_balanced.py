import dask.dataframe as dd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving/loading models

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

# Train Logistic Regression
logging.info("Training LogisticRegression...")
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)


# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Default threshold (0.5)
y_pred_default = (y_prob >= 0.5).astype(int)

# Adjusted threshold (e.g., 0.6)
y_pred_adjusted = (y_prob >= 0.6).astype(int)

# Evaluate performance for both thresholds
from sklearn.metrics import classification_report
print("Default Threshold (0.5):\n", classification_report(y_test, y_pred_default))
print("Adjusted Threshold (0.6):\n", classification_report(y_test, y_pred_adjusted))


logging.info("saving the model and vectorizer...")

# Save the trained model and vectorizer
joblib.dump(model, "../models/logistic_regression_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

logging.info("script done...")
