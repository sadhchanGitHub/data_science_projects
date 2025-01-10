import dask.dataframe as dd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving/loading models
import os
import time

timestamp = int(time.time())

# Logging configuration
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"05_c_model3_logisticRegression_Balanced_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logging.info("called via 05_c_model3_logisticRegression_Balanced.py...\n")
logging.info(" Script Started ...\n")


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
logging.info("Training LogisticRegression Balanced...")
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)


# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Default threshold (0.5)
y_pred_default = (y_prob >= 0.5).astype(int)

# Adjusted threshold (e.g., 0.6)
y_pred_adjusted = (y_prob >= 0.6).astype(int)


# Evaluate the model
logging.info("Evaluating the model...")

# accuracy_report for Threshold 0.5
accuracy_report_threshold05 = accuracy_score(y_test, y_pred_default)
logging.info("Accuracy_report for Threshold 0.5 is as follows: %.2f%%" % (accuracy_report_threshold05 * 100))

# classification_report for Threshold 0.5
clf_report_threshold05 = classification_report(y_test, y_pred_default)
logging.info("Classification Report for Threshold 0.5 is as follows: ")
logging.info(clf_report_threshold05)

# accuracy_report for Threshold 0.6
accuracy_report_threshold06 = accuracy_score(y_test, y_pred_adjusted)
logging.info("Accuracy_report is for Threshold 0.6 as follows: %.2f%%" % (accuracy_report_threshold06 * 100))

# classification_report for Threshold 0.6
clf_report_threshold06 = classification_report(y_test, y_pred_adjusted)
logging.info("Classification Report for Threshold 0.6 is as follows: ")
logging.info(clf_report_threshold06)

logging.info("Not best compared to normal logistic regression in 05_b_model2_logisticRegression.py, so skipping to save")
# logging.info("saving the model and vectorizer...")

# Save the trained model and vectorizer
# joblib.dump(model, "../models/logistic_regression_balanced_model.pkl")
# joblib.dump(vectorizer, "../models/tfidf_vectorizer_using_LogisticRegressionbalanced_.pkl")

logging.info("script done \n")
