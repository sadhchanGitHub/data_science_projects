import nltk
from nltk.tokenize import word_tokenize

# Add the correct NLTK data path
nltk.data.path.append('/home/hyd_in_zrh/nltk_data')

# Sample text
sample_text = "This is a test sentence for tokenization."

# Tokenize
tokens = word_tokenize(sample_text)
print(tokens)
