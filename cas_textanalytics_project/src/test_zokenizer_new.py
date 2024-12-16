from nltk.tokenize.punkt import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()
sample_text = "This is a test sentence for tokenization."
tokens = tokenizer.tokenize(sample_text)
print(tokens)
