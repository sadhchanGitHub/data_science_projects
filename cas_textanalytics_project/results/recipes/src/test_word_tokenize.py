from nltk.tokenize.punkt import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()
tokens = tokenizer.tokenize("This is a test sentence.")
print(tokens)
