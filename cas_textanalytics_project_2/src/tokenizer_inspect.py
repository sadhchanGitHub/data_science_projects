def inspect_tokenizer(tokenizer):
    print("Top 10 words in the tokenizer vocabulary:")
    for word, index in list(tokenizer.word_index.items())[:10]:
        print(f"{word}: {index}")
