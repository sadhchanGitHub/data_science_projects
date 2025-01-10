from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import logging

def visualize_embeddings(tokenizer, embedding_matrix, words_to_plot, output_path="pca_data.npz"):
    vectors = [embedding_matrix[tokenizer.word_index[word]] for word in words_to_plot if word in tokenizer.word_index]
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Save PCA data for visualization in a notebook
    np.savez(output_path, words=words_to_plot, vectors=reduced_vectors)
    print(f"PCA data saved to {output_path}")

    # Plot and save figure
    plt.figure(figsize=(8, 6))
    for word, vector in zip(words_to_plot, reduced_vectors):
        plt.scatter(vector[0], vector[1])
        plt.text(vector[0] + 0.01, vector[1] + 0.01, word, fontsize=12)
    plt.title("Word Embeddings Visualization")
    plt.savefig("../outputs/tier2_01_execute_CNN_model_word_embeddings_visualization.png")
    logging.info("Visualization saved as '../outputs/tier2_01_execute_CNN_model_word_embeddings_visualization.png'.")


