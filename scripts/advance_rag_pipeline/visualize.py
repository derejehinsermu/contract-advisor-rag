import matplotlib.pyplot as plt
import numpy as np
import umap
from tqdm import tqdm

def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def visualize_embeddings(
    original_query, original_query_embedding, augmented_queries, augmented_query_embeddings, 
    result_embeddings, dataset_embeddings, umap_transform
):
    # Project embeddings
    project_original_query = project_embeddings([original_query_embedding], umap_transform)
    project_augmented_queries = project_embeddings(augmented_query_embeddings, umap_transform)
    projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)
    projected_dataset_embeddings = project_embeddings(dataset_embeddings, umap_transform)
    
    # Plot the projected queries and retrieved documents in the embedding space
    plt.figure(figsize=(10, 8))
    plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, alpha=0.2, label='Dataset Embeddings')
    plt.scatter(project_augmented_queries[:, 0], project_augmented_queries[:, 1], s=150, marker='X', color='orange', label='Augmented Query Embeddings')
    plt.scatter(projected_result_embeddings[:, 0], projected_result_embeddings[:, 1], s=100, facecolors='none', edgecolors='g', label='Retrieved Embeddings')
    plt.scatter(project_original_query[:, 0], project_original_query[:, 1], s=150, marker='X', color='r', label='Original Query Embedding')

    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{original_query}', fontsize=14)
    plt.legend()
    plt.axis('off')
    plt.show()
