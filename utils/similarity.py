from sklearn.preprocessing import normalize
import numpy as np
from sentence_transformers import util
from umap import UMAP

class Node_simliarity:
    def __init__(self, num_neighbor=10, threshold=0.00, metric="cosine"):
        self.num_neighbor = num_neighbor
        self.threshold = threshold
        self.metric = metric
    def dimension_reduction(self, embeddings):
        n_neighbors= 15
        n_components= 5
        random_state= None
        umap_embeddings = (UMAP(n_neighbors=n_neighbors,
                        n_components=n_components,
                        min_dist=0.0,
                        metric='cosine',
                        low_memory=True,
                        random_state=random_state)
                   .fit_transform(embeddings))
        return umap_embeddings
    def get_similar_index(self, node_embeddings):
        hypergraph = list()
        # node_embeddings = self.dimension_reduction(node_embeddings)
        for i, node_embedding in enumerate(node_embeddings):
            ranks = self.retrieve_similarities(node_embeddings, node_embedding, self.metric)
            indexes = np.flip(np.argsort(ranks)[-self.num_neighbor:])
            # indexes = np.argpartition(ranks, range(self.num_neighbor))[:self.num_neighbor] #working weired

            # recalculate the number of docuements to connect based on the fixed threshold
            # if self.threshold:
            #     limit = len(np.where(ranks >= self.threshold)[0])
            # indexes = np.flip(np.argsort(ranks)[-self.num_neighbor:])
            # indexes = indexes[:limit]
            # if len(indexes) < 2:
            #     continue;

            hypergraph.append(indexes.tolist())

        return hypergraph

    def retrieve_similarities(self, embeddings, embedding, metric):
        if metric == "product":
            ranks = self.product_similarity(embeddings, embedding)
        elif metric == "cosine":
            ranks = self.cosine_similarity(embeddings, embedding)
        elif metric == "euclidean":
            ranks = self.euclidean_distance(embeddings, embedding)
        else:
            print("not implemented metric is given")
        return ranks

    def product_similarity(self, embeddings, embedding):
        embeddings = normalize(embeddings)
        ranks = np.inner(embeddings, embedding)
        return ranks

    def cosine_similarity(self, embeddings, embedding):
        ranks = util.cos_sim(embeddings, embedding)
        ranks = ranks.reshape(-1)
        return ranks.cpu().detach().numpy()

    def euclidean_distance(self, embeddings, embedding):
        ranks = list()
        for compared_embedding in embeddings:
            dist = np.linalg.norm(embedding - compared_embedding)
            ranks.append(dist)
        return ranks
