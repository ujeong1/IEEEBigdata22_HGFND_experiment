import numpy as np
import scipy.sparse as sp
from gensim.utils import *
from itertools import chain
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import ListedColormap
# import seaborn as sns
from umap import UMAP
import hdbscan
import random


class MultiviewClustering:
    def __init__(self, num_views=5, n_clusters=15):
        self.num_views = num_views
        self.n_clusters = n_clusters
        self.random_state = random.randint(0, 100)

    def cluster_transform(self, embeddings):
        multiview_clusters = []
        for i in range(self.num_views):
            string = "Creating {}-th hyperedge out of {} views.".format(str(i + 1), str(self.num_views))
            print(string)
            cluster_labels = self.generate_cluster_labels(embeddings, n_neighbors=15, n_components=5, min_cluster_size=5)
            multiview_clusters.append(cluster_labels)

        return multiview_clusters

    def generate_cluster_labels(self, message_embeddings, n_neighbors, n_components, min_cluster_size, random_state=None):
        candidate_models = ["HDBSCAN", "KMEANS"]
        model = candidate_models[1]
        if model == candidate_models[0]:
            umap_embeddings = (UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components,
                                    min_dist=0.0,
                                    metric='cosine',
                                    low_memory=True,
                                    random_state=random_state)
                               .fit_transform(message_embeddings))

            cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                      metric='euclidean',
                                      cluster_selection_method='eom').fit(umap_embeddings)
            cluster_labels = cluster.labels_
        elif model == candidate_models[1]:
            cluster = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(message_embeddings)
            cluster_labels = cluster.labels_

        else:
            print("Not implemented clustering algorithm given")
        return cluster_labels

    def get_hypergraph_from_cluster(self, clusters_labels, num_nodes=128):
        hypergraph = []
        nodes_seq = np.arange(
            num_nodes)  # Let's change it to document id next so that we can explain how it contributed the model
        for cluster_labels in clusters_labels:
            cluster_node_pairs = []
            labels = cluster_labels
            for label, seq_num in zip(labels, nodes_seq):
                cluster_node_pair = [seq_num, label]
                cluster_node_pairs.append(cluster_node_pair)
            values = set(map(lambda x: x[1], cluster_node_pairs))
            hyperedge = [[y[0] for y in cluster_node_pairs if y[1] == x] for x in values]
            hypergraph += hyperedge
        return hypergraph

    @staticmethod
    def normalizer(X):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(X)
        X = normalize(scaled_features)
        return X

    # @staticmethod
    # def plotDBSCAN( X, labels):
    #     sns.color_palette("Paired")
    #     sns.scatterplot(X[:, 0], X[:, 1], hue=["cluster-{}".format(x) for x in labels])
    #     plt.show()
    #     # fig = pyplot.figure()
    #     # ax = Axes3D(fig)
    #     # cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
    #     # ax.scatter(X[:,0], X[:,1],X[:,2])#, c=X, marker='o', cmap=cmap, alpha=1, label=labels)
    #     # pyplot.show()
    #     # fig.savefig('DBSCAN.png', dpi=fig.dpi)

    @staticmethod
    def epsilonSearch():
        # for i in tqdm(np.arange(0.001, 1, 0.002)):
        #     dbscan = DBSCAN(eps=i, min_samples=2, metric='cosine').fit(X)
        #     n_classes.update({i: len(pd.Series(dbscan.labels_).value_counts())})
        assert NotImplementedError

    @staticmethod
    def getClusterGroup(sentences, labels, group_num):
        results = pd.DataFrame({'label': labels, 'sent': sentences})
        samples = results[results.label == group_num].sent.tolist()
        # event_df = df[df.id.isin(samples)][['id']]
        # event_df['id'] = pd.to_datetime(event_df.id)
        # event_df = event_df.sort_values(by='date').dropna()
        return samples
