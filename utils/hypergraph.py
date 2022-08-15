import numpy as np
import scipy.sparse as sp
from utils.clustering import MultiviewClustering
import torch
import pickle
from utils.similarity import Node_simliarity


class Hypergraph:
    def __init__(self, opt):
        self.use_user = opt.use_user
        self.use_date = opt.use_date
        self.use_entity = opt.use_entity
        self.dataset = opt.dataset

    def get_hyperedges(self, not_train_idx):
        hyperedges = []

        if self.use_user:
            H = self.get_user_incidence_matrix()
            hyperedges += H

        if self.use_date:
            H = self.get_date_incidence_matrix()
            hyperedges += H

        if self.use_entity:
            H = self.get_entity_incidence_matrix()
            hyperedges += H

        result = list()
        for hyperedge in hyperedges:
            hyperedge = list(set(hyperedge).difference(set(not_train_idx)))
            #Filter hyperedges less than 2 nodes according to the definition of hypergraph
            if len(hyperedge) < 2:
                continue
            result.append(hyperedge)

        return result

    def get_user_incidence_matrix(self):
        dirname = "data/"
        if self.dataset == "politifact":
            filename = "hyperedges_pol_user.pkl"
        elif self.dataset == "gossipcop":
            filename = "hyperedges_gos_user.pkl"

        with open(dirname + filename, 'rb') as handle:
            data = pickle.load(handle)

        hyperedges = []
        for hyperedge in data.values():
            hyperedges.append(hyperedge)

        return hyperedges

    def get_date_incidence_matrix(self):
        dirname = "data/"
        if self.dataset == "politifact":
            filename = "hyperedges_pol_date.pkl"
        if self.dataset == "gossipcop":
            filename = "hyperedges_gos_date.pkl"

        with open(dirname + filename, "rb") as f:
            data = pickle.load(f)

        hypergraph = []

        for hyperedge in data.values():
            hypergraph.append(hyperedge)
        return hypergraph

    def get_entity_incidence_matrix(self):
        threshold = 3
        dirname = "data/"
        if self.dataset == "politifact":
            filename = "hyperedges_pol_entity.pkl"
        elif self.dataset == "gossipcop":
            filename = "hyperedges_gos_entity.pkl"

        with open(dirname + filename, 'rb') as handle:
            data = pickle.load(handle)

        hyperedges = []
        for hyperedge in data.values():
            if len(hyperedge) >= threshold:
                continue
            hyperedges.append(hyperedge)

        return hyperedges

    def get_adj_matrix(self, hyperedges, nodes_seq):
        items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []

        node_list = nodes_seq
        node_set = list(set(node_list))
        node_dic = {node_set[i]: i for i in range(len(node_set))}

        rows = []
        cols = []
        vals = []
        max_n_node = len(node_set)
        max_n_edge = len(hyperedges)
        total_num_node = len(node_set)

        # num_hypergraphs can be used for batching different size of hypergraphs for training
        num_hypergraphs = 1
        for idx in range(num_hypergraphs):
            # e.g., hypergraph = [[12, 31, 111, 232],[12, 31, 111, 232],[12, 31, 111, 232] ...]
            for hyperedge_seq, hyperedge in enumerate(hyperedges):
                # e.g., hyperedge = [12, 31, 111, 232]
                for node_id in hyperedge:
                    rows.append(node_dic[node_id])
                    cols.append(hyperedge_seq)
                    vals.append(1)
            u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))
            alias_inputs.append([j for j in range(max_n_node)])
            node_masks.append([1 for j in range(total_num_node)] + (max_n_node - total_num_node) * [0])

        return alias_inputs, HT, node_masks
