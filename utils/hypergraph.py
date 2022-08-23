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
    def statics(self, hypergraph):
        total_nodes = []
        print("Number of edges", len(hypergraph))
        for nodes in hypergraph:
            total_nodes.append(len(nodes))
        import statistics
        avg_edge_degree = statistics.mean(total_nodes)
        edge_std = statistics.stdev(total_nodes)
        max_edge_degree = max(total_nodes)
        node_ids = []
        for nodes in hypergraph:
            node_ids+=nodes
        node_ids = list(set(node_ids))

        node_id_dict = dict()
        for node_id in node_ids:
            node_id_dict[node_id] = 0
        for nodes in hypergraph:
            for node_id in nodes:
                node_id_dict[node_id]+=1
        num_nodes = len(node_ids)

        avg_node_degree = sum(node_id_dict.values())/num_nodes
        node_std = statistics.stdev(node_id_dict.values())
        max_node_degree = max(node_id_dict.values())
        print(avg_node_degree,max_node_degree, node_std, avg_edge_degree, max_edge_degree, edge_std)
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
        # self.statics(result)
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
        dirname = "data/"
        if self.dataset == "politifact":
            filename = "hyperedges_pol_entity.pkl"
        elif self.dataset == "gossipcop":
            filename = "hyperedges_gos_entity.pkl"

        with open(dirname + filename, 'rb') as handle:
            data = pickle.load(handle)

        hyperedges = []
        for hyperedge in data.values():
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
