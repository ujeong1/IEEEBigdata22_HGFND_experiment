import argparse
import warnings

from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from utils.data_loader import *
def edge_list():
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='politifact',
                        choices=['politifact', 'gossipcop'])
    parser.add_argument('--feature', type=str, default='bert',
                        choices=['profile', 'spacy', 'bert', 'content'])
    parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size for propagation encoding')
    parser.add_argument('--graph', type=bool, default=True, help='using news representation based on news propagation (not news content)')
    parser.add_argument('--concat', type=bool, default=True, help='concat contents and propagation for initial news representation')
    parser.add_argument('--model', type=str, default='SAGE', choices=['GCN', 'GAT', 'SAGE'])
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.01, help='l2 penalty')
    parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay')
    parser.add_argument('--lr_dc_step', type=int, default=20,
                        help='the number of steps after which the learning rate decay')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--epoch', type=int, default=60, help='epoch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffling training index')
    parser.add_argument('--use_user', action='store_true',
                        help='use shared user among news for building hyperedges')
    parser.add_argument('--use_interval', action='store_true', help='use rounded up timestamp of user tweet/retweet for building hyperedges')
    parser.add_argument('--num_views', type=int, default=0,
                        help='number of clustering results after running the different algorithms')
    parser.add_argument('--num_clusters', type=int, default=0,
                        help='the fixed number of clusters for KMEANS')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    #Experimental Parameters
    parser.add_argument('--train_node_ratio', type=float, default=1.0, help='used portion of news (nodes) for train datsaet. The maximum is 1.0')
    parser.add_argument('--hyperedge_ratio', type=float, default=1.0, help='used portion of hyperedges for the model. The maximum is 1.0')
    parser.add_argument('--filter_edge', type=bool, default=False,
                        help='Filter duplicate edges and edge with single node')
    parser.add_argument('--output_dir', type=str, default='graph',
                        help='The name of directory where the best checkpoint is saved')
    parser.add_argument('--save', type=bool, default=False, help='the checkpoint will be saved at /result')
    parser.add_argument('--num_run', type=int, default=0, help='Parameter used for getting the average of each run ')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    dataset = FNNDataset(root='data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())


    num_train = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_train + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])
    # all_set = training_set.union(validation_set.union(test_set))
    loader = DataLoader

    all_loader = loader(dataset, batch_size=len(dataset), shuffle=False)
    train_loader = loader(training_set, batch_size=args.batchSize, shuffle=True)
    val_loader = loader(validation_set, batch_size=args.batchSize, shuffle=False)
    test_loader = loader(test_set, batch_size=args.batchSize, shuffle=False)

    all_dataset = all_loader.dataset
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    import numpy as np
    def load_edgelist(G):
        edge_list_name = "../data/gossipcop/raw/"+"A.txt"
        edges_unordered = np.genfromtxt(edge_list_name, dtype=np.int32, delimiter=",")
        edges = edges_unordered.tolist()
        G.add_edges_from(edges)
        print(len(edges))

    newsList = list(training_set.indices)
    import networkx as nx
    import matplotlib.pyplot as plt
    import pickle
    from tqdm import tqdm
    with open("../data/hyperedges_pol_user.pkl", "rb") as f:
        hypergraph = pickle.load(f)

    G = nx.Graph()
    G.add_nodes_from(newsList)
    #create bipartite
    for user, hyperedge in tqdm(hypergraph.items()):
        G.add_node(user)
        for news in hyperedge:
            edge = (user, news)
            G.add_edge(*edge)
    users = list(hypergraph.keys())
    color_map=[]
    for node in G:
        if node in users:
            color_map.append('red')
        else:
            color_map.append('blue')
    nx.draw(G, node_color = color_map, pos=nx.spring_layout(G), node_size=2)
    plt.show()