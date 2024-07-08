# Pytorch 
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool
import torch.nn.functional as F
from layers import ImplicitGraph

# General	
import numpy as np	
import torch
from ignn_utils import aug_normalized_adjacency, sparse_mx_to_torch_sparse_tensor, get_spectral_rad
from scipy.sparse import coo_array
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, conv_type, dropout, kappa=0.9, adj_orig=None):
        super(TrainNet, self).__init__()

        self.num_nodes = 993
        self.conv_type = conv_type
        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        # GINConv
        if conv_type == "gin": 
            nn1 = nn.Sequential(nn.Linear(nfeat, nhid))
            self.conv1 = GINConv(nn1)
            nn2 = nn.Sequential(nn.Linear(nhid, nclass))
            self.conv2 = GINConv(nn2)

        # GCNConv
        if conv_type == "gcn":
            self.conv1 = GCNConv(nfeat, nhid)
            self.conv2 = GCNConv(nhid, nclass)

        # IGNN
        if conv_type == "ignn":
            self.ig1 = ImplicitGraph(nfeat, nhid, self.num_nodes, kappa)
            self.ig2 = ImplicitGraph(nhid, nhid, self.num_nodes, kappa)
            self.ig3 = ImplicitGraph(nhid, nhid, self.num_nodes, kappa)
            self.adj_orig = adj_orig
            self.adj_rho = None
            self.X_0 = None
            self.V_0 = nn.Linear(nhid, nhid)
            self.V_1 = nn.Linear(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, edge_index): # include batch parameter
        # self.adj_rho = 1
        # edge_list_file = '../datasets/em_user/edge_list.txt' # directly go from this -> adj
        # edges = np.loadtxt(edge_list_file, dtype=int)
        # G = nx.Graph()
        # G.add_edges_from(edges)
        # adj = nx.adjacency_matrix(G)
        # adj = torch.tensor(adj.todense()).to(device)
        
        # move this logic for adj outside of forward
        # collect subgraph information (masks)
        # iterate 
        edge_feats = [1] * edge_index.shape[1]
        edge_srcs = edge_index.cpu().numpy()[0]
        edge_dsts = edge_index.cpu().numpy()[1]
        N = self.num_nodes
        import pdb
        adj = aug_normalized_adjacency(coo_array((edge_feats, (edge_srcs, edge_dsts)), shape=(N,N)))
        adj = sparse_mx_to_torch_sparse_tensor(adj)     
        self.adj_rho = get_spectral_rad(adj)
        adj = adj.to(device)
        # print("ADJ: ", adj.shape[1])
        # print("x: ", x.shape)
        if self.conv_type == "gin" or self.conv_type == "gcn":
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p = self.dropout, training = self.training)
            return self.conv2(x, edge_index)

        elif self.conv_type == "ignn":
            x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
            x = self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
            x = self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
            # print("Batch: ", batch)

            # this is the baseline
            '''
            take in a batch list batch[i] corresponds to index for subgraph
            treat it as a mask
            '''
            # x = global_add_pool(x, batch)
            x = F.relu(self.V_0(x))
            
            # after batch probably shouldnt do dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.V_1(x)
            
            return F.log_softmax(x, dim=1)
            
            '''
            task now is to directly predict the subgraph classes
            '''
