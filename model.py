import torch
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (
    add_self_loops, sort_edge_index, remove_self_loops)
from torch_sparse import spspmm


class GNN(torch.nn.Module):
    def __init__(self, indim, ratio, nclass=2):
        '''
        Graph Neural Network (GNN) for graph level classification.

        Parameters:
        - indim (int): Node feature dimension.
        - ratio (float): Pooling ratio in the range (0, 1).
        - nclass (int): Number of classes.

        '''
        super(GNN, self).__init__()

        # Define model dimensions
        self.indim = indim
        self.dim1 = 32
        self.dim2 = 8

        # First GAT layer and TopKPooling
        self.conv1 = GATConv(self.indim, self.dim1)
        self.pool1 = TopKPooling(
            self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

        # Fully connected layers for final classification
        self.fc1 = torch.nn.Linear((self.dim1)*2, self.dim1)
        self.bn1 = torch.nn.BatchNorm1d(self.dim1)
        self.fc2 = torch.nn.Linear(self.dim1, self.dim2)
        self.bn2 = torch.nn.BatchNorm1d(self.dim2)
        self.fc3 = torch.nn.Linear(self.dim2, nclass)

    def forward(self, x, edge_index, batch, edge_attr):
        '''
        Forward pass of the GNN.
        If `batch` is more than one, the all the graphs are concatenated together.
        `batch` This is basically an array that tells us which nodes are part of which graph
        e.g. [0, 0, 0, 1, 1, 1, ..., 31, 31, 31] here the first three nodes are form graph 0 and so on.


        Parameters:
        - x (Tensor): Node features. x has shape [|n_batch|*|N|, in_channels]
        - edge_index (LongTensor): Graph edge indices. edge_index has shape [2, |E|*|n_batch|].
          It is in COO format where each column represents an edge from the node in
          the first row to the node in the second row of that column.
        - batch (LongTensor): Batch assignment for each node.
        - edge_attr (Tensor): Edge attributes. edge_attr has shape [1, |E|*|n_batch|] which is
          the value of the correlation for each edge.

        Returns:
        - x (Tensor): Output logits for each node.
        - pool1_weight (Tensor): Attention weights from the first pooling layer.
        - pool2_weight (Tensor): Attention weights from the second pooling layer.
        - score1 (Tensor): Sigmoid scores from the first pooling layer.
        - score2 (Tensor): Sigmoid scores from the second pooling layer.

        '''
        # First GAT layer and TopKPooling
        # Convert graph(s) with shape [|n_batch|*|N|, in_channels] to graph(s) with shape [|n_batch|*|N|, dim1]
        # This only reduces the node embeddings and doesn't change the edge configuration.
        x = self.conv1(x, edge_index, edge_attr) 

        # Reduces the number of nodes to |batch|*(|N|*ratio). The embedding vector size stays the same.
        # The number of edges, however, cannot be computed before hand as not all nodes that drop based on `ratio` have
        # the same number of edges.
        # 'perm' gives the index of the ROIs (nodes)  in the original graph that are left after pruning. It has shape [|batch|*(|N|*ratio)]. This
        # is useful later on, when we want to know which ROIs are kept.
        # 'score' has shape [|batch|*(|N|*ratio)] as well.
        x, edge_index, edge_attr, batch, perm1, score1 = self.pool1(
            x, edge_index, edge_attr, batch)

        # https://stats.stackexchange.com/questions/601997/how-to-interpret-the-global-max-pooling-operation-in-graph-neural-networks
        # For each graph(s) takes the max and mean of the embedding vectors accross the nodes. This reduces all the node embedding in 
        # a graph to a single node.
        # In x1 for each graph in the batch there is a single node embedding that is the concat of gap and gmp nodes of that graph.
        # So x1 would be of shappe [|batch|, 2*dim1]
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Concatenate and apply fully connected layers
        # The final summary vector is obtained as the concatenation of all the previous layer summaries
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, p=0.7, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x, self.pool1.weight, torch.sigmoid(score1).view(x.size(0), -1), perm1.view(x.size(0), -1)