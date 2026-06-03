import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

TOP_N_GENES = 100  # must match build_graph.py


class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(TOP_N_GENES, 64, heads=2, concat=False)
        self.conv2 = GATConv(64, 32, heads=2, concat=False)
        self.conv3 = GATConv(32, 2, heads=1, concat=False)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x