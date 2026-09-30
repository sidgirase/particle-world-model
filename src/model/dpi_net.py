import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch_geometric.transforms as T

class ParticleEdgeNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        # Input: sender node + receiver node + edge attributes
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x_i, x_j, edge_attr):
        # x_i: Receiver node features, x_j: Sender node features
        input_feats = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(input_feats)

class ParticleNodeNetwork(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        # Input: node features + aggregated messages
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, aggregated_messages):
        input_feats = torch.cat([x, aggregated_messages], dim=-1)
        return self.mlp(input_feats)

class DPINetLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='mean') # Mean aggregation of messages
        self.edge_net = ParticleEdgeNetwork(node_dim, edge_dim, hidden_dim)
        self.node_net = ParticleNodeNetwork(node_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # Step 1: Message Passing
        # x: Node features [N, node_dim]
        # edge_index: Graph connectivity [2, E]
        # edge_attr: Edge features (e.g., relative distance) [E, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Step 2: Compute messages on edges
        return self.edge_net(x_i, x_j, edge_attr)

    def update(self, aggr_out, x):
        # Step 3: Update node features using aggregated messages
        return self.node_net(x, aggr_out)

class DPINet(nn.Module):
    """
    Dynamic Particle Interaction Network (DPI-Net)
    Predicts the displacement of object particles given hand/object particles.
    """
    def __init__(self, num_layers=3, hidden_dim=128):
        super().__init__()
        # Node features: pos (3) + velocity (3) + type (1) = 7
        self.node_dim = 7
        # Edge features: relative pos (3) + distance (1) = 4
        self.edge_dim = 4
        
        self.node_encoder = nn.Linear(self.node_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            DPINetLayer(hidden_dim, self.edge_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Predicts next position displacement for particles
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) 
        )

    def forward(self, data):
        """
        data: torch_geometric.data.Data containing:
        - data.x: Node features [N, 7]
        - data.edge_index: Connectivity [2, E]
        - data.edge_attr: Edge features [E, 4]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encode nodes
        x = self.node_encoder(x)
        
        # Message passing layers
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_attr) # Residual connection
            
        # Decode to position displacement
        predicted_displacement = self.decoder(x)
        
        return predicted_displacement
