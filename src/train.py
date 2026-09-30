import argparse
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model.dpi_net import DPINet
import os

def dummy_dataset(num_samples=10):
    # Generates fake graph data to test the pipeline before real RGB-D data is ready
    from torch_geometric.data import Data
    dataset = []
    for _ in range(num_samples):
        num_nodes = 50
        num_edges = 200
        
        # Random node features: pos(3) + vel(3) + type(1)
        x = torch.randn((num_nodes, 7))
        # Random edges
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        # Random edge attributes: rel_pos(3) + dist(1)
        edge_attr = torch.randn((num_edges, 4))
        # Target displacement
        y = torch.randn((num_nodes, 3))
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        dataset.append(data)
    return dataset

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = DPINet(num_layers=3, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss() # Or Chamfer Distance from PyTorch3D

    # Load dataset (using dummy for now, swap with DexYCB dataloader later)
    dataset = dummy_dataset(100)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Predict displacements
            pred_disp = model(batch)
            
            # Mask out hand particles, we only predict object particle movement
            # Assuming batch.x[:, 6] == 1 means object particle
            object_mask = batch.x[:, 6] > 0.5
            
            if object_mask.sum() > 0:
                loss = criterion(pred_disp[object_mask], batch.y[object_mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/dpi_net_latest.pth')
    print("Training complete. Model saved.")

if __name__ == '__main__':
    train()
