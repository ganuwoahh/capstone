import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import numpy as np
import networkx as nx
import pandas as pd
import json

# Load dataset
csv_path = "PEMS08.csv"
npz_path = "PEMS08.npz"

# Load graph from CSV
pems08_csv = pd.read_csv(csv_path)
G = nx.DiGraph()
for _, row in pems08_csv.iterrows():
    G.add_edge(row['from'], row['to'], weight=row['cost'])

# Convert to edge index and weights
edge_index = torch.tensor(list(G.edges)).t().contiguous().long()
edge_weight = torch.tensor([d['weight'] for _, _, d in G.edges(data=True)], dtype=torch.float)

# Load node features
pems08_npz = np.load(npz_path)
node_features = pems08_npz['data']  # Shape: (17856, 170, 3)

# Normalize node features
min_vals = node_features.min(axis=(0, 1), keepdims=True)
max_vals = node_features.max(axis=(0, 1), keepdims=True)
normalized_features = (node_features - min_vals) / (max_vals - min_vals)

# Prepare input (X) and target (y)
X = normalized_features[:-1]  # Features up to the second-to-last time step
y = normalized_features[1:]   # Targets from the second time step onwards

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert datasets to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float)
y_train, y_test = torch.tensor(y_train, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)


def compute_metrics(predictions, targets):
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(predictions - targets))
    
    # Mean Absolute Percentage Error (handle small targets gracefully)
    mape = torch.mean(torch.abs((predictions - targets) / torch.clamp(targets, min=0.1))) * 100
    
    # Accuracy (percentage of predictions within an absolute error margin of 0.1)
    within_margin = torch.abs(predictions - targets) <= 0.1
    accuracy = torch.mean(within_margin.float()) * 100
    
    return mae.item(), mape.item(), accuracy.item()


# Define the Temporal GCN model
class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes):
        super(TemporalGCN, self).__init__()
        self.num_nodes = num_nodes
        
        # Graph convolution layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Temporal modeling with GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        batch_size, num_nodes, feature_dim = x.size()
        
        # Reshape for GCN processing (combine batch and node dimensions)
        x = x.view(-1, feature_dim)  # Shape: (batch_size * num_nodes, feature_dim)
        
        # Apply GCN layers
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        
        # Reshape back for temporal processing
        x = x.view(batch_size, num_nodes, -1)
        
        # Temporal modeling with GRU (process node-wise)
        x, _ = self.gru(x)  # Shape: (batch_size, num_nodes, hidden_dim)
        
        # Prediction
        x = self.fc(x)  # Shape: (batch_size, num_nodes, output_dim)
        return x



# Model initialization
input_dim = 3   # Number of features per node
hidden_dim = 64 # Hidden dimension for GCN and GRU layers
output_dim = 3  # Predicting the same 3 features for the next time step
num_nodes = 170

model = TemporalGCN(input_dim, hidden_dim, output_dim, num_nodes)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
metrics = {
    "train_loss": [],
    "test_loss": [],
    "mae": [],
    "mape": [],
    "accuracy": []
}

# Training loop
num_epochs = 75
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X_train, edge_index)
    loss = criterion(predictions, y_train)
    metrics["train_loss"].append(loss.item())  # Save train loss
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test, edge_index)
        test_loss = criterion(test_predictions, y_test)
        metrics["test_loss"].append(test_loss.item())  # Save test loss
        
        # Compute additional metrics
        mae, mape, accuracy = compute_metrics(test_predictions, y_test)
        metrics["mae"].append(mae)
        metrics["mape"].append(mape)
        metrics["accuracy"].append(accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, "
          f"Test Loss: {test_loss.item():.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, Accuracy: {accuracy:.2f}%")

# Save metrics to a JSON file
with open("temporal_gcn_metrics_PEMS.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save the trained model
torch.save(model.state_dict(), "temporal_gcn_model1.pth")
print("Model training complete and saved!")