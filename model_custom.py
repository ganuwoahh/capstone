import json
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load the dataset
file_path = "screenshot_data_color.xlsx"
df = pd.read_excel(file_path)

# Remove the first 8 columns (those are not useful for prediction)
df = df.iloc[:, 8:]

# Ensure all edge columns are categorical (green, yellow, orange, red)
color_mapping = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}

# Convert color labels to integers in the edge columns
df = df.applymap(lambda x: color_mapping.get(x, -1))  # Convert color labels to integers

# Define number of nodes based on the dataset columns
num_nodes = len(df.columns)  # Number of edges (since each column corresponds to an edge)

# Prepare sequential data where current timestep (t) predicts the next timestep (t+1)
X_data = []  # Feature data for current timestep
y_data = []  # Target data (edges' colors for next timestep)

for i in range(len(df) - 1):  # Iterate through the rows, using each row as the current timestep
    X_data.append(df.iloc[i].values)  # Current timestep data (edges at time t)
    y_data.append(df.iloc[i + 1].values)  # Next timestep data (edges at time t+1)

# Convert to numpy arrays and then to PyTorch tensors
X_data = np.array(X_data)
y_data = np.array(y_data)

# Convert input features (X_data) and labels (y_data) to PyTorch tensors
X_data = torch.tensor(X_data, dtype=torch.float)
y_data = torch.tensor(y_data, dtype=torch.long)

# Now we need to prepare the edge index
# Assume that each row of nodes is connected in a chain: node 0 to node 1, node 1 to node 2, ...
edges = []
for i in range(num_nodes - 1):  # Create a chain of edges between consecutive nodes
    edges.append([i, i + 1])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Create node features directly from the edge color for each node at the current timestep
node_features = torch.zeros((num_nodes, 4), dtype=torch.float)  # 4 classes: green, yellow, orange, red

# Iterate through each timestep to update node features
for t in range(len(df) - 1):  # Iterate over timesteps
    for i, edge_color in enumerate(df.iloc[t].values):  # Iterate over edges in the current timestep
        node_features[i][edge_color] = 1  # Set the node feature corresponding to the edge color

# Now, create the PyTorch Geometric Data object for each sequence
sequences = []
for i in range(X_data.shape[0]):  # Iterate through each timestep
    data = Data(x=node_features, edge_index=edge_index, y=y_data[i])  # Use the same edge_index for each timestep
    sequences.append(data)

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the model
model = GCN(num_node_features=node_features.size(1), num_classes=4)  # 4 classes: green, yellow, orange, red

# Set up the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Initialize dictionary to save metrics
metrics = {
    "train_loss": [],
    "test_loss": [],
    "mae": [],
    "mape": [],
    "accuracy": []
}

# Training loop
for epoch in range(800):
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    correct = 0
    total = 0

    # Process each sequence
    for sequence in sequences:
        out = model(sequence.x, sequence.edge_index)

        # We are using edge colors as labels
        loss = criterion(out.view(-1, 4), sequence.y.view(-1))  # Flatten both output and target
        total_loss += loss.item()

        # Get predictions (class with the highest probability)
        _, predicted = torch.max(out, dim=1)

        # Calculate accuracy
        correct += (predicted == sequence.y).sum().item()
        total += sequence.y.size(0)

        loss.backward()

    optimizer.step()

    # Calculate accuracy for the current epoch
    accuracy = 100 * correct / total

    # Append metrics for the current epoch
    metrics["train_loss"].append(total_loss / len(sequences))
    metrics["accuracy"].append(accuracy)

    # Placeholder values for test_loss, mae, and mape (update these if you have validation logic)
    metrics["test_loss"].append(0.0)
    metrics["mae"].append(0.0)
    metrics["mape"].append(0.0)

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {metrics['train_loss'][-1]:.4f}, Accuracy: {metrics['accuracy'][-1]:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "traffic_gcn_model.pth")

# Save metrics to a JSON file
with open("traffic_gcn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model and metrics saved!")
