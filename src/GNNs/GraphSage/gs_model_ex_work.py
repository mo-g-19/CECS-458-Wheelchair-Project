import torch
from torch import nn
from torch_geometric.nn import Data 
from torch_gepometric.nn import SAGEConv

#Creating a tiny graph with 4 nodes to work on

#Nodes:
#  Node 0: Resturaunt
#  Node 1: Resturanut
#  Node 2: Customer B
#  Node 3: Customer C

#Iniial node features (embeddings) where the ratio is accessible:popular
#  Resturaunt: [0.8, 0.2]   (accessible but mediocre popularity)
#  Resturanut: [0.4, 0.6]   (somewhat accessible and moderate popularity)
#  Customer B: [1.0, 0.0]   (Only focused on accessibility)
#  Customer C: [0.5, 0.5]   (Equal focus on accessibility and popularity))

x = torch.tensor([ 
    [0.8, 0.2],     #0
    [0.4, 0.6],     #1
    [1.0, 0.0],     #2
    [0.5, 0.5]      #3
], dtype=torch.float)

#Undirected edges representing interactions nearby

edges = [
    (2,0), (0,2),    #Customer B <-> Resturaunt
    (3,0), (0,3),    #Customer C <-> Restura
    (3, 1), (1,3),   #Customer C <-> Resturanut
    (0,1), (1,0)     #Resturaunt <-> Resturanut
]

edges_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Targets: regress restaurant accessibility (only for restaurants)
# Let's treat the first feature as "accessibility score"
y = torch.tensor([0.8, 0.4, 0.0, 0.0], dtype=torch.float).view(-1, 1)

# Masks: train only on restaurant nodes; users are unlabeled (context only)
is_restaurant = torch.tensor([True, True, False, False])
train_mask = is_restaurant.clone()   # in this toy, we train on both restaurants
test_mask = torch.tensor([False, False, False, False])  # not meaningful here, but shown for completeness

data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.test_mask = test_mask

#Defining the actual tiny GraphSAGE model (above will get replaced by API calls later)
class GraphSAGERegressor(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, out_dim=8):
        super().__init__()

        #Doing default mean aggregation -> update own features and neghborhood features
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        #Using ReLU activation: learned in class for CNN and how still effective and less computationally expensive here
        self.act1 = nn.ReLU()
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.act2 = nn.ReLU()

        #Regression head to predict accessibility score
        self.head = nn.Sequential(
            nn.Linear(out_dim, 1)  # Output single value for regression
        )

    def forward(self, x, edge_index):
        #Forward pass through GraphSAGE layers (same explaination as init)
        h = self.conv1(x, edge_index)   #aggregate
        h = self.act1(x)
        h = self.conv2(h, edge_index)   #aggreatate again
        h = self.act2(h)

        #Predict scalar per node (only compute loss on resturaunts)
        y_hat = self.head(h)
        return h, y_hat
    
model = GraphSAGERegressor(in_dim=data.num_node_features, hidden_dim=16, out_dim=8)

#Train the model (Mean Squared Error on resturaunts only)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("Initial node features:\n", data.x)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()

    #Forward pass
    h, y_hat = model(data.x, data.edge_index)

    #Compute loss only on restaurant nodes
    loss = loss_fn(y_hat[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        with torch.no_grad():
            mse = loss.item()
        print(f"Test MSE: {mse:.4f}")

#Inspect the learned node embeddings and predictions
model.eval()
with torch.no_grad():
    h, y_hat = model(data.x, data.edge_index)
    #print("\nLearned node embeddings:\n", h)
    #print("\nPredicted accessibility scores:\n", y_hat)\
    
print("\nLearned node embeddings:\n")
for i, name in enumerate(["Resturaunt 0", "Resturanut 1", "Customer B 2", "Customer C 3"]):
    print(f"{name}: {h[i].numpy()}")

print("\nPredicted accessibility scores (y_hat) vs target (y) for resturaunts only:\n")
for i, name in enumerate(["Resturaunt 0", "Resturanut 1"]):
    print(f"{name}: Predicted: {y_hat[i].item():.4f}, Target: {data.y[i].item():.4f}")

#Show the raw 1-hop mean aggreagation for R1 (layer 0 to layer 1)

with torch.no_grad():
    #Manually compute SAGEConv( x ) aggregator part for R1 BEFORE learned linear transform
    #SAGEConv does: h_v = W * [x_v || mean_{u in N(v)} x_u], then nonlinearity
    #Here only illustrate mean of neighbors for R1 on raw x
    neighbors_R1 = [2, 3, 1] #Cust B, Cust C, Resturanut 2
    raw_neighbors = data.x[neighbors_R1]
    mean_agg_R1 = raw_neighbors.mean(dim=0)
    print("\nManual neighbor mean for R1 on raw features (before learned transform):\n", raw_neighbors.numpy())
    print("     neighbors(Cust B, Cust C, Resturanut) mean = ", mean_agg_R1.numpy())