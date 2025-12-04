### GCN Purpose
The Graph Convolutional Network works by analyzing the spatial relationship between nodes (similiar to the Convolutional Neural Network except it is nodes in a graph instead of pixels in a photo). For each update, the nodes' embedding is the average of its neighbors and itself based on the weighted connection.

Compared to the other GNNs, this DL's purpose is to take a subgraph of all the networks, small representation of the total graphs, and create a small list for the user to see. The GCN specifically focuses on the Resturaunt to Region relationship (how close a resturaunt is from the location the user gives) and the Resturaunt to Resturaunt relationship (based on either the location or the type of food offered)

GCN: User to Location and User to Review