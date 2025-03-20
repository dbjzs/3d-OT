class SetConv(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        super(SetConv, self).__init__()

        self.fc1 = torch.nn.Conv2d(nb_feat_in+2, nb_feat_out, 1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc2 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn2 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn3 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self, signal,graph):
        
        # Input features dimension
        b, n, c = signal.shape
        n_out = graph.size[0] // b


        
        # Concatenate input features with edge features
        signal = signal.reshape(b * n, c)
        signal = torch.cat((signal[graph.edges], graph.edge_feats), -1)
        signal = signal.view(b, n_out, graph.k_neighbors, c + 2)
        signal = signal.transpose(1, -1)

        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.bn1,
            self.lrelu,
            self.fc2,
            self.bn2,
            self.lrelu,
            self.fc3,
            self.bn3,
            self.lrelu,
            self.pool,
        ]:
            signal = func(signal)

        return signal.transpose(1, -1)


class MODEL(torch.nn.Module):
    def __init__(self, args,input_dim):
        super(MODEL, self).__init__()
        self.input_dim=input_dim
        # Hand-chosen parameters. Define the number of channels.
        n = 32
        # Feature extraction
        self.feat_conv1 = SetConv(self.input_dim, 8*n)
        self.feat_conv2 = SetConv(8*n, 4 * n)
        self.feat_conv3 = SetConv(4* n, 2* n)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2*n, nhead=4),
            num_layers=3
        )
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(2 * n,4 * n),  # Reduce dimensionality
                torch.nn.ReLU(),
                torch.nn.Linear(4 * n,8*n),  # Further reduce
                torch.nn.ReLU(),
                torch.nn.Linear(8*n, self.input_dim),# Output 3000-dimensional features
            )

    def get_features(self,graph):
        x = self.feat_conv1(graph.express,graph)
        x = self.feat_conv2(x,graph)
        x = self.feat_conv3(x,graph)
        x = self.transformer(x)
        return x


    def decode(self, x):
        B, N, _ = x.shape
        x = x.view(-1, x.size(-1))
        decoded_features = self.decoder(x)
        decoded_features = decoded_features.view(B, N, self.input_dim)
        return decoded_features
import time

def train_graph_extractor(pcloud_list, model, optimizer, device, epochs=300):
    best_model = None
    min_loss = float('inf')
    model.to(device)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        total_loss = 0.0

        for graph in pcloud_list:
            input_dim = graph.express.shape[-1]
            #model = MODEL(args=None, input_dim=input_dim)
            
            z= model.get_features(graph)
            decoded_features = model.decode(z)
            loss = F.mse_loss(decoded_features, graph.express)
            
            total_loss += loss
            
        total_loss.backward()
        optimizer.step()
        running_loss = total_loss.item()

        print(f"\rEpoch {epoch + 1}/{epochs}, Loss: {running_loss:.6f}, Min Loss: {min_loss:.6f}", end="")
        
        if running_loss < min_loss:
            min_loss = running_loss
            best_model = model.state_dict()

    end_time = time.time()  # End time recording
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"\nTraining finished in {elapsed_time:.2f} seconds.")
    return best_model, min_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

import torch.nn as nn

class SetConv(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        super(SetConv, self).__init__()

        self.fc1 = torch.nn.Conv2d(nb_feat_in+2, nb_feat_out, 1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc2 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn2 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn3 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)
        
    def forward(self, signal,graph):
        
        # Input features dimension
        b, n, c = signal.shape
        n_out = graph.size[0] // b


        
        # Concatenate input features with edge features
        signal = signal.reshape(b * n, c)
        signal = torch.cat((signal[graph.edges], graph.edge_feats), -1)
        signal = signal.view(b, n_out, graph.k_neighbors, c + 2)
        signal = signal.transpose(1, -1)

        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.bn1,
            self.lrelu,
            self.fc2,
            self.bn2,
            self.lrelu,
            self.fc3,
            self.bn3,
            self.lrelu,
            self.pool,
        ]:
            signal = func(signal)

        return signal.transpose(1, -1)


class extractMODEL(torch.nn.Module):
    def __init__(self, args,input_dim):
        super(extractMODEL, self).__init__()
        self.input_dim=input_dim
        # Hand-chosen parameters. Define the number of channels.
        n = 32
        # Feature extraction
        self.feat_conv1 = SetConv(self.input_dim, 8*n)
        self.feat_conv2 = SetConv(8*n, 4 * n)
        self.feat_conv3 = SetConv(4* n, 2* n)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2*n, nhead=4),
            num_layers=3
        )
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(2 * n,4 * n),  # Reduce dimensionality
                torch.nn.ReLU(),
                torch.nn.Linear(4 * n,8*n),  # Further reduce
                torch.nn.ReLU(),
                torch.nn.Linear(8*n, self.input_dim),# Output 3000-dimensional features
            )
    

    def get_features(self,graph):
        x = self.feat_conv1(graph.express,graph)
        x = self.feat_conv2(x,graph)
        x = self.feat_conv3(x,graph)
        x = self.transformer(x)
        return x


    def decode(self, x):
        B, N, _ = x.shape
        x = x.view(-1, x.size(-1))
        decoded_features = self.decoder(x)
        decoded_features = decoded_features.view(B, N, self.input_dim)
        return decoded_features


def train_graph_extractor(graph,model,optimizer, device, epochs=300):
    best_model = None
    min_loss = float('inf')
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        total_loss = 0.0
        
        z= model.get_features(graph)
        decoded_features = model.decode(z)
        loss = F.mse_loss(decoded_features, graph.express)
        total_loss = loss
            
        total_loss.backward()
        optimizer.step()
        running_loss = total_loss.item()

        print(f"\rEpoch {epoch + 1}/{epochs}, Loss: {running_loss:.6f}, Min Loss: {min_loss:.6f}", end="")
        
        if running_loss < min_loss:
            min_loss = running_loss
            best_model = model.state_dict()
            
    return best_model,min_loss






