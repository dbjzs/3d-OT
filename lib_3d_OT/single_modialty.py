import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import time
import torch.nn as nn
from lib_3d_OT.ottools.utils import log_string
from lib_3d_OT.ottools import ot, reconstruction as R
from lib_3d_OT.ottools.losses import compute_loss_unsupervised,chamfer_loss
from lib_3d_OT.utils import set_seed
set_seed(7)
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



class UnifiedModel(nn.Module):
    def __init__(self, input_dim1,input_dim2,simk,otk,reconk,best_encoder1=None,best_encoder2=None):
        super(UnifiedModel, self).__init__()
        self.encoder1 = extractMODEL(args=None, input_dim=input_dim1)
        self.encoder2 = extractMODEL(args=None, input_dim=input_dim2)
        self.encoder1.load_state_dict(best_encoder1)
        self.encoder2.load_state_dict(best_encoder2)

        self.log_gamma = torch.nn.Parameter(torch.zeros(1))
        self.log_epsilon = torch.nn.Parameter(torch.zeros(1))

        self.simk=simk
        self.otk=otk
        self.neighbors=reconk

    
    def get_recon_flow(self,pclouds, feats):
        set_seed(7)
        feats_0, feats_1 = feats[0], feats[1]
        transport_cross, similarity_cross = ot.sinkhorn(
            feats_0,
            feats_1,
            pclouds[0],
            pclouds[1],
            epsilon=torch.exp(self.log_epsilon),
            gamma=torch.exp(self.log_gamma),
            max_iter=100,
            sim_k=self.simk,
            dist_k=self.otk,
        )
        

        if self.neighbors > 0:
            source_cross_nn_weight, _, source_cross_nn_idx, _, _, _ = \
                R.get_s_t_neighbors(self.neighbors, transport_cross, sim_normalization="none", s_only=True)

            # Target point cloud cross reconstruction
            cross_weight_sum = source_cross_nn_weight.sum(-1, keepdim=True)
            source_cross_nn_weight_normalized = source_cross_nn_weight / (cross_weight_sum + 1e-8)
            target_cross_recon = R.reconstruct(pclouds[1], source_cross_nn_idx, source_cross_nn_weight_normalized, self.neighbors)

            # Matching probability
            cross_nn_sim, _, _, _ = R.get_s_t_topk(similarity_cross, self.neighbors, s_only=True, nn_idx=source_cross_nn_idx)
            nn_sim_weighted = cross_nn_sim * source_cross_nn_weight_normalized
            nn_sim_weighted = torch.sum(nn_sim_weighted, dim=2)
            corr_conf = (nn_sim_weighted + 1) / 2
            # else:
            #     corr_conf = torch.clamp_min(nn_sim_weighted, 0.0)
        else:
            row_sum = transport_cross.sum(-1, keepdim=True)
            target_cross_recon = (transport_cross @ pclouds[1]) / (row_sum + 1e-8)
            corr_conf = None

        # Estimate flow from target cross reconstruction
        recon_flow = target_cross_recon - pclouds[0]

        return recon_flow, corr_conf, target_cross_recon, similarity_cross

    def get_encoder_results(self,model1,model2,graph1,graph2):
        with torch.no_grad():
            model1.eval()
            model2.eval()
            features1= model1.get_features(graph1)
            features2= model2.get_features(graph2)
            
            features1=model1.decode(features1)
            features2=model2.decode(features2)
        return features1,features2,graph1,graph2


    def min_max_normalize(self, spatial_coords):

        flattened_coords = spatial_coords.view(-1, 2) 
        min_vals = flattened_coords.min(dim=0, keepdim=True)[0]  # (1, 2)
        max_vals = flattened_coords.max(dim=0, keepdim=True)[0]  # (1, 2)
        min_vals = min_vals.view(1, 1, 2)
        max_vals = max_vals.view(1, 1, 2)
        normalized_coords = (spatial_coords - min_vals) / (max_vals - min_vals)
        return normalized_coords
    

        
        
    def forward(self,graph1,graph2):
        features1,features2,graph1,graph2 = self.get_encoder_results(self.encoder1,self.encoder2,graph1,graph2)
        
        pcloud1 = graph1.pcloud
        pcloud2 = graph2.pcloud
        
        pcloud1=self.min_max_normalize(pcloud1)
        pcloud2=self.min_max_normalize(pcloud2)

        pclouds=[pcloud1, pcloud2]
        
        recon_flow, corr_conf, target_cross_recon, similarity_cross = self.get_recon_flow(
            pclouds=pclouds,
            feats=[features1, features2]
        )
     

        return recon_flow, corr_conf, target_cross_recon, graph1,pclouds





def train(model, pcloud_list, optimizer, scheduler, device, nb_epochs=1, use_corr_conf=True, use_smooth_flow=True, use_div_flow=True, args=None):
    """
    Unified training function for GNN and flow reconstruction over all time points together.
    """
    model = model.to(device)
    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for idx in range(len(pcloud_list) - 1):

            graph1, graph2 = pcloud_list[idx], pcloud_list[idx + 1]

            recon_flow, corr_conf, target_recon, graph,pclouds = model(graph1, graph2)

            loss, target_recon_loss,smooth_flow_loss, div_flow_loss = compute_loss_unsupervised(
                recon_flow, corr_conf, target_recon, graph, pclouds, args,device
            )
            loss = loss / (len(pcloud_list) - 1)
            loss.backward()
            total_loss += loss.item()



        optimizer.step()
        scheduler.step()

        print(f"\rTime Pair {idx},total_loss: {total_loss:.4f},smooth_flow_loss: {smooth_flow_loss.item():.4f} "
                  f"Target Recon Loss: {target_recon_loss.item():.8f},Div Flow Loss: {div_flow_loss.item():.4f}",end="")




