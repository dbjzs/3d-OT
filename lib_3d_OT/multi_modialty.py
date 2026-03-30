import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
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
        
class SetTransformerEncoder(nn.Module):
    def __init__(self, input_dim,n_heads=4, num_transfomer_layers=3,hidden=32):
        super(SetTransformerEncoder, self).__init__()
        self.feat_conv1 = SetConv(input_dim, 2 * hidden)
        self.feat_conv2 = SetConv(2 * hidden, 4 * hidden)
        self.feat_conv3 = SetConv(4 * hidden, 8*hidden)
    def forward(self, graph):
        x = self.feat_conv1(graph.express, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)
        return x



class extractModel(nn.Module):
    def __init__(self, input_dim,hidden_dim=32, n_heads=4, n_layers=3):
        super(extractModel, self).__init__()


        self.encoder1 = SetTransformerEncoder(input_dim, n_heads, n_layers, hidden_dim)
        self.encoder2 = SetTransformerEncoder(input_dim, n_heads, n_layers, hidden_dim)
        

        self.decoder1 = nn.Sequential(
            nn.Linear(8 * hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim), 
            nn.ReLU(),
            nn.Linear(2* hidden_dim, input_dim),  
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(8 * hidden_dim, 4 * hidden_dim), 
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),  
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, input_dim),  
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(16 * hidden_dim, 8 * hidden_dim),
            nn.Linear(8 * hidden_dim, 8 * hidden_dim),
        )


        self.mse_loss = nn.MSELoss()
        

    def forward(self, graph1, graph2):

        embedding1 = self.encoder1(graph1)
        embedding2 = self.encoder2(graph2)

        fused_features = torch.cat([embedding1, embedding2], dim=-1)
        mixed_modal_features = self.fusion_mlp(fused_features)


        recon1 = self.decoder1(mixed_modal_features)
        recon2 = self.decoder2(mixed_modal_features)
        
        return recon1, recon2, mixed_modal_features  

    def compute_loss(self, graph1, graph2, recon1, recon2, mixed_modal_features):
        loss1 = self.mse_loss(recon1, graph1.express)
        loss2 = self.mse_loss(recon2, graph2.express)
        total_loss = loss1 + loss2
        return loss1, loss2, total_loss


def train_graph_extractor(graph1,graph2, model,optimizer, device, epochs=300):
    best_model = None
    min_loss = float('inf')
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        total_loss = 0.0
        
        recon1, recon2, mixed_modal_features = model(graph1, graph2)
        
        loss1,loss2,loss = model.compute_loss(graph1, graph2,recon1, recon2, mixed_modal_features)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}", end="\r")
        
        if running_loss < min_loss:
            min_loss = running_loss
            best_model = model.state_dict()
            
    return best_model



class UnifiedModel(nn.Module):
    def __init__(self, input_dim,simk,otk,reconk,hidden_dim,best_encoder1=None,best_encoder2=None):
        super(UnifiedModel, self).__init__()
        
        self.encoder1 = extractModel(input_dim=input_dim,hidden_dim=hidden_dim)
        self.encoder2 = extractModel(input_dim=input_dim,hidden_dim=hidden_dim)
        self.encoder1.load_state_dict(best_encoder1)
        self.encoder2.load_state_dict(best_encoder2)
        self.log_gamma = torch.nn.Parameter(torch.zeros(1))
        self.log_epsilon = torch.nn.Parameter(torch.zeros(1))

        self.simk=simk
        self.otk=otk
        self.neighbors=reconk
        self.mse_loss = nn.MSELoss()

        
    
    def get_recon_flow(self,pclouds, feats):
        feats_0, feats_1 = feats[0], feats[1]
        transport_cross, similarity_cross = ot.sinkhorn(
            feats_0,
            feats_1,
            pclouds[0],
            pclouds[1],
            epsilon=torch.exp(self.log_epsilon),
            gamma=torch.exp(self.log_gamma),
            max_iter=50,
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

        else:
            row_sum = transport_cross.sum(-1, keepdim=True)
            target_cross_recon = (transport_cross @ pclouds[1]) / (row_sum + 1e-8)
            corr_conf = None

        recon_flow = target_cross_recon - pclouds[0]

        return recon_flow, corr_conf, target_cross_recon, similarity_cross

    def get_encoder_results(self,model1,model2,pcloud_list1,pcloud_list2):
        with torch.no_grad():
            model1.eval()
            model2.eval()
            recon1, recon2, mixed_modal_features1 = model1(pcloud_list1[0], pcloud_list1[1])
            recon3, recon4, mixed_modal_features2 = model2(pcloud_list2[0], pcloud_list2[1])
            epsilon = 1e-5
            min_val = mixed_modal_features1.min()
            max_val = mixed_modal_features1.max()
            normalized_features1 = (mixed_modal_features1 - min_val) / (max_val - min_val + epsilon)
            min_val = mixed_modal_features2.min()
            max_val = mixed_modal_features2.max()
            normalized_features2 = (mixed_modal_features2 - min_val) / (max_val - min_val + epsilon)
        return normalized_features1,normalized_features2,pcloud_list1[0],pcloud_list2[0]

    

    def min_max_normalize(self, spatial_coords):
        flattened_coords = spatial_coords.view(-1, 2)
        min_vals = flattened_coords.min(dim=0, keepdim=True)[0]  
        max_vals = flattened_coords.max(dim=0, keepdim=True)[0]  
        min_vals = min_vals.view(1, 1, 2)
        max_vals = max_vals.view(1, 1, 2)
        normalized_coords = (spatial_coords - min_vals) / (max_vals - min_vals)
        return normalized_coords
    

    def forward(self,graph1,graph2):
        pcloud_list1=graph1
        pcloud_list2=graph2
        features1,features2,graph1,graph2 = self.get_encoder_results(self.encoder1,self.encoder2,pcloud_list1,pcloud_list2)
        
        pcloud1 = pcloud_list1[0].pcloud
        pcloud2 = pcloud_list2[0].pcloud
        pcloud1=self.min_max_normalize(pcloud1)
        pcloud2=self.min_max_normalize(pcloud2)

        
        pclouds=[pcloud1, pcloud2]
        recon_flow, corr_conf, target_cross_recon, similarity_cross = self.get_recon_flow(
            pclouds=pclouds,
            feats=[features1, features2]
        )
     
        return recon_flow, corr_conf, target_cross_recon, graph1,pclouds





def train(model, pcloud_list, optimizer, scheduler, device, nb_epochs=1, use_smooth_flow=True, use_div_flow=True, args=None):

    model = model.to(device)

    for epoch in range(nb_epochs): 
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for idx in range(len(pcloud_list) - 1): 

            graph1, graph2 = pcloud_list[idx], pcloud_list[idx + 1]

            recon_flow, corr_conf, target_recon, graph,pclouds = model(graph1,graph2)

            loss, target_recon_loss, smooth_flow_loss, div_flow_loss = compute_loss_unsupervised(
                recon_flow, corr_conf, target_recon, graph, pclouds, args,device
            )
            loss = loss / (len(pcloud_list) - 1)
            loss.backward()
            total_loss += loss.item() 

     
        optimizer.step()
        scheduler.step() 

        print(f"\rTime Pair {idx},total_loss: {total_loss:.4f},smooth_flow_loss: {smooth_flow_loss.item():.4f} "
                  f"Target Recon Loss: {target_recon_loss.item():.4f}, Div Flow Loss: {div_flow_loss.item():.4f}",end="")