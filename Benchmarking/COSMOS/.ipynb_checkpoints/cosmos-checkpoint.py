'''
This open-source software is for implementing the COSMOS algorithm. 
Paper: Cooperative Integration of Spatially Resolved Multi-Omics Data with COSMOS

Please contact our team if you have any questions:
Yuansheng Zhou (Yuansheng.Zhou@UTSouthwestern.edu)
Xue Xiao (Xiao.Xue@UTSouthwestern.edu)
Chen Tang (Chen.Tang@UTSouthwestern.edu)
Lin Xu (Lin.Xu@UTSouthwestern.edu)

Please contact Xue Xiao for programming questions about the *.py files.

Version: 10/10/2024

Please see the "LICENSE" file for the copyright information. 

NOTICE: This COSMOS software is adapted from the spaceflow code 
        (https://github.com/hongleir/SpaceFlow). 
        Please see the "LICENSE" file for copyright details of the spaceflow software.
        The implementation of the spaceflow software is described in 
        the publication "Identifying multicellular spatiotemporal organization of cells with SpaceFlow." 
        (https://www.nature.com/articles/s41467-022-31739-w).

        This COSMOS software includes functionality from pyWNN 
        (Weighted Nearest Neighbors Analysis implemented in Python), which is based on code 
        from the https://github.com/dylkot/pyWNN. 
        Please see the "LICENSE" file for copyright details of the pyWNN software.

        The DeepGraphInfomaxWNN function in the COSMOS software is adapted from the 
        torch_geometric.nn.models.deep_graph_infomax function in PyTorch Geometric (PyG),
        available at https://github.com/pyg-team/pytorch_geometric/tree/master. 
        Please see the "LICENSE" file for copyright details of the PyG software.
'''
import math
import os
import torch
import random
import gudhi
import anndata
import cmcrameri
import numpy as np
import scanpy as sc
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph
from .modulesWNN import DeepGraphInfomaxWNN
from .pyWNN import pyWNN


def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

    
def corruptionWNNit(x1, x2, edge_index,adata,w,w1,w2):
    return x1[torch.randperm(x1.size(0))], x2[torch.randperm(x2.size(0))], edge_index,adata,0,w1,w2

class Cosmos(object):
    """An object for analysis of spatial transcriptomics data.
    :param adata1 / adata2: `anndata.AnnData` object as input, see `https://anndata.readthedocs.io/en/latest/` for more info about`anndata`.
    :type adata: class:`anndata.AnnData`
    :param count_matrix1 / count_matrix2: count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
    :type count_matrix: class:`numpy.ndarray`
    :param spatial_locs: spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_cells,)
    :type spatial_locs: class:`numpy.ndarray`
    :param sample_names: list of sample names in 1D numpy str array of size (n_cells,), optional
    :type sample_names: class:`numpy.ndarray` or `list` of `str`
    :param gene_names: list of gene names in 1D numpy str array of size (n_genes,), optional
    :type gene_names: class:`numpy.ndarray` or `list` of `str`
    """

    def __init__(self, adata1=None, adata2=None, count_matrix1=None, count_matrix2=None, spatial_locs=None, sample_names=None, gene_names=None):
        """
        Inputs
        ------
        adata1 / adata2: the anndata.AnnData type object, optional (either input `adata` or both `count_matrix` and `spatial_locs`)
        count_matrix1 / count_matrix2 : count matrix of gene expression, 2D numpy array of size (n_cells, n_genes)
        spatial_locs : spatial locations of cells (or spots) match to rows of the count matrix, 1D numpy array of size (n_cells,)
        sample_names : list of sample names in 1D numpy str array of size (n_cells,), optional
        gene_names : list of gene names in 1D numpy str array of size (n_genes,), optional
        """
        if adata1 and isinstance(adata1, anndata.AnnData):
            self.adata1 = adata1
        if adata2 and isinstance(adata2, anndata.AnnData):
            self.adata2 = adata2
        elif count_matrix1 is not None and count_matrix2 is not None and spatial_locs is not None:
            self.adata1 = anndata.AnnData(count_matrix1.astype(float))
            self.adata1.obsm['spatial'] = spatial_locs.astype(float)
            self.adata2 = anndata.AnnData(count_matrix2.astype(float))
            self.adata2.obsm['spatial'] = spatial_locs.astype(float)
            if gene_names:
                self.adata1.var_names = np.array(gene_names).astype(str)
                self.adata2.var_names = np.array(gene_names).astype(str)
            if sample_names:
                self.adata1.obs_names = np.array(sample_names).astype(str)
                self.adata2.obs_names = np.array(sample_names).astype(str)
        else:
            print("Please input either an anndata.AnnData or both the count_matrix (count matrix of gene expression, 2D int numpy array of size (n_cells, n_genes)) and spatial_locs (spatial locations of cells (or spots) in 1D float numpy array of size (n_locations,)) to initiate COSMOS object.")
            exit(1)

    
    def preprocessing_data(self, do_norm = False, do_log = False, n_top_genes=None, do_pca = False, n_neighbors=10):
        """
        Preprocessing the spatial transcriptomics data
        Generates:  `self.adata_filtered`: (n_cells, n_locations) `numpy.ndarray`
                    `self.spatial_graph`: (n_cells, n_locations) `numpy.ndarray`
        :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
        :type adata: class:`anndata.annData`
        :param do_norm: whether or not perfomr normalization on the data
        :type do_norm: bool, optional, default: False
        :param do_log: whether or not perfomrm log transformation on the data
        :type do_log: bool, optional, default: False
        :param n_top_genes: the number of top highly variable genes
        :type n_top_genes: int, optional,default: None
        :param do_pca: whether or not perfomrm pca  on the data
        :type do_pca: bool, optional, default: False
        :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph
        :type n_neighbors: int, optional
        :return: a preprocessed annData object of the spatial transcriptomics data
        :rtype: class:`anndata.annData`
        :return: a geometry-aware spatial proximity graph of the spatial spots of cells
        :rtype: class:`scipy.sparse.csr_matrix`
        """
        adata1 = self.adata1
        adata2 = self.adata2
        if not adata1 or not adata2:
            print("Not enough annData objects")
            return
        if do_norm:
            sc.pp.normalize_total(adata1, target_sum=1e4)
            sc.pp.normalize_total(adata2, target_sum=1e4)
        if do_log:
            sc.pp.log1p(adata1)
            sc.pp.log1p(adata2)
        if n_top_genes:
            sc.pp.highly_variable_genes(adata1, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
            sc.pp.highly_variable_genes(adata2, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        if do_pca:
            sc.pp.pca(adata1)
            sc.pp.pca(adata2)
        spatial_locs = adata1.obsm['spatial']
        spatial_graph = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
        self.adata1_preprocessed = adata1
        self.adata2_preprocessed = adata2
        self.spatial_graph = spatial_graph

    def train(self, embedding_save_filepath="./embedding.tsv", weights_save_filepath="./weights.tsv", spatial_regularization_strength=0.05, z_dim=50, lr=1e-3, wnn_epoch  = 100, total_epoch = 1000, max_patience_bef=10, max_patience_aft=30, min_stop=100, random_seed=42, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000):
        adata1_preprocessed, adata2_preprocessed,spatial_graph = self.adata1_preprocessed, self.adata2_preprocessed, self.spatial_graph
        """
        Training the Deep GraphInfomax Model
        :param embedding_save_filepath: the default save path for the low-dimensional embeddings
        :type embedding_save_filepath: class:`str`
        :param spatial_regularization_strength: the strength for spatial regularization
        :type spatial_regularization_strength: float, optional, default: 0.05
        :param z_dim: the size of latent dimension
        :type z_dim: int, optional, default: 50
        :param lr: the learning rate for model optimization
        :type lr: float, optional, default: 1e-3
        :param wnn_epoch : the iteration number before performing WNN
        :type wnn_epoch : int, optional, default: 100
        :param total_epoch: the max epoch number 
        :type total_epoch: int, optional, default: 1000
        :param max_patience_bef: the max tolerance before doing WNN 
        :type max_patience_bef: int, optional, default: 10
        :param max_patience_aft: the tolerance epoch number without training loss decrease
        :type max_patience_aft: int, optional, default: 30
        :param min_stop: the minimum epoch number for training before any early stop
        :type min_stop: int, optional, default: 100
        :param random_seed: the random seed
        :type random_seed: int, optional, default: 42
        :param gpu: the index for gpu device that will be used for model training, if no gpu detected, cpu will be used.
        :type gpu: int, optional, default: 0
        :param regularization_acceleration: whether or not accelerate the calculation of regularization loss using edge subsetting strategy
        :type regularization_acceleration: bool, optional, default: True
        :param edge_subset_sz: the edge subset size for regularization acceleration
        :type edge_subset_sz: int, optional, default: 1000000
        :return: low dimensional embeddings for the ST data, shape: n_cells x z_dim
        :rtype: class:`numpy.ndarray`
        """
        if not adata1_preprocessed or not adata2_preprocessed:
            print("The data has not been preprocessed, please run preprocessing_data() method first!")
            return
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        
        model = DeepGraphInfomaxWNN(
        hidden_channels=z_dim, encoder=GraphEncoderWNNit(adata1_preprocessed.shape[0],adata1_preprocessed.shape[1],adata2_preprocessed.shape[1], z_dim),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruptionWNNit).to(device)
        
        expr1 = adata1_preprocessed.X.todense() if type(adata1_preprocessed.X).__module__ != np.__name__ else adata1_preprocessed.X
        expr1 = torch.tensor(expr1.copy()).float().to(device)
        
        expr2 = adata2_preprocessed.X.todense() if type(adata2_preprocessed.X).__module__ != np.__name__ else adata2_preprocessed.X
        expr2 = torch.tensor(expr2.copy()).float().to(device)
        
        edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)

        model.train()
        min_loss = np.inf
        patience_aft = 0
        patience_bef = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_params = model.state_dict()
        w1=0.5
        w2=0.5
        for epoch in range(1,total_epoch):
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            if epoch == wnn_epoch  or patience_bef > max_patience_bef: 
                # setting early stop to run wnn
                # running wnn for only one time
                z, neg_z, summary,w1,w2 = model(expr1, expr2, edge_list, adata1_preprocessed,1,0,0)
                wnn_epoch  = 0 
                min_loss = np.inf
                max_patience_bef = total_epoch
            else:
                z, neg_z, summary,w1,w2 = model(expr1, expr2, edge_list, adata1_preprocessed,0,w1,w2)           
                
                
            loss = model.loss(z, neg_z, summary)
            coords = torch.tensor(adata1_preprocessed.obsm['spatial']).float().to(device)
            if spatial_regularization_strength > 0:
                if regularization_acceleration or adata1_preprocessed.shape[0] > 5000:
                    cell_random_subset_1, cell_random_subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,)).to(
                        device), torch.randint(0, z.shape[0], (edge_subset_sz,)).to(device)
                    z1, z2 = torch.index_select(z, 0, cell_random_subset_1), torch.index_select(z, 0, cell_random_subset_2)
                    c1, c2 = torch.index_select(coords, 0, cell_random_subset_1), torch.index_select(coords, 0,
                                                                                                     cell_random_subset_1)
                    pdist = torch.nn.PairwiseDistance(p=2)

                    z_dists = pdist(z1, z2)
                    z_dists = z_dists / torch.max(z_dists)

                    sp_dists = pdist(c1, c2)
                    sp_dists = sp_dists / torch.max(sp_dists)

                    n_items = z_dists.size(dim=0)
                else:
                    z_dists = torch.cdist(z, z, p=2)
                    z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
    
                    sp_dists = torch.cdist(coords, coords, p=2)
                    sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
            
                    n_items = z.size(dim=0) * z.size(dim=0)
                penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            else: penalty_1 = 0 
            
            loss = loss + spatial_regularization_strength * penalty_1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if epoch > wnn_epoch:
                if train_loss > min_loss:
                    patience_aft += 1
                else:
                    patience_aft = 0
                    min_loss = train_loss
                    best_params = model.state_dict()
            else:
                if train_loss > min_loss:
                    patience_bef += 1
                else:
                    patience_bef = 0
                    min_loss = train_loss
                    best_params = model.state_dict()
            if epoch % 10 == 1:
                print(f"Epoch {epoch}/{total_epoch}, Loss: {str(train_loss)}")
            if patience_aft > max_patience_aft and epoch > min_stop:
                break

        model.load_state_dict(best_params)

        z, _, _,w1,w2 = model(expr1, expr2, edge_list, adata1_preprocessed,0,w1,w2)
        embedding = z.cpu().detach().numpy()
        w1 = w1.cpu().detach().numpy().reshape(-1,1)
        w2 = w2.cpu().detach().numpy().reshape(-1,1)
        ww = np.hstack((w1,w2))

        self.embedding = embedding
        self.weights = ww
        return embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphEncoderWNNit(nn.Module):
    def __init__(self,nsample, in_channels1, in_channels2, hidden_channels):
        super(GraphEncoderWNNit, self).__init__()
        self.conv1 = GCNConv(in_channels1, hidden_channels, cached=False)
        self.conv2 = GCNConv(in_channels2, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)
        self.prelu3 = nn.PReLU(hidden_channels)
        self.prelu4 = nn.PReLU(hidden_channels)
        self.mddim = hidden_channels

    def forward(self, x1, x2, edge_index, adata,w,w1,w2):

        x1 = self.conv1(x1, edge_index)
        x1 = self.prelu(x1)
        x1 = self.conv3(x1, edge_index)
        x1 = self.prelu2(x1)
        x1 = nn.functional.normalize(x1, p=2.0, dim=1)

        x2 = self.conv2(x2, edge_index)
        x2 = self.prelu3(x2)
        x2 = self.conv4(x2, edge_index)
        x2 = self.prelu4(x2)
        x2 = nn.functional.normalize(x2, p=2.0, dim=1)
        
        
        if w==1:
            pc1 = x1.detach().cpu().numpy()
            pc2 = x2.detach().cpu().numpy()
            adata.obsm['Omics1_PCA'] = pc1
            adata.obsm['Omics2_PCA'] = pc2
            WNNobj = pyWNN(adata, reps=['Omics1_PCA', 'Omics2_PCA'], npcs=[self.mddim,self.mddim], n_neighbors=20, seed=14)
            adata = WNNobj.compute_wnn(adata)
            ww=adata.obsm['Weights']
            ww = ww.astype(np.float32)
            w1 = torch.reshape(torch.from_numpy(ww[:,0]),(-1,1)).to(device)
            w2 = torch.reshape(torch.from_numpy(ww[:,1]),(-1,1)).to(device)
            
        else:
            w1 = w1
            w2 = w2
            
        x1 = x1 * w1
        x2 = x2 * w2
        x = x1 + x2
        return x,w1,w2





