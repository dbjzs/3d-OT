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
# -*- coding: utf-8 -*-
import torch
import numpy as np

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index
