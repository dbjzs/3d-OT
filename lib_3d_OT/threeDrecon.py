from typing import List
import os
import torch
from torch import optim
import numpy as np
from lib_3d_OT.single_modialty import *
from lib_3d_OT.utils import *
def pairwise_dpca_and_train(adatalist: List, join: str = 'inner', n_comps: int = 50, epochs: int = 1150,neighbors=8, device=None) -> tuple:
    corrected_adatas = []  
    graphs = []  
    best_models = [] 
    import time
    
    for i in range(len(adatalist) - 1):
        print(f"Processing pair: {i} -> {i+1}")

        current_pair = [adatalist[i], adatalist[i + 1]]
        corrected_adata1, corrected_adata2 = dpca(current_pair, join=join, n_comps=n_comps)
        

        corrected_adatas.append(corrected_adata1)
        corrected_adatas.append(corrected_adata2) 

        set_seed(7)
        start_time = time.time()
        graph1 = prepare_data(corrected_adata1, location="spatial", nb_neighbors=neighbors)
        graph1 = graph1.to(device)
        input_dim1 = graph1.express.shape[-1]
        model1 = extractMODEL(args=None, input_dim=input_dim1)
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
        best_model1, min_loss1 = train_graph_extractor(graph1, model1, optimizer1, device, epochs=epochs)
        training_time = time.time() - start_time
        print(f"Training time for adata{i} in pair {i}->{i+1}: {training_time:.2f} seconds")
        graphs.append(graph1)
        best_models.append(best_model1)
        
        set_seed(7)
        start_time = time.time()
        graph2 = prepare_data(corrected_adata2, location="spatial", nb_neighbors=8)
        graph2=graph2.to(device)
        input_dim2 = graph2.express.shape[-1]
        model2 = extractMODEL(args=None, input_dim=input_dim2)
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        best_model2, min_loss2 = train_graph_extractor(graph2, model2, optimizer2, device, epochs=epochs)
        training_time = time.time() - start_time
        print(f"Training time for adata{i+1} in pair {i}->{i+1}: {training_time:.2f} seconds")
        graphs.append(graph2)
        best_models.append(best_model2)
    
    return corrected_adatas, graphs, best_models





def pairwise_align_reverse(pclouds_list, best_models, device, nb_epochs=1,simk=5,otk=200):
    aligned_models = []  
    
    for i in range(1, len(pclouds_list), 2):  
        print(f"Aligning pair: graph{i} -> graph{i-1} (Pair {i//2})")
        set_seed(7)
        input_dim1 = pclouds_list[i].express.shape[-1]      
        input_dim2 = pclouds_list[i - 1].express.shape[-1]  
        best_encoder1 = best_models[i]                    
        best_encoder2 = best_models[i - 1]              
        
        model = UnifiedModel(
            input_dim1=input_dim1,
            input_dim2=input_dim2,
            simk=simk,
            otk=otk,
            reconk=1,
            best_encoder1=best_encoder1,
            best_encoder2=best_encoder2
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        lr_lambda = lambda epoch: 1.0 if epoch < 340 else 1.0 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        args = {
            "backward_dist_weight": 1.0,
            "use_corr_conf": 0,
            "corr_conf_loss_weight": 1.0,
            "use_smooth_flow": 1,
            "smooth_flow_loss_weight": 1.0,
            "use_div_flow": 1,
            "div_flow_loss_weight": 1.0,
            "div_neighbor": 8,
            "lattice_steps": 10,
            "nb_neigh_smooth_flow": 32,
        }
        
        current_pair = [pclouds_list[i], pclouds_list[i - 1]]
        
        train(
            model=model,
            pcloud_list=current_pair,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            nb_epochs=nb_epochs,
            use_corr_conf=False,
            use_smooth_flow=True,
            use_div_flow=True,
            args=args
        )
        
        aligned_models.append(model)
    
    return aligned_models