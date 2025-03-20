def pairwise_dpca_and_train(adatalist: List, join: str = 'inner', n_comps: int = 50, epochs: int = 1150,neighbors=8, device=None) -> tuple:
    corrected_adatas = []  # 存储所有校正后的 adata
    graphs = []  # 存储所有训练的 graph
    best_models = []  # 存储所有训练的模型
    import time
    
    # 处理所有配对
    for i in range(len(adatalist) - 1):
        print(f"Processing pair: {i} -> {i+1}")
        # 每次都使用原始的 adata 进行 DPCA
        current_pair = [adatalist[i], adatalist[i + 1]]
        corrected_adata1, corrected_adata2 = dpca(current_pair, join=join, n_comps=n_comps)
        
        # 存储校正后的 adata（每对都存两个）
        corrected_adatas.append(corrected_adata1)  # 每对的 corrected_adata1
        corrected_adatas.append(corrected_adata2)  # 每对的 corrected_adata2
        
        # 训练 corrected_adata1 的模型
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
        
        # 训练 corrected_adata2 的模型
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

import os
import torch
from torch import optim
import numpy as np

# 假设这些函数和类已定义
# from your_module import UnifiedModel, train, plot_selected_cell_type_flow, prepare_data, extractMODEL, train_graph_extractor

# 输出目录（仅用于绘图）
output_flow_dir = "/home/dbj/mouse/flow_plots"
os.makedirs(output_flow_dir, exist_ok=True)

def pairwise_align_reverse(pclouds_list, best_models, device, nb_epochs=1,simk=5,otk=200):
    aligned_models = []  # 存储所有训练好的 UnifiedModel
    
    # 每个配对内对齐，共5次
    for i in range(1, len(pclouds_list), 2):  # 从 1 到 9，步长 2，共5次
        print(f"Aligning pair: graph{i} -> graph{i-1} (Pair {i//2})")
        set_seed(7)
        # 获取配对内的输入维度和最佳编码器
        input_dim1 = pclouds_list[i].express.shape[-1]      # 第二个图 (graph1, graph3, ...)
        input_dim2 = pclouds_list[i - 1].express.shape[-1]  # 第一个图 (graph0, graph2, ...)
        best_encoder1 = best_models[i]                      # 第二个模型
        best_encoder2 = best_models[i - 1]                  # 第一个模型
        
        # 初始化 UnifiedModel
        model = UnifiedModel(
            input_dim1=input_dim1,
            input_dim2=input_dim2,
            simk=simk,
            otk=otk,
            reconk=1,
            best_encoder1=best_encoder1,
            best_encoder2=best_encoder2
        ).to(device)
        
        # 设置优化器和学习率调度器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        lr_lambda = lambda epoch: 1.0 if epoch < 340 else 1.0  # 可调整
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # 训练参数
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
        
        # 当前配对内的 pcloud_list
        current_pair = [pclouds_list[i], pclouds_list[i - 1]]  # 对齐 graph{i} -> graph{i-1}
        
        # 训练 UnifiedModel
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
        
        # 添加到列表（不保存到本地）
        aligned_models.append(model)
    
    return aligned_models