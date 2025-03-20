def plot_selected_cell_type_flow(
    pcloud_list, model, device, finaltruth,
    selected_cell_type, arrow_sample_size_per_type=55, 
    xlim=(-0.2, 1.5), ylim=(-0.5, 2), height_scale=1.0,save_path=None
):
    """绘制特定细胞类型的流动"""
    # 创建画布
    fig = plt.figure(figsize=(19.5,9.5))  # 单个子图的宽度
    
    # 创建子图2：显示特定细胞类型的流动
    ax = fig.add_subplot(111, projection="3d")
    all_arrow_ends, layer_1_pcloud_3D=_plot_flows(
        ax=ax,
        pcloud_list=pcloud_list,
        model=model,
        device=device,
        finaltruth=finaltruth,
        arrow_sample_size_per_type={selected_cell_type: 200000},
        selected_cell_type=selected_cell_type,
        xlim=xlim, ylim=ylim,
        height_scale=height_scale,
        title=f"Cell Type {selected_cell_type}"
    )
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Image saved as: {save_path}")
    return all_arrow_ends, layer_1_pcloud_3D 
    
def _plot_flows(
    ax, pcloud_list, model, device,finaltruth,
    arrow_sample_size_per_type, selected_cell_type,
    xlim, ylim, height_scale, title,
    show_start_points=True,  # 新增控制参数
    show_arrow_lines=True     # 新增控制参数
):
    """通用绘图逻辑（新增箭头显示控制）"""
    model.eval()
    
    # 初始化数据容器
    all_pclouds = []
    all_colors = []
    all_arrow_starts = []
    all_arrow_ends = []

    # 创建颜色映射
    all_celltypes = sorted(list({celltype for data in pcloud_list for celltype in data.truth}))
    color_map = {celltype: cm.tab20(i / len(all_celltypes)) for i, celltype in enumerate(all_celltypes)}
    
    # 处理每个时间步
    for idx in range(len(pcloud_list) - 1):
        data1, data2 = pcloud_list[idx], pcloud_list[idx + 1]
        
        # 模型预测
        with torch.no_grad():
            recon_flow, _, _, _, pclouds = model(data1, data2)
        
        # 数据转换
        pcloud_np = pclouds[0].squeeze(0).cpu().numpy()
        flow_np = recon_flow.squeeze(0).cpu().numpy()
        z_coord = np.full((pcloud_np.shape[0], 1), idx * height_scale)
        pcloud_3D = np.hstack((pcloud_np, z_coord))
        flow_3D = np.hstack((flow_np, np.full((flow_np.shape[0], 1), height_scale)))
        
        # 颜色处理
        cell_types = data1.truth
        colors = np.array([color_map[ct] for ct in cell_types])
        all_colors.append(colors)
        all_pclouds.append(pcloud_3D)

        # 箭头采样逻辑
        if isinstance(arrow_sample_size_per_type, int):
            sample_size_dict = {ct: arrow_sample_size_per_type for ct in all_celltypes}
        else:
            sample_size_dict = arrow_sample_size_per_type
            
        for cell_type, sample_size in sample_size_dict.items():
            if selected_cell_type is not None and cell_type != selected_cell_type:
                continue
                
            mask = cell_types == cell_type
            selected_indices = np.where(mask)[0]
            if len(selected_indices) == 0:
                continue
                
            # 自动处理全选逻辑
            sample_indices = np.random.choice(
                selected_indices, 
                size=min(sample_size, len(selected_indices)), 
                replace=False
            )
            
            # 记录箭头坐标
            starts = pcloud_3D[sample_indices]
            ends = starts + flow_3D[sample_indices]
            all_arrow_starts.append(starts)
            all_arrow_ends.append(ends)

    # 添加最后一个时间步
    final_pcloud = pclouds[1].squeeze(0).cpu().numpy()
    final_z = (len(pcloud_list)-1)*height_scale
    final_pcloud_3D = np.hstack((final_pcloud, np.full((final_pcloud.shape[0],1), final_z)))
    final_colors = np.array([color_map[ct] for ct in pcloud_list[-1].truth])
    final_truth = pcloud_list[-1].truth  # 获取最后一个时间步的细胞类型标签
    #layer_1_mask = (final_truth == finaltruth)
    #layer_1_mask = final_truth == selected_cell_type
    layer_1_mask = (final_truth == finaltruth)|(final_truth == '12_2')
    
# Use the mask to filter the points in the 3D point cloud
    layer_1_pcloud_3D = final_pcloud_3D[layer_1_mask]
# 获取细胞类型为 'Layer_1' 的点
    # layer_1_mask = final_truth == selected_cell_type  # 筛选出细胞类型为 'Layer_1' 的点的布尔索引
    # layer_1_pcloud_3D = final_pcloud_3D[layer_1_mask] 

    #print(f"Layer_1 Points count: {layer_1_pcloud_3D.shape[1]}")
    all_pclouds.append(final_pcloud_3D)
    all_colors.append(final_colors)

    # 合并数据
    all_pclouds = np.concatenate(all_pclouds)
    all_colors = np.concatenate(all_colors)
    if len(all_arrow_starts) > 0:
        all_arrow_starts = np.concatenate(all_arrow_starts)
        all_arrow_ends = np.concatenate(all_arrow_ends)
        #print(f"Layer_1 Points count: {all_arrow_ends.shape[0]}")
    
    # 绘制散点
    ax.scatter(
        all_pclouds[:,0], all_pclouds[:,1], all_pclouds[:,2],
        c=all_colors, s=1, alpha=0.4
    )
    
    # 绘制箭头元素
    if len(all_arrow_starts) > 0:
        # 绘制起始点（条件控制）
        if show_start_points:
            ax.scatter(
                all_arrow_starts[:,0], all_arrow_starts[:,1], all_arrow_starts[:,2],
                c="red", s=2, label="Start Points", alpha=0.6
            )
        
        # 始终绘制终点（保持颜色一致性）
        ax.scatter(
            all_arrow_ends[:,0], all_arrow_ends[:,1], all_arrow_ends[:,2],
            c="blue", s=5, label="End Points", alpha=1
        )
        
        # 绘制箭头线（条件控制）
        if show_arrow_lines:
            for start, end in zip(all_arrow_starts, all_arrow_ends):
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color="grey", linewidth=0.6, alpha=1,linestyle="--"
                )

    # 设置视角与坐标轴
    ax.view_init(elev=30, azim=-15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, len(pcloud_list) * height_scale)
    #ax.set_title(title)
    ax.set_axis_off()
    # Before returning, print the lengths to ensure they contain data
    print(f"Number of arrow ends: {len(all_arrow_ends)}")
    print(f"Layer 1 points count: {len(layer_1_pcloud_3D)}")

    return all_arrow_ends,layer_1_pcloud_3D



import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

def plot_all_pairs_cell_type_flow(
    graphs, aligned_models, device, finaltruth,
    selected_cell_type, arrow_sample_size_per_type=55,
    xlim=(-0.2, 1.5), ylim=(-0.5, 2), height_scale=1.0, save_path=None
):
    """绘制所有配对中特定细胞类型的流动"""
    # 创建画布
    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111, projection="3d")

    # 调用底层绘图函数，处理所有配对
    all_arrow_ends= _plot_all_pairs_flows(
        ax=ax,
        graphs=graphs,
        aligned_models=aligned_models,
        device=device,
        finaltruth=finaltruth,
        arrow_sample_size_per_type={selected_cell_type: 200000},
        selected_cell_type=selected_cell_type,
        xlim=xlim, ylim=ylim,
        height_scale=height_scale,
        title=f"Cell Type {selected_cell_type} Across All Pairs"
    )

    # 保存图像（如果指定路径）
    if save_path:
        fig.savefig(save_path, dpi=500)
        print(f"图像保存为: {save_path}")

    return all_arrow_ends

def _plot_all_pairs_flows(
    ax, graphs, aligned_models, device, finaltruth,
    arrow_sample_size_per_type, selected_cell_type,
    xlim, ylim, height_scale, title,
    show_start_points=True, show_arrow_lines=True
):
    """绘制所有配对的流动（支持 n 张切片和 n-1 个配对）"""
    # 初始化数据容器
    all_pclouds = []
    all_colors = []
    all_arrow_starts = []
    all_arrow_ends = []

    # 创建颜色映射
    all_celltypes = sorted(list({celltype for data in graphs for celltype in data.truth}))
    color_map = {celltype: cm.tab20(i / len(all_celltypes)) for i, celltype in enumerate(all_celltypes)}

    for pair_idx, model in enumerate(aligned_models):
        # 配对索引（逆向对齐，例如 graph1 -> graph0, graph3 -> graph2）
        idx = 2 * pair_idx + 1  # 从 graph1 开始
        if idx >= len(graphs):
            break
        
        data2, data1 = graphs[idx], graphs[idx -1]  # 当前配对
        print(f"处理配对: graph{idx} -> graph{idx-1} (配对 {pair_idx})")

        # 模型预测
        model.eval()
        with torch.no_grad():
            recon_flow, _, _, _, pclouds = model(data1.to(device), data2.to(device))
        
        # 数据转换
        pcloud_np = pclouds[0].squeeze(0).cpu().numpy()  # 起始图（graph[idx]）
        flow_np = recon_flow.squeeze(0).cpu().numpy()     # 流动
        z_coord = np.full((pcloud_np.shape[0], 1), pair_idx * height_scale)  # 当前时间步的高度
        pcloud_3D = np.hstack((pcloud_np, z_coord))
        flow_3D = np.hstack((flow_np, np.full((flow_np.shape[0], 1), height_scale)))

        # 颜色处理
        cell_types = data1.truth
        colors = np.array([color_map[ct] for ct in cell_types])
        all_pclouds.append(pcloud_3D)
        all_colors.append(colors)

        # 箭头采样逻辑
        sample_size_dict = (
            {ct: arrow_sample_size_per_type for ct in all_celltypes}
            if isinstance(arrow_sample_size_per_type, int)
            else arrow_sample_size_per_type
        )

        for cell_type, sample_size in sample_size_dict.items():
            if selected_cell_type is not None and cell_type != selected_cell_type:
                continue
                
            mask = cell_types == cell_type
            selected_indices = np.where(mask)[0]
            if len(selected_indices) == 0:
                continue

            sample_indices = np.random.choice(
                selected_indices,
                size=min(sample_size, len(selected_indices)),
                replace=False
            )

            # 记录箭头坐标
            starts = pcloud_3D[sample_indices]
            ends = starts + flow_3D[sample_indices]
            all_arrow_starts.append(starts)
            all_arrow_ends.append(ends)

    # 添加最后一个时间步（最终切片）
    final_pcloud = pclouds[1].squeeze(0).cpu().numpy()
    final_z = (len(graphs)/2) * height_scale
    final_pcloud_3D = np.hstack((final_pcloud, np.full((final_pcloud.shape[0], 1), final_z)))
    final_colors = np.array([color_map[ct] for ct in graphs[-1].truth])
    final_truth = graphs[-1].truth
    
    # 为最终切片创建掩码
    selected_mask = (final_truth == selected_cell_type)
    other_mask = ~selected_mask

    # 添加最终切片到总集合
    all_pclouds.append(final_pcloud_3D)
    all_colors.append(final_colors)
    
    # 合并所有数据
    all_pclouds = np.concatenate(all_pclouds)
    all_colors = np.concatenate(all_colors)
    if all_arrow_starts:
        all_arrow_starts = np.concatenate(all_arrow_starts)
        all_arrow_ends = np.concatenate(all_arrow_ends)

    # 绘制散点
    ax.scatter(
        all_pclouds[:, 0], all_pclouds[:, 1], all_pclouds[:, 2],
        c=all_colors, s=1, alpha=0.2
    )
    ax.scatter(
        final_pcloud_3D[selected_mask, 0],
        final_pcloud_3D[selected_mask, 1],
        final_pcloud_3D[selected_mask, 2],
        c='red',
        s=2,
        alpha=0.5,
        label=f'Selected {selected_cell_type}'
    )

    # 绘制箭头元素
    if all_arrow_starts.size > 0:
        if show_start_points:
            ax.scatter(
                all_arrow_starts[:, 0], all_arrow_starts[:, 1], all_arrow_starts[:, 2],
                c="red", s=2, label="Start Points", alpha=0.6
            )
        
        ax.scatter(
            all_arrow_ends[:, 0], all_arrow_ends[:, 1], all_arrow_ends[:, 2],
            c="blue", s=4, label="End Points", alpha=1
        )
        
        if show_arrow_lines:
            for start, end in zip(all_arrow_starts, all_arrow_ends):
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color="grey", linewidth=0.6, alpha=1, linestyle="--"
                )

    # 设置视角与坐标轴
    ax.view_init(elev=10, azim=-15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, (len(graphs)/2) * height_scale)
    ax.set_axis_off()

    return all_arrow_ends