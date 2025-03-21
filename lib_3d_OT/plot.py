import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

def plot_selected_cell_type_flow(
    pcloud_list, model, device, finaltruth,
    selected_cell_type, arrow_sample_size_per_type=55, 
    xlim=(-0.2, 1.5), ylim=(-0.5, 2), height_scale=1.0,size=1,alpha=1,save_path=None
):

    fig = plt.figure(figsize=(18.5,8.5))
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
        size=size,
        alpha=alpha,
        title=f"Cell Type {selected_cell_type}"
    )
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Image saved as: {save_path}")
    return all_arrow_ends, layer_1_pcloud_3D 
    
def _plot_flows(
    ax, pcloud_list, model, device,finaltruth,
    arrow_sample_size_per_type, selected_cell_type,
    xlim, ylim, height_scale,size,alpha, title,
    show_start_points=True, 
    show_arrow_lines=True  
):
    model.eval()
    

    all_pclouds = []
    all_colors = []
    all_arrow_starts = []
    all_arrow_ends = []

    all_celltypes = sorted(list({celltype for data in pcloud_list for celltype in data.truth}))
    color_map = {celltype: cm.tab20(i / len(all_celltypes)) for i, celltype in enumerate(all_celltypes)}

    for idx in range(len(pcloud_list) - 1):
        data1, data2 = pcloud_list[idx], pcloud_list[idx + 1]

        with torch.no_grad():
            recon_flow, _, _, _, pclouds = model(data1, data2)

        pcloud_np = pclouds[0].squeeze(0).cpu().numpy()
        flow_np = recon_flow.squeeze(0).cpu().numpy()
        z_coord = np.full((pcloud_np.shape[0], 1), idx * height_scale)
        pcloud_3D = np.hstack((pcloud_np, z_coord))
        flow_3D = np.hstack((flow_np, np.full((flow_np.shape[0], 1), height_scale)))

        cell_types = data1.truth
        colors = np.array([color_map[ct] for ct in cell_types])
        all_colors.append(colors)
        all_pclouds.append(pcloud_3D)

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

            sample_indices = np.random.choice(
                selected_indices, 
                size=min(sample_size, len(selected_indices)), 
                replace=False
            )
            

            starts = pcloud_3D[sample_indices]
            ends = starts + flow_3D[sample_indices]
            all_arrow_starts.append(starts)
            all_arrow_ends.append(ends)


    final_pcloud = pclouds[1].squeeze(0).cpu().numpy()
    final_z = (len(pcloud_list)-1)*height_scale
    final_pcloud_3D = np.hstack((final_pcloud, np.full((final_pcloud.shape[0],1), final_z)))
    final_colors = np.array([color_map[ct] for ct in pcloud_list[-1].truth])
    final_truth = pcloud_list[-1].truth
    layer_1_mask = (final_truth == finaltruth[0])
    for truth_value in finaltruth[1:]: 
        layer_1_mask |= (final_truth == truth_value)

    layer_1_pcloud_3D = final_pcloud_3D[layer_1_mask]
 
    all_pclouds.append(final_pcloud_3D)
    all_colors.append(final_colors)

    all_pclouds = np.concatenate(all_pclouds)
    all_colors = np.concatenate(all_colors)
    if len(all_arrow_starts) > 0:
        all_arrow_starts = np.concatenate(all_arrow_starts)
        all_arrow_ends = np.concatenate(all_arrow_ends)

    ax.scatter(
        all_pclouds[:,0], all_pclouds[:,1], all_pclouds[:,2],
        c=all_colors, s=size, alpha=alpha
    )
    
    if len(all_arrow_starts) > 0:
        if show_start_points:
            ax.scatter(
                all_arrow_starts[:,0], all_arrow_starts[:,1], all_arrow_starts[:,2],
                c="red", s=3, label="Start Points", alpha=0.8
            )
        ax.scatter(
            all_arrow_ends[:,0], all_arrow_ends[:,1], all_arrow_ends[:,2],
            c="blue", s=5, label="End Points", alpha=1
        )

        if show_arrow_lines:
            for start, end in zip(all_arrow_starts, all_arrow_ends):
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color="grey", linewidth=0.6, alpha=1,linestyle="--"
                )
    ax.view_init(elev=30, azim=-15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, len(pcloud_list) * height_scale)
    ax.set_axis_off()
    print(f"Number of arrow ends: {len(all_arrow_ends)}")
    print(f"Layer 1 points count: {len(layer_1_pcloud_3D)}")

    return all_arrow_ends,layer_1_pcloud_3D





def plot_all_pairs_cell_type_flow(
    graphs, aligned_models, device, finaltruth,
    selected_cell_type, arrow_sample_size_per_type=55,
    xlim=(-0.2, 1.5), ylim=(-0.5, 2), height_scale=1.0, save_path=None
):

    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(111, projection="3d")

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
    if save_path:
        fig.savefig(save_path, dpi=500)
        print(f"savepath: {save_path}")

    return all_arrow_ends

def _plot_all_pairs_flows(
    ax, graphs, aligned_models, device, finaltruth,
    arrow_sample_size_per_type, selected_cell_type,
    xlim, ylim, height_scale, title,
    show_start_points=True, show_arrow_lines=True
):
    all_pclouds = []
    all_colors = []
    all_arrow_starts = []
    all_arrow_ends = []

    all_celltypes = sorted(list({celltype for data in graphs for celltype in data.truth}))
    color_map = {celltype: cm.tab20(i / len(all_celltypes)) for i, celltype in enumerate(all_celltypes)}

    for pair_idx, model in enumerate(aligned_models):
        idx = 2 * pair_idx + 1
        if idx >= len(graphs):
            break
        
        data2, data1 = graphs[idx], graphs[idx -1]
        print(f"pair: graph{idx} -> graph{idx-1}")

        model.eval()
        with torch.no_grad():
            recon_flow, _, _, _, pclouds = model(data1.to(device), data2.to(device))

        pcloud_np = pclouds[0].squeeze(0).cpu().numpy() 
        flow_np = recon_flow.squeeze(0).cpu().numpy()   
        z_coord = np.full((pcloud_np.shape[0], 1), pair_idx * height_scale) 
        pcloud_3D = np.hstack((pcloud_np, z_coord))
        flow_3D = np.hstack((flow_np, np.full((flow_np.shape[0], 1), height_scale)))

        cell_types = data1.truth
        colors = np.array([color_map[ct] for ct in cell_types])
        all_pclouds.append(pcloud_3D)
        all_colors.append(colors)

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

            starts = pcloud_3D[sample_indices]
            ends = starts + flow_3D[sample_indices]
            all_arrow_starts.append(starts)
            all_arrow_ends.append(ends)

    final_pcloud = pclouds[1].squeeze(0).cpu().numpy()
    final_z = (len(graphs)/2) * height_scale
    final_pcloud_3D = np.hstack((final_pcloud, np.full((final_pcloud.shape[0], 1), final_z)))
    final_colors = np.array([color_map[ct] for ct in graphs[-1].truth])
    final_truth = graphs[-1].truth

    selected_mask = (final_truth == selected_cell_type)

    all_pclouds.append(final_pcloud_3D)
    all_colors.append(final_colors)

    all_pclouds = np.concatenate(all_pclouds)
    all_colors = np.concatenate(all_colors)
    if all_arrow_starts:
        all_arrow_starts = np.concatenate(all_arrow_starts)
        all_arrow_ends = np.concatenate(all_arrow_ends)

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

    if all_arrow_starts.size > 0:
        if show_start_points:
            ax.scatter(
                all_arrow_starts[:, 0], all_arrow_starts[:, 1], all_arrow_starts[:, 2],
                c="red", s=2, label="Start Points", alpha=0.7
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

    ax.view_init(elev=10, azim=-15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, (len(graphs)/2) * height_scale)
    ax.set_axis_off()

    return all_arrow_ends








def plot_multi_selected_cell_type_flow(
    pcloud_list, model, device, finaltruth,
    selected_cell_type, arrow_sample_size_per_type=55, 
    xlim=(-0.2, 1.5), ylim=(-0.5, 2), height_scale=1.0,save_path=None
):

    fig = plt.figure(figsize=(18, 8)) 
    

    ax = fig.add_subplot(111, projection="3d")
    all_arrow_ends, layer_1_pcloud_3D=_plot_multi_flows(
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
        fig.savefig(save_path, dpi=500)
        print(f"Image saved as: {save_path}")
    return all_arrow_ends, layer_1_pcloud_3D 
    
def _plot_multi_flows(
    ax, pcloud_list, model, device,finaltruth,
    arrow_sample_size_per_type, selected_cell_type,
    xlim, ylim, height_scale, title,
    show_start_points=True,
    show_arrow_lines=True  
):
    model.eval()
    

    all_pclouds = []
    all_colors = []
    all_arrow_starts = []
    all_arrow_ends = []


    all_celltypes = sorted(list({celltype for data in pcloud_list for celltype in data[0].truth}))
    color_map = {celltype: cm.tab20(i / len(all_celltypes)) for i, celltype in enumerate(all_celltypes)}
    

    for idx in range(len(pcloud_list) - 1):
        data1, data2 = pcloud_list[idx], pcloud_list[idx + 1]
        

        with torch.no_grad():
            recon_flow, _, _, _, pclouds = model(data1, data2)

        pcloud_np = pclouds[0].squeeze(0).cpu().numpy()
        flow_np = recon_flow.squeeze(0).cpu().numpy()
        z_coord = np.full((pcloud_np.shape[0], 1), idx * height_scale)
        pcloud_3D = np.hstack((pcloud_np, z_coord))
        flow_3D = np.hstack((flow_np, np.full((flow_np.shape[0], 1), height_scale)))
        
        cell_types = data1[0].truth
        colors = np.array([color_map[ct] for ct in cell_types])
        all_colors.append(colors)
        all_pclouds.append(pcloud_3D)

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
                
            sample_indices = np.random.choice(
                selected_indices, 
                size=min(sample_size, len(selected_indices)), 
                replace=False
            )
            
            starts = pcloud_3D[sample_indices]
            ends = starts + flow_3D[sample_indices]
            all_arrow_starts.append(starts)
            all_arrow_ends.append(ends)

    final_pcloud = pclouds[1].squeeze(0).cpu().numpy()
    final_z = (len(pcloud_list)-1)*height_scale
    final_pcloud_3D = np.hstack((final_pcloud, np.full((final_pcloud.shape[0],1), final_z)))
    final_colors = np.array([color_map[ct] for ct in pcloud_list[-1][0].truth])
    final_truth = pcloud_list[-1][0].truth 
    layer_1_mask = (final_truth == finaltruth[0])
    for truth_value in finaltruth[1:]: 
        layer_1_mask |= (final_truth == truth_value)

    layer_1_pcloud_3D = final_pcloud_3D[layer_1_mask]

    all_pclouds.append(final_pcloud_3D)
    all_colors.append(final_colors)


    all_pclouds = np.concatenate(all_pclouds)
    all_colors = np.concatenate(all_colors)
    if len(all_arrow_starts) > 0:
        all_arrow_starts = np.concatenate(all_arrow_starts)
        all_arrow_ends = np.concatenate(all_arrow_ends)

    
    ax.scatter(
        all_pclouds[:,0], all_pclouds[:,1], all_pclouds[:,2],
        c=all_colors, s=1, alpha=0.35
    )
    
    if len(all_arrow_starts) > 0:

        if show_start_points:
            ax.scatter(
                all_arrow_starts[:,0], all_arrow_starts[:,1], all_arrow_starts[:,2],
                c="red", s=3, label="Start Points", alpha=1
            )

        ax.scatter(
            all_arrow_ends[:,0], all_arrow_ends[:,1], all_arrow_ends[:,2],
            c="blue", s=3, label="End Points", alpha=1
        )

        if show_arrow_lines:
            for start, end in zip(all_arrow_starts, all_arrow_ends):
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color="grey", linewidth=0.6, alpha=0.8,linestyle="--"
                )

    ax.view_init(elev=30, azim=-15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, len(pcloud_list)*height_scale)
    #ax.set_title(title)
    ax.set_axis_off()
    # Before returning, print the lengths to ensure they contain data
    print(f"Number of arrow ends: {len(all_arrow_ends)}")
    print(f"Layer 1 points count: {len(layer_1_pcloud_3D)}")

    return all_arrow_ends,layer_1_pcloud_3D








import numpy as np
import torch
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial import distance
from matplotlib import cm

def compute_all_pairs_flows_and_labels(
    graphs, aligned_models, device, height_scale=1.0
):
    all_arrow_starts = []
    all_arrow_ends = []
    all_start_labels = []
    all_end_labels = []

    for pair_idx, model in enumerate(aligned_models):
        idx = 2 * pair_idx + 1 
        if idx >= len(graphs):
            break
        
        data2, data1 = graphs[idx], graphs[idx - 1] 
        model.eval()
        with torch.no_grad():
            recon_flow, _, _, _, pclouds = model(data1.to(device), data2.to(device))
        
        pcloud_np = pclouds[0].squeeze(0).cpu().numpy()
        flow_np = recon_flow.squeeze(0).cpu().numpy()
        z_coord = np.full((pcloud_np.shape[0], 1), pair_idx * height_scale)
        pcloud_3D = np.hstack((pcloud_np, z_coord))
        flow_3D = np.hstack((flow_np, np.full((flow_np.shape[0], 1), height_scale)))

        start_labels = data1.truth
        starts = pcloud_3D
        ends = starts + flow_3D

        next_pcloud_np = pclouds[1].squeeze(0).cpu().numpy()
        next_z_coord = np.full((next_pcloud_np.shape[0], 1), (pair_idx + 1) * height_scale)
        next_pcloud_3D = np.hstack((next_pcloud_np, next_z_coord))
        end_labels = transfer_labels_to_flow_ends(ends, next_pcloud_3D, data2.truth)

        all_start_labels.extend([f"slice{pair_idx}_{lbl}" for lbl in start_labels])
        all_end_labels.extend([f"slice{pair_idx + 1}_{lbl}" for lbl in end_labels])
        all_arrow_starts.append(starts)
        all_arrow_ends.append(ends)

    all_arrow_starts = np.concatenate(all_arrow_starts)
    all_arrow_ends = np.concatenate(all_arrow_ends)
    all_start_labels = np.array(all_start_labels)
    all_end_labels = np.array(all_end_labels)

    print(f" {len(all_start_labels)} pair flow")
    return all_arrow_starts, all_arrow_ends, all_start_labels, all_end_labels

def transfer_labels_to_flow_ends(all_arrow_ends, final_pcloud_3D, final_truth):
    transferred_labels = np.empty(len(all_arrow_ends), dtype=object)
    
    if len(all_arrow_ends) == 0 or len(final_pcloud_3D) == 0:
        transferred_labels.fill('Unknown')
        return transferred_labels
    
    dist_matrix = distance.cdist(all_arrow_ends, final_pcloud_3D, 'euclidean')
    nearest_indices = np.argsort(dist_matrix, axis=1)[:, :8]
    
    for i in range(len(all_arrow_ends)):
        nearest_labels = final_truth[nearest_indices[i]]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        transferred_labels[i] = unique_labels[np.argmax(counts)]
    
    return transferred_labels

def plot_multi_slice_sankey_with_same_label_priority(
    graphs, aligned_models, device, height_scale=1.0, 
    min_flow_threshold=10, save_path=None, csv_output_path=None
):

    all_arrow_starts, all_arrow_ends, all_start_labels, all_end_labels = compute_all_pairs_flows_and_labels(
        graphs=graphs, aligned_models=aligned_models, device=device, height_scale=height_scale
    )


    if csv_output_path:
        df = pd.DataFrame({
            'Start_Label': all_start_labels,
            'End_Label': all_end_labels,
            'Start_X': all_arrow_starts[:, 0],
            'Start_Y': all_arrow_starts[:, 1],
            'Start_Z': all_arrow_starts[:, 2],
            'End_X': all_arrow_ends[:, 0],
            'End_Y': all_arrow_ends[:, 1],
            'End_Z': all_arrow_ends[:, 2]
        })
        df.to_csv(csv_output_path, index=False)
        print(f"folw file save: {csv_output_path}")


    unique_labels = sorted(np.unique(np.concatenate([all_start_labels, all_end_labels])))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    source = [label_to_idx[start] for start in all_start_labels]
    target = [label_to_idx[end] for end in all_end_labels]
    value = [1] * len(source)


    flow_dict = {}
    for s, t in zip(source, target):
        flow_dict[(s, t)] = flow_dict.get((s, t), 0) + 1

    filtered_flows = {}
    for (s, t), count in flow_dict.items():
        start_label = unique_labels[s]
        end_label = unique_labels[t]
        start_type = start_label.split('_', 1)[1]
        end_type = end_label.split('_', 1)[1]
        
        if start_type == end_type:
            filtered_flows[(s, t)] = count
        elif count > min_flow_threshold:
            filtered_flows[(s, t)] = count

    if not filtered_flows:
        print(f"no link")
        return


    source_agg = [k[0] for k in filtered_flows.keys()]
    target_agg = [k[1] for k in filtered_flows.keys()]
    value_agg = list(filtered_flows.values())


    used_labels = sorted(set([unique_labels[s] for s in source_agg] + [unique_labels[t] for t in target_agg]))
    label_to_idx_filtered = {label: i for i, label in enumerate(used_labels)}


    unique_cell_types = sorted(set([lbl.split('_', 1)[1] for lbl in used_labels]))
    color_map = {ct: cm.tab20(i / len(unique_cell_types)) for i, ct in enumerate(unique_cell_types)}
    

    node_colors = []
    for label in used_labels:
        cell_type = label.split('_', 1)[1]
        rgba = color_map[cell_type]
        rgba_str = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.3)"
        node_colors.append(rgba_str)

    source_agg = [label_to_idx_filtered[unique_labels[s]] for s in source_agg]
    target_agg = [label_to_idx_filtered[unique_labels[t]] for t in target_agg]

    print(f"filter: {len(source_agg)}  > {min_flow_threshold}）")

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.01),
            label=used_labels,
            color=node_colors 
        ),
        link=dict(
            source=source_agg,
            target=target_agg,
            value=value_agg,
            color="rgba(128, 128, 128, 0.4)" 
        )
    )])

    fig_sankey.update_layout(
        title_text=f"Multi-Slice Cell Type Flow Sankey Diagram (Same Labels Retained, Others > {min_flow_threshold})",
        font_size=15,
        font_color="black"
    )

    if save_path:
        fig_sankey.write_html(save_path)
        print(f"sankeplot save: {save_path}")