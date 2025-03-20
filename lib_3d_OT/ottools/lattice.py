import os
import sys
import numpy as np
import time
import torch

# add path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)


def fetch_knn_points(point_cloud, knn_indices):
        b, n, k = knn_indices.shape
        knn_indices=knn_indices.unsqueeze(-1).expand(-1,-1,-1,3)
        point_cloud=point_cloud.unsqueeze(2).expand(-1,-1,k,-1)
        knn_points=torch.gather(point_cloud,1,knn_indices)

        return knn_points

class Lattice(torch.nn.Module):
    def __init__(self, nb, spacing, batchsize, xmin=0, ymin=0, xmax=2 * np.pi, ymax=2 * np.pi, device=None):
        super(Lattice, self).__init__()
        self.nb = nb
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.spacing = spacing

        x = torch.linspace(self.xmin, self.xmax, self.spacing)
        y = torch.linspace(self.ymin, self.ymax, self.spacing)
        xx, yy = torch.meshgrid(x, y)

        self.grid_coords = torch.concat([xx.unsqueeze(-1), yy.unsqueeze(-1)], dim=-1).view(-1, 2).contiguous().repeat(batchsize, 1, 1).to(device)
        self.grid_flows = torch.zeros((batchsize, self.grid_coords.size(1), 2)).to(device)



    def idw_interpolation(self, point_cloud, value, grid_coords, power=2):

        dist = torch.cdist(grid_coords, point_cloud, p=2) 
        
        knn_dist, knn_idx = torch.topk(dist, k=self.nb, largest=False, dim=-1)
        knn_values = torch.gather(
            value.unsqueeze(1).expand(-1, grid_coords.size(1), -1, -1), 
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, value.size(-1)),
        )
        
        weights = torch.pow(knn_dist + 1e-8, -power) 
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        interpolated_values = torch.sum(weights.unsqueeze(-1) * knn_values, dim=2)
        return interpolated_values


    def splat(self,flow,coords):
        interpolated_values=self.idw_interpolation(coords,flow,self.grid_coords)
        self.grid_flows=interpolated_values

    def gradient(self):
        grid_flows = self.grid_flows.view(-1, self.spacing, self.spacing, 2).contiguous()
        dFx_dx = torch.gradient(grid_flows[:, :, :, 0], spacing=(2 * np.pi / self.spacing), dim=1)[0]  # ∂Fx/∂x
        dFy_dy = torch.gradient(grid_flows[:, :, :, 1], spacing=(2 * np.pi / self.spacing), dim=2)[0]  # ∂Fy/∂y
        return dFx_dx, dFy_dy

        

    
    def divergence_per_point(self):
        dFx_dx, dFy_dy = self.gradient()
        return (dFx_dx + dFy_dy).view(-1, self.spacing * self.spacing).contiguous()
    
    def divergence(self):
        div_pp = self.divergence_per_point()
        div_pp_norm = torch.abs(div_pp)
        return torch.mean(div_pp_norm, dim=-1).mean(dim=-1)
    

    def forward(self, flow, coords):
        """
        Forward pass for 2D lattice divergence computation.
        """
        self.splat(flow=flow, coords=coords)
        return self.divergence()

if __name__ == "__main__":
    l = Lattice(nb=8, spacing=10, batchsize=2)
    coords = (torch.rand((2, 2048, 2)) * 2 * np.pi).cuda()
    coords.requires_grad = True
    flow = coords**2 - 30 

    print(l(flow, coords)) 
    from torch.autograd import grad

    def gradient(inputs, outputs, create_graph=True, retain_graph=True):
        d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
        points_grad = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]
        return points_grad

    dx = gradient(coords, flow[:, :, 0])
    dy = gradient(coords, flow[:, :, 1])
    div = dx[:, :, 0] + dy[:, :, 1]
    print(div.mean(dim=-1)) 
    