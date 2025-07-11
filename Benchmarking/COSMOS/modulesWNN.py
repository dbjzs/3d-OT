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

Notice: The DeepGraphInfomaxWNN function in the COSMOS software is adapted from the 
        torch_geometric.nn.models.deep_graph_infomax function in PyTorch Geometric (PyG),
        https://github.com/pyg-team/pytorch_geometric/tree/master.
        Please see the "LICENSE" file for copyright details of the PyG software.
'''
import copy
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from torch_geometric.nn.inits import reset, uniform

EPS = 1e-15


class DeepGraphInfomaxWNN(torch.nn.Module):
    """The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """
    def __init__(
        self,
        hidden_channels: int,
        encoder: Module,
        summary: Callable,
        corruption: Callable,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)


    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z,w1,w2 = self.encoder(*args, **kwargs)

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        cor_args = cor[:len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value

        neg_z,_,_ = self.encoder(*cor_args, **cor_kwargs)

        summary = self.summary(pos_z, *args, **kwargs)

        return pos_z, neg_z, summary,w1,w2


    def discriminate(self, z: Tensor, summary: Tensor,
                     sigmoid: bool = True) -> Tensor:
        """Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (torch.Tensor): The latent space.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value


    def loss(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor) -> Tensor:
        """Computes the mutual information maximization objective."""
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


    def test(
        self,
        train_z: Tensor,
        train_y: Tensor,
        test_z: Tensor,
        test_y: Tensor,
        solver: str = 'lbfgs',
        multi_class: str = 'auto',
        *args,
        **kwargs,
    ) -> float:
        """Evaluates latent space quality via a logistic regression downstream
        task."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'
