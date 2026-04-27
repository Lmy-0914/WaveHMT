import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    """Edge loss for super-resolution.

    Computes the L1 loss between edges extracted from SR and HR images
    using Sobel or Laplacian filters.

    Args:
        loss_weight (float): Weight of this loss. Default: 1.0.
        reduction (str): Reduction method. Default: 'mean'.
        edge_type (str): Edge detector type, 'sobel' or 'laplacian'. Default: 'sobel'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', edge_type='sobel'):
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.edge_type = edge_type

        if edge_type == 'sobel':
            kernel_x = torch.tensor(
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

            kernel_y = torch.tensor(
                [[-1, -2, -1],
                 [0, 0, 0],
                 [1, 2, 1]], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)

            self.register_buffer('kernel_x', kernel_x)
            self.register_buffer('kernel_y', kernel_y)

        elif edge_type == 'laplacian':
            kernel = torch.tensor(
                [[0, -1, 0],
                 [-1, 4, -1],
                 [0, -1, 0]], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)
            self.register_buffer('kernel', kernel)
        else:
            raise ValueError(f"edge_type must be 'sobel' or 'laplacian', got {edge_type}")

    def extract_edges(self, x):
        b, c, h, w = x.shape

        x_flat = x.view(b * c, 1, h, w)

        if self.edge_type == 'sobel':
            edge_x = F.conv2d(x_flat, self.kernel_x, padding=1)
            edge_y = F.conv2d(x_flat, self.kernel_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        else:  # laplacian
            edge = F.conv2d(x_flat, self.kernel, padding=1)
            edge = torch.abs(edge)

        return edge.view(b, c, h, w)

    def forward(self, sr, hr):
        """
        Args:
            sr (Tensor): Super-resolved image, shape (B, C, H, W).
            hr (Tensor): High-resolution ground truth, shape (B, C, H, W).
        Returns:
            Tensor: Edge loss value.
        """
        sr_edge = self.extract_edges(sr)
        hr_edge = self.extract_edges(hr)

        loss = F.l1_loss(sr_edge, hr_edge, reduction=self.reduction)
        return self.loss_weight * loss