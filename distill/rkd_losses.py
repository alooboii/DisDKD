import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_representation(x: torch.Tensor) -> torch.Tensor:
    """Flatten features to [B, D] for relational comparisons."""
    if x.dim() == 4:
        return x.flatten(1)
    if x.dim() == 3:
        return x.flatten(1)
    if x.dim() == 2:
        return x
    raise ValueError(f"Unsupported tensor shape for flattening: {tuple(x.shape)}")


def _pairwise_distance(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = flatten_representation(x)
    dist = torch.cdist(x, x, p=2)
    return torch.clamp(dist, min=eps)


def _upper_triangular_values(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    if n < 2:
        return x.new_zeros(1)
    idx = torch.triu_indices(n, n, offset=1, device=x.device)
    return x[idx[0], idx[1]]


def rkd_distance_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    RKD distance-wise loss.

    Distances are normalized by their mean pairwise distance before matching.
    """
    d_s = _pairwise_distance(source)
    d_t = _pairwise_distance(target)

    vec_s = _upper_triangular_values(d_s)
    vec_t = _upper_triangular_values(d_t)

    mean_s = vec_s.mean().detach().clamp_min(1e-12)
    mean_t = vec_t.mean().detach().clamp_min(1e-12)

    vec_s = vec_s / mean_s
    vec_t = vec_t / mean_t

    return F.smooth_l1_loss(vec_s, vec_t)


def _angle_features(x: torch.Tensor) -> torch.Tensor:
    """Compute normalized pairwise angle features used in RKD angle loss."""
    x = flatten_representation(x)
    if x.size(0) < 3:
        return x.new_zeros(1)

    diff = x.unsqueeze(0) - x.unsqueeze(1)  # [B, B, D]
    diff = F.normalize(diff, p=2, dim=2)
    angle = torch.bmm(diff, diff.transpose(1, 2))
    return angle.reshape(-1)


def rkd_angle_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RKD angle-wise loss."""
    a_s = _angle_features(source)
    a_t = _angle_features(target)
    return F.smooth_l1_loss(a_s, a_t)


class RKDLoss(nn.Module):
    """Convenience wrapper returning both distance and angle losses."""

    def __init__(self):
        super().__init__()

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        dist = rkd_distance_loss(source, target)
        angle = rkd_angle_loss(source, target)
        return dist, angle
