import torch
import torch.nn as nn
import torch.nn.functional as F


def _compress_flat_dim(x: torch.Tensor, max_flat_dim: int) -> torch.Tensor:
    """Downsample flattened feature dimension to keep RKD memory bounded."""
    if x.size(1) <= max_flat_dim:
        return x
    x = x.unsqueeze(1)  # [B, 1, D]
    x = F.interpolate(x, size=max_flat_dim, mode="linear", align_corners=False)
    return x.squeeze(1)


def flatten_representation(
    x: torch.Tensor,
    max_tokens: int = 4,
    max_flat_dim: int = 2048,
) -> torch.Tensor:
    """
    Flatten features to [B, D] for relational comparisons.

    Uses lightweight pooling/interpolation before flattening to avoid OOM for
    high-resolution CNN features and long token sequences.
    """
    if x.dim() == 4:
        side = max(1, int(round(max_tokens**0.5)))
        x = F.adaptive_avg_pool2d(x, output_size=(side, side))
        x = x.flatten(1)
        return _compress_flat_dim(x, max_flat_dim)
    if x.dim() == 3:
        if x.size(1) != max_tokens:
            x = x.transpose(1, 2)
            x = F.interpolate(x, size=max_tokens, mode="linear", align_corners=False)
            x = x.transpose(1, 2)
        x = x.flatten(1)
        return _compress_flat_dim(x, max_flat_dim)
    if x.dim() == 2:
        return _compress_flat_dim(x, max_flat_dim)
    raise ValueError(f"Unsupported tensor shape for flattening: {tuple(x.shape)}")


def _pairwise_distance(
    x: torch.Tensor,
    eps: float = 1e-12,
    max_tokens: int = 4,
    max_flat_dim: int = 2048,
) -> torch.Tensor:
    x = flatten_representation(x, max_tokens=max_tokens, max_flat_dim=max_flat_dim)
    dist = torch.cdist(x, x, p=2)
    return torch.clamp(dist, min=eps)


def _upper_triangular_values(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    if n < 2:
        return x.new_zeros(1)
    idx = torch.triu_indices(n, n, offset=1, device=x.device)
    return x[idx[0], idx[1]]


def rkd_distance_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    max_tokens: int = 4,
    max_flat_dim: int = 2048,
) -> torch.Tensor:
    """
    RKD distance-wise loss.

    Distances are normalized by their mean pairwise distance before matching.
    """
    d_s = _pairwise_distance(
        source, max_tokens=max_tokens, max_flat_dim=max_flat_dim
    )
    d_t = _pairwise_distance(
        target, max_tokens=max_tokens, max_flat_dim=max_flat_dim
    )

    vec_s = _upper_triangular_values(d_s)
    vec_t = _upper_triangular_values(d_t)

    mean_s = vec_s.mean().detach().clamp_min(1e-12)
    mean_t = vec_t.mean().detach().clamp_min(1e-12)

    vec_s = vec_s / mean_s
    vec_t = vec_t / mean_t

    return F.smooth_l1_loss(vec_s, vec_t)


def _angle_features(
    x: torch.Tensor,
    max_tokens: int = 4,
    max_flat_dim: int = 2048,
) -> torch.Tensor:
    """Compute normalized pairwise angle features used in RKD angle loss."""
    x = flatten_representation(x, max_tokens=max_tokens, max_flat_dim=max_flat_dim)
    if x.size(0) < 3:
        return x.new_zeros(1)

    diff = x.unsqueeze(0) - x.unsqueeze(1)  # [B, B, D]
    diff = F.normalize(diff, p=2, dim=2)
    angle = torch.bmm(diff, diff.transpose(1, 2))
    return angle.reshape(-1)


def rkd_angle_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    max_tokens: int = 4,
    max_flat_dim: int = 2048,
    max_angle_batch: int = 32,
) -> torch.Tensor:
    """RKD angle-wise loss."""
    src = flatten_representation(source, max_tokens=max_tokens, max_flat_dim=max_flat_dim)
    tgt = flatten_representation(target, max_tokens=max_tokens, max_flat_dim=max_flat_dim)

    if src.size(0) > max_angle_batch:
        # Use the same sampled batch indices for source and target.
        idx = torch.randperm(src.size(0), device=src.device)[:max_angle_batch]
        src = src[idx]
        tgt = tgt[idx]

    a_s = _angle_features(src, max_tokens=max_tokens, max_flat_dim=max_flat_dim)
    a_t = _angle_features(tgt, max_tokens=max_tokens, max_flat_dim=max_flat_dim)
    return F.smooth_l1_loss(a_s, a_t)


class RKDLoss(nn.Module):
    """Convenience wrapper returning both distance and angle losses."""

    def __init__(self):
        super().__init__()

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        dist = rkd_distance_loss(source, target)
        angle = rkd_angle_loss(source, target)
        return dist, angle
