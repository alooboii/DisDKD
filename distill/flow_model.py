import random
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualTokenBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class PairConditionedVelocity(nn.Module):
    """Velocity model v_phi(z, t, a, b) on tokenized latent states."""

    def __init__(
        self,
        z_dim: int,
        num_layers: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_layers = num_layers

        self.time_mlp = nn.Sequential(
            nn.Linear(1, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )
        self.start_embed = nn.Embedding(num_layers, z_dim)
        self.end_embed = nn.Embedding(num_layers, z_dim)

        self.blocks = nn.ModuleList(
            [ResidualTokenBlock(z_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Linear(z_dim, z_dim)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"Expected z shape [B, T, D], got {tuple(z.shape)}")

        bsz = z.size(0)
        t = t.view(-1, 1)
        if t.size(0) == 1 and bsz > 1:
            t = t.expand(bsz, -1)

        cond = self.time_mlp(t)
        s = self.start_embed.weight[start_idx].view(1, -1).expand(bsz, -1)
        e = self.end_embed.weight[end_idx].view(1, -1).expand(bsz, -1)
        cond = cond + s + e

        x = z + cond.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


def build_layer_pairs(layer_names: Sequence[str], mode: str = "adjacent") -> List[Tuple[int, int]]:
    n = len(layer_names)
    if n < 2:
        return []

    adjacent = [(i, i + 1) for i in range(n - 1)]
    long_pairs = [(i, j) for i in range(n) for j in range(i + 2, n)]

    if mode == "adjacent":
        return adjacent
    if mode == "long":
        return long_pairs
    if mode == "mixed":
        return adjacent + long_pairs

    raise ValueError(f"Unknown pair mode: {mode}. Use adjacent|long|mixed")


def sample_layer_pairs(
    layer_names: Sequence[str],
    mode: str,
    num_pairs: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    candidates = build_layer_pairs(layer_names, mode)
    if not candidates:
        return []

    k = min(max(1, num_pairs), len(candidates))
    if k == len(candidates):
        return list(candidates)
    return rng.sample(candidates, k)


def euler_integrate(
    velocity_model: PairConditionedVelocity,
    z0: torch.Tensor,
    start_idx: int,
    end_idx: int,
    steps: int,
    t0: float = 0.0,
    t1: float = 1.0,
):
    """Euler integration from t0 to t1."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    z = z0
    dt = (t1 - t0) / float(steps)

    for k in range(steps):
        t = t0 + (k / float(steps)) * (t1 - t0)
        t_tensor = z.new_full((z.size(0), 1), t)
        v = velocity_model(z, t_tensor, start_idx, end_idx)
        z = z + dt * v

    return z


def euler_rollout_path(
    velocity_model: PairConditionedVelocity,
    z0: torch.Tensor,
    start_idx: int,
    end_idx: int,
    steps: int,
):
    """
    Full Euler rollout over [0,1], returning checkpoints at each step including t=0.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    z = z0
    dt = 1.0 / float(steps)
    states = [z0]

    for k in range(steps):
        t = k / float(steps)
        t_tensor = z.new_full((z.size(0), 1), t)
        v = velocity_model(z, t_tensor, start_idx, end_idx)
        z = z + dt * v
        states.append(z)

    return states


def latent_endpoint_error(z_hat_b: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z_hat_b, z_b)


def compute_flow_losses_for_pairs(
    velocity_model: PairConditionedVelocity,
    z_by_layer: Dict[str, torch.Tensor],
    layer_names: Sequence[str],
    pairs: Sequence[Tuple[int, int]],
    steps: int,
    lambda_fm: float = 1.0,
    lambda_path: float = 1.0,
    lambda_end: float = 1.0,
):
    """Compute FM + path + endpoint losses for sampled layer pairs."""
    if len(pairs) == 0:
        zero = next(velocity_model.parameters()).new_tensor(0.0)
        return zero, {
            "total": 0.0,
            "fm": 0.0,
            "path": 0.0,
            "end": 0.0,
        }

    fm_terms = []
    path_terms = []
    end_terms = []

    for start_idx, end_idx in pairs:
        start_layer = layer_names[start_idx]
        end_layer = layer_names[end_idx]

        z_a = z_by_layer[start_layer]
        z_b = z_by_layer[end_layer]

        t = z_a.new_empty((z_a.size(0), 1)).uniform_(0.0, 1.0)
        z_t = (1.0 - t).unsqueeze(-1) * z_a + t.unsqueeze(-1) * z_b
        u_t = z_b - z_a

        v_pred = velocity_model(z_t, t, start_idx, end_idx)
        if v_pred.shape != u_t.shape:
            raise RuntimeError(
                f"Flow shape mismatch for pair ({start_layer}->{end_layer}): "
                f"v_pred={tuple(v_pred.shape)} vs target={tuple(u_t.shape)}"
            )
        fm_terms.append(F.mse_loss(v_pred, u_t))

        rollout = euler_rollout_path(
            velocity_model=velocity_model,
            z0=z_a,
            start_idx=start_idx,
            end_idx=end_idx,
            steps=steps,
        )
        z_hat_end = rollout[-1]
        end_terms.append(F.mse_loss(z_hat_end, z_b))

        if end_idx - start_idx > 1:
            inter_losses = []
            for mid_idx in range(start_idx + 1, end_idx):
                tau_mid = (mid_idx - start_idx) / float(end_idx - start_idx)
                step_idx = int(round(tau_mid * steps))
                step_idx = max(0, min(step_idx, steps))
                z_hat_mid = rollout[step_idx]
                z_mid = z_by_layer[layer_names[mid_idx]]
                if z_hat_mid.shape != z_mid.shape:
                    raise RuntimeError(
                        f"Path shape mismatch at mid layer {layer_names[mid_idx]}: "
                        f"z_hat={tuple(z_hat_mid.shape)} vs z_mid={tuple(z_mid.shape)}"
                    )
                inter_losses.append(F.mse_loss(z_hat_mid, z_mid))

            if inter_losses:
                path_terms.append(torch.stack(inter_losses).mean())

    device = next(velocity_model.parameters()).device
    fm = torch.stack(fm_terms).mean() if fm_terms else torch.tensor(0.0, device=device)
    path = (
        torch.stack(path_terms).mean() if path_terms else torch.tensor(0.0, device=device)
    )
    end = torch.stack(end_terms).mean() if end_terms else torch.tensor(0.0, device=device)

    total = lambda_fm * fm + lambda_path * path + lambda_end * end

    metrics = {
        "total": total.item(),
        "fm": fm.item(),
        "path": path.item(),
        "end": end.item(),
    }
    return total, metrics
