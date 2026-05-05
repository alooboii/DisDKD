from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Build sinusoidal embeddings for scalar time values in [0,1]."""
    if dim <= 0:
        raise ValueError("time embedding dim must be positive")

    t = t.view(-1, 1)
    half = dim // 2
    if half == 0:
        return t

    device = t.device
    freq = torch.exp(
        -torch.arange(half, device=device, dtype=t.dtype)
        * (torch.log(torch.tensor(10000.0, device=device, dtype=t.dtype)) / max(half - 1, 1))
    )
    angles = t * freq.view(1, -1)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _VelocityBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int,
        use_attention: bool,
        num_heads: int,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        if use_attention:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
                )
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True,
            )
        else:
            self.attn = None

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.mlp(h2)
        return x


class DirectTrajectoryVelocity(nn.Module):
    """Time-conditioned velocity model for ViT hidden-state trajectories."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 512,
        num_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 8,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _VelocityBlock(
                    hidden_dim=hidden_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                    use_attention=use_attention,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"Expected hidden state [B, N, D], got {tuple(h.shape)}")
        if h.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch in velocity model: got {h.size(-1)}, expected {self.hidden_dim}"
            )

        bsz = h.size(0)
        t = t.view(-1, 1)
        if t.size(0) == 1 and bsz > 1:
            t = t.expand(bsz, 1)
        if t.size(0) != bsz:
            raise ValueError(f"Time batch mismatch: t batch={t.size(0)}, hidden batch={bsz}")

        time_emb = _sinusoidal_time_embedding(t, self.time_embed_dim)
        time_bias = self.time_mlp(time_emb).unsqueeze(1)

        x = self.in_norm(h) + time_bias
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        v = self.out_proj(x)
        if v.shape != h.shape:
            raise RuntimeError(
                f"Velocity output shape mismatch: got {tuple(v.shape)}, expected {tuple(h.shape)}"
            )
        return v


def euler_solve(
    velocity_model: DirectTrajectoryVelocity,
    h0: torch.Tensor,
    start_t: float,
    end_t: float,
    steps: int,
    return_states: bool = False,
):
    """Euler integrate dh/dt=v(h,t) from start_t to end_t."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    h = h0
    dt = (end_t - start_t) / float(steps)
    states = [h0] if return_states else None

    for k in range(steps):
        tk = start_t + (k / float(steps)) * (end_t - start_t)
        t = h.new_full((h.size(0), 1), tk)
        v = velocity_model(h, t)
        if v.shape != h.shape:
            raise RuntimeError(
                f"Flow shape check failed: velocity {tuple(v.shape)} vs hidden {tuple(h.shape)}"
            )
        h = h + dt * v
        if return_states:
            states.append(h)

    if return_states:
        return h, states
    return h


def _state_from_rollout(
    rollout: Sequence[torch.Tensor],
    tau: float,
    start_t: float,
    end_t: float,
) -> torch.Tensor:
    steps = len(rollout) - 1
    if steps <= 0:
        raise ValueError("rollout must contain at least 2 states")

    if end_t <= start_t:
        raise ValueError("end_t must be greater than start_t")

    rel = (tau - start_t) / float(end_t - start_t)
    rel = min(1.0, max(0.0, rel))
    idx = int(round(rel * steps))
    idx = max(0, min(idx, steps))
    return rollout[idx]


def compute_fm_loss(
    velocity_model: DirectTrajectoryVelocity,
    h_start: torch.Tensor,
    h_end: torch.Tensor,
) -> torch.Tensor:
    t = h_start.new_empty((h_start.size(0), 1)).uniform_(0.0, 1.0)
    h_t = (1.0 - t).unsqueeze(-1) * h_start + t.unsqueeze(-1) * h_end
    target_v = h_end - h_start
    pred_v = velocity_model(h_t, t)
    if pred_v.shape != target_v.shape:
        raise RuntimeError(
            f"FM shape mismatch: pred {tuple(pred_v.shape)} vs target {tuple(target_v.shape)}"
        )
    return F.mse_loss(pred_v, target_v)


def compute_end_loss(
    velocity_model: DirectTrajectoryVelocity,
    h_start: torch.Tensor,
    h_end: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_hat = euler_solve(
        velocity_model=velocity_model,
        h0=h_start,
        start_t=0.0,
        end_t=1.0,
        steps=steps,
        return_states=False,
    )
    loss = F.mse_loss(h_hat, h_end)
    return loss, h_hat


def compute_path_loss(
    velocity_model: DirectTrajectoryVelocity,
    h_start: torch.Tensor,
    targets_by_tau: Sequence[Tuple[float, torch.Tensor]],
    steps: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Path constraints at intermediate times tau in [0,1]."""
    if len(targets_by_tau) == 0:
        zero = h_start.new_tensor(0.0)
        return zero, {}

    _, rollout = euler_solve(
        velocity_model=velocity_model,
        h0=h_start,
        start_t=0.0,
        end_t=1.0,
        steps=steps,
        return_states=True,
    )

    losses = []
    detail = {}
    for tau, h_target in targets_by_tau:
        h_hat = _state_from_rollout(rollout, tau=tau, start_t=0.0, end_t=1.0)
        l = F.mse_loss(h_hat, h_target)
        losses.append(l)
        detail[f"path_tau_{tau:.4f}"] = l.item()

    return torch.stack(losses).mean(), detail


def generate_teacher_trajectory_targets(
    h_start: torch.Tensor,
    velocity_model: DirectTrajectoryVelocity,
    student_num_layers: int,
    flow_steps: int,
    start_t: float = 0.0,
    end_t: float = 1.0,
) -> List[torch.Tensor]:
    """Generate continuous teacher trajectory targets at student depths."""
    if student_num_layers <= 0:
        raise ValueError("student_num_layers must be positive")

    _, rollout = euler_solve(
        velocity_model=velocity_model,
        h0=h_start,
        start_t=start_t,
        end_t=end_t,
        steps=flow_steps,
        return_states=True,
    )

    targets: List[torch.Tensor] = []
    for k in range(1, student_num_layers + 1):
        tau = start_t + (k / float(student_num_layers)) * (end_t - start_t)
        target = _state_from_rollout(rollout, tau=tau, start_t=start_t, end_t=end_t)
        targets.append(target)

    return targets
