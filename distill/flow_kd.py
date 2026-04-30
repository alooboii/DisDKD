import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module


class FeatureHooks:
    """Capture intermediate features from named layers."""

    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []

        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output

            return _hook

        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))

    def clear(self):
        self.features.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class VelocityHead(nn.Module):
    """
    Small MLP head for velocity prediction.

    Inputs:
      - pooled student feature h_s(x)
      - interpolated logits z_tau
      - tau embedding
    """

    def __init__(
        self, student_channels, num_classes, time_emb_dim=32, hidden_dim=256
    ):
        super(VelocityHead, self).__init__()
        self.num_classes = num_classes
        input_dim = student_channels + num_classes + time_emb_dim
        self.tau_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, student_feat, z_tau, tau):
        if student_feat.dim() == 4:
            h = F.adaptive_avg_pool2d(student_feat, 1).flatten(1)
        else:
            h = student_feat.flatten(1)

        tau = tau.view(-1, 1)
        tau_emb = self.tau_embed(tau)
        combined = torch.cat([h, z_tau, tau_emb], dim=1)
        return self.predictor(combined)


class FlowKD(nn.Module):
    """
    Flow-Matching KD over teacher logits.
    """

    def __init__(
        self,
        teacher,
        student,
        student_layer,
        student_channels,
        num_classes,
        base_logits="zeros",
        flow_target="logits",
        temperature=4.0,
        time_emb_dim=32,
        head_hidden_dim=256,
    ):
        super(FlowKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.student_layer = student_layer
        self.num_classes = num_classes
        self.base_logits = base_logits
        self.flow_target = flow_target
        self.temperature = temperature

        if student_channels is None:
            raise ValueError(
                f"FlowKD requires known channel count for layer '{student_layer}'."
            )

        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )
        self.velocity_head = VelocityHead(
            student_channels=student_channels,
            num_classes=num_classes,
            time_emb_dim=time_emb_dim,
            hidden_dim=head_hidden_dim,
        )

    def _build_base_logits(self, z_t):
        if self.base_logits == "zeros":
            return torch.zeros_like(z_t)
        if self.base_logits == "gaussian":
            return torch.randn_like(z_t)
        raise ValueError(
            f"Unknown base_logits mode: {self.base_logits}. Use zeros|gaussian."
        )

    def _build_velocity_target(self, z_t, z_0):
        if self.flow_target == "logits":
            return z_t - z_0

        if self.flow_target == "probabilities":
            p_t = F.softmax(z_t / self.temperature, dim=1)
            p_0 = F.softmax(z_0 / self.temperature, dim=1)
            return p_t - p_0

        raise ValueError(
            f"Unknown flow_target mode: {self.flow_target}. Use logits|probabilities."
        )

    def forward(self, x):
        with torch.no_grad():
            z_t = self.teacher(x)

        student_logits = self.student(x)
        student_feat = self.student_hooks.features.get(self.student_layer)
        if student_feat is None:
            raise ValueError(f"Missing student features for layer: {self.student_layer}")

        z_0 = self._build_base_logits(z_t)
        tau = torch.rand(z_t.size(0), 1, device=z_t.device)
        z_tau = (1.0 - tau) * z_0 + tau * z_t
        v_star = self._build_velocity_target(z_t, z_0)
        v_pred = self.velocity_head(student_feat, z_tau, tau)

        if v_pred.shape != v_star.shape:
            raise RuntimeError(
                f"FlowKD shape mismatch: v_pred {v_pred.shape} vs v_star {v_star.shape}"
            )

        fm_loss = F.mse_loss(v_pred, v_star)

        self.student_hooks.clear()
        return z_t, student_logits, fm_loss
