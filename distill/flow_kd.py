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
      - interpolated state state_tau (logits or probabilities)
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

    def forward(self, student_feat, state_tau, tau):
        if student_feat.dim() == 4:
            h = F.adaptive_avg_pool2d(student_feat, 1).flatten(1)
        else:
            h = student_feat.flatten(1)

        tau = tau.view(-1, 1)
        tau_emb = self.tau_embed(tau)
        combined = torch.cat([h, state_tau, tau_emb], dim=1)
        return self.predictor(combined)


class FlowKD(nn.Module):
    """
    Flow-Matching KD over teacher logits.

    Notes:
      - Logit flow is the default and usually the most stable branch.
      - Probability flow must interpolate probabilities (not logits).
      - Velocity head is a training-only auxiliary module and is not used for
        inference-time predictions.
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
        debug=False,
    ):
        super(FlowKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.student_layer = student_layer
        self.num_classes = num_classes
        self.base_logits = base_logits
        self.flow_target = flow_target
        self.temperature = temperature
        self.debug = debug
        self._debug_printed = False

        if student_channels is None:
            raise ValueError(
                f"FlowKD requires known channel count for layer '{student_layer}'."
            )

        if self.debug:
            print(
                f"[FlowKD INIT] flow_target={self.flow_target} "
                f"base_logits={self.base_logits} temperature={self.temperature}"
            )

        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )
        self.velocity_head = VelocityHead(
            student_channels=student_channels,
            num_classes=num_classes,
            time_emb_dim=time_emb_dim,
            hidden_dim=head_hidden_dim,
        )

    def train(self, mode=True):
        """Keep teacher in eval mode even when student/head are in train mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def _build_base_logits(self, z_t):
        if self.base_logits == "zeros":
            return torch.zeros_like(z_t)
        if self.base_logits == "gaussian":
            return torch.randn_like(z_t)
        raise ValueError(
            f"Unknown base_logits mode: {self.base_logits}. Use zeros|gaussian."
        )

    def _build_state_and_velocity_target(self, z_t, z_0, tau):
        # Default stable branch: interpolate and match velocity in logit space.
        if self.flow_target == "logits":
            state_tau = (1.0 - tau) * z_0 + tau * z_t
            v_star = z_t - z_0
            return state_tau, v_star

        # Probability-flow branch: interpolate probabilities, then match
        # probability-space velocity. Do not mix logit-space state with
        # probability-space target.
        if self.flow_target == "probabilities":
            p_t = F.softmax(z_t / self.temperature, dim=1)
            p_0 = F.softmax(z_0 / self.temperature, dim=1)
            state_tau = (1.0 - tau) * p_0 + tau * p_t
            v_star = p_t - p_0
            return state_tau, v_star

        raise ValueError(
            f"Unknown flow_target mode: {self.flow_target}. Use logits|probabilities."
        )

    def forward(self, x):
        with torch.no_grad():
            z_t = self.teacher(x).detach()

        # Clear stale features from previous batch before student forward.
        self.student_hooks.clear()
        student_logits = self.student(x)
        student_feat = self.student_hooks.features.get(self.student_layer)
        if student_feat is None:
            raise ValueError(f"Missing student features for layer: {self.student_layer}")

        z_0 = self._build_base_logits(z_t)
        tau = torch.rand(z_t.size(0), 1, device=z_t.device)
        state_tau, v_star = self._build_state_and_velocity_target(z_t, z_0, tau)
        v_pred = self.velocity_head(student_feat, state_tau, tau)

        if v_pred.shape != v_star.shape:
            raise RuntimeError(
                f"FlowKD shape mismatch: v_pred {v_pred.shape} vs v_star {v_star.shape}"
            )

        fm_loss = F.mse_loss(v_pred, v_star)

        if self.debug and not self._debug_printed:
            with torch.no_grad():
                teacher_row_sum = z_t[:5].sum(dim=1)
                teacher_softmax_row_sum = F.softmax(z_t, dim=1)[:5].sum(dim=1)

                print(f"[FlowKD DEBUG] flow_target: {self.flow_target}")
                print(f"[FlowKD DEBUG] base_logits: {self.base_logits}")
                print(
                    "[FlowKD DEBUG] z_t min/max/mean/std:",
                    z_t.min().item(),
                    z_t.max().item(),
                    z_t.mean().item(),
                    z_t.std().item(),
                )
                print("[FlowKD DEBUG] teacher row sum first 5:", teacher_row_sum)
                print(
                    "[FlowKD DEBUG] teacher softmax row sum first 5:",
                    teacher_softmax_row_sum,
                )
                print("[FlowKD DEBUG] z_0 std:", z_0.std().item())
                print("[FlowKD DEBUG] state_tau mean/std:", state_tau.mean().item(), state_tau.std().item())
                print("[FlowKD DEBUG] v_star mean abs:", v_star.abs().mean().item())
                print("[FlowKD DEBUG] v_star std:", v_star.std().item())
                print("[FlowKD DEBUG] v_star mse from zero:", (v_star ** 2).mean().item())
                print("[FlowKD DEBUG] v_pred mean/std:", v_pred.mean().item(), v_pred.std().item())
                print("[FlowKD DEBUG] fm_loss:", fm_loss.item())
                # Logit-flow FM usually has larger scale than probability-flow FM.
                # If fm_loss is unexpectedly tiny in logit mode, inspect z_t/v_star stats.
            self._debug_printed = True

        self.student_hooks.clear()
        return z_t, student_logits, fm_loss
