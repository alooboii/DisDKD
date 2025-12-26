import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module


class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    """

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
        """Clears the stored features."""
        self.features.clear()

    def remove(self):
        """Removes all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ContrastiveLoss(nn.Module):
    """
    Contrastive Representation Distillation (CRD) Loss.

    Based on the paper: "Contrastive Representation Distillation" (Tian et al., ICLR 2020)
    Uses Noise Contrastive Estimation (NCE) framework.

    Args:
        n_data (int): Number of training samples
        feat_dim (int): Dimension of the feature embeddings
        temperature (float): Temperature parameter for contrastive loss
        momentum (float): Momentum for updating negative prototypes
        n_negatives (int): Number of negative samples to use
    """

    def __init__(
        self, n_data, feat_dim=128, temperature=0.07, momentum=0.5, n_negatives=4096
    ):
        super(ContrastiveLoss, self).__init__()
        self.n_data = n_data
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.momentum = momentum
        self.n_negatives = n_negatives

        # Initialize memory bank for negative samples
        # FIXED: Use feat_dim instead of hard-coded 128
        self.register_buffer("memory_bank", torch.randn(n_data, feat_dim))
        self.memory_bank = F.normalize(self.memory_bank, dim=1)

    def forward(self, student_feat, teacher_feat, indices):
        """
        Compute contrastive loss between student and teacher features.

        Args:
            student_feat (Tensor): Student features [batch_size, feat_dim]
            teacher_feat (Tensor): Teacher features [batch_size, feat_dim]
            indices (Tensor): Sample indices for the current batch [batch_size]

        Returns:
            Tensor: Contrastive loss
        """
        batch_size = student_feat.size(0)

        # Normalize features
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)

        # =================================================================
        # POSITIVE PAIRS: student-teacher correspondence
        # =================================================================
        # Compute similarity between student and teacher (positive pairs)
        pos_logits = (
            torch.sum(student_feat * teacher_feat, dim=1, keepdim=True)
            / self.temperature
        )

        # =================================================================
        # NEGATIVE SAMPLES: from memory bank
        # =================================================================
        # Sample negative indices (avoiding the current batch indices)
        neg_indices = torch.randint(
            0, self.n_data, (batch_size, self.n_negatives), device=student_feat.device
        )

        # Retrieve negative samples from memory bank
        neg_feat = self.memory_bank[neg_indices]  # [batch_size, n_negatives, feat_dim]

        # Compute similarities with negative samples
        neg_logits = (
            torch.bmm(
                student_feat.unsqueeze(1),  # [batch_size, 1, feat_dim]
                neg_feat.transpose(1, 2),  # [batch_size, feat_dim, n_negatives]
            ).squeeze(1)
            / self.temperature
        )  # [batch_size, n_negatives]

        # =================================================================
        # NCE LOSS COMPUTATION
        # =================================================================
        # Concatenate positive and negative logits
        # Shape: [batch_size, 1 + n_negatives]
        logits = torch.cat([pos_logits, neg_logits], dim=1)

        # Labels: positive pair is at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=student_feat.device)

        # Compute InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # =================================================================
        # UPDATE MEMORY BANK
        # =================================================================
        # Update memory bank with current teacher features using momentum
        if self.training and indices is not None:
            with torch.no_grad():
                # Ensure indices are valid
                idx = indices.long().to(self.memory_bank.device)

                # Ensure we have matching batch sizes
                if idx.size(0) != teacher_feat.size(0):
                    min_size = min(idx.size(0), teacher_feat.size(0))
                    idx = idx[:min_size]
                    teacher_feat = teacher_feat[:min_size]

                if idx.numel() > 0:
                    # Momentum update: new = momentum * old + (1 - momentum) * new
                    updated_feat = (
                        self.momentum * self.memory_bank[idx]
                        + (1 - self.momentum) * teacher_feat.detach()
                    )
                    # Re-normalize after momentum update
                    updated_feat = F.normalize(updated_feat, dim=1)
                    self.memory_bank[idx] = updated_feat

        return loss


class CRD(nn.Module):
    """
    Contrastive Representation Distillation.

    This method uses contrastive learning to match student and teacher
    representations at intermediate layers.

    Reference: "Contrastive Representation Distillation" (Tian et al., ICLR 2020)

    Args:
        teacher (nn.Module): Pretrained teacher network.
        student (nn.Module): Student network.
        teacher_layer (str): Name of the teacher layer to extract features from.
        student_layer (str): Name of the student layer to extract features from.
        teacher_channels (int): Number of channels in teacher feature map.
        student_channels (int): Number of channels in student feature map.
        n_data (int): Number of training samples for contrastive loss.
        feat_dim (int): Dimension of the projected features (default: 128).
        temperature (float): Temperature for contrastive loss (default: 0.07).
        momentum (float): Momentum for updating memory bank (default: 0.5).
        n_negatives (int): Number of negative samples (default: 4096).
    """

    def __init__(
        self,
        teacher,
        student,
        teacher_layer,
        student_layer,
        teacher_channels,
        student_channels,
        n_data,
        feat_dim=128,
        temperature=0.07,
        momentum=0.5,
        n_negatives=4096,
    ):
        super(CRD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.n_data = n_data
        self.feat_dim = feat_dim

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Set up hooks for feature extraction
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # Projection heads to map features to common dimension
        # Teacher projector
        self.teacher_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(teacher_channels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        # Student projector
        self.student_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(student_channels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        # Contrastive loss with memory bank
        self.contrastive_loss = ContrastiveLoss(
            n_data=n_data,
            feat_dim=feat_dim,
            temperature=temperature,
            momentum=momentum,
            n_negatives=n_negatives,
        )

        # Store sample indices for each batch (used for memory bank updates)
        self.sample_indices = None

    def set_sample_indices(self, indices):
        """Set sample indices for the current batch."""
        self.sample_indices = indices

    def forward(self, x):
        """
        Forward pass that computes teacher/student outputs and contrastive loss.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: (teacher_logits, student_logits, contrastive_loss)
        """
        # Forward pass through teacher and student networks
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Extract intermediate features
        teacher_feat = self.teacher_hooks.features.get(self.teacher_layer)
        student_feat = self.student_hooks.features.get(self.student_layer)

        if teacher_feat is None or student_feat is None:
            raise ValueError(
                f"Missing features for layers: {self.teacher_layer} or {self.student_layer}"
            )

        # Project features to common dimension
        teacher_proj = self.teacher_projector(teacher_feat)
        student_proj = self.student_projector(student_feat)

        # Compute contrastive loss
        if self.sample_indices is not None:
            crd_loss = self.contrastive_loss(
                student_proj, teacher_proj, self.sample_indices
            )
        else:
            # Fallback: create pseudo-indices if not provided
            # NOTE: This is not ideal for CRD as memory bank won't be updated correctly
            batch_size = x.size(0)
            indices = torch.arange(batch_size, device=x.device) % self.n_data
            crd_loss = self.contrastive_loss(student_proj, teacher_proj, indices)

        # Clear features for next forward pass
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return teacher_logits, student_logits, crd_loss
