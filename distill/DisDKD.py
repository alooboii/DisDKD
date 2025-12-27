"""
DisDKD: Discriminator-guided Decoupled Knowledge Distillation

Training paradigm:
    Phase 1 (Adversarial - first N epochs, e.g., 10):
        Interleaved discriminator and generator training within each batch

        for batch in loader:
            # Train discriminator for k steps (typically k=1-2)
            for _ in range(k):
                model.set_discriminator_mode()
                outputs = model(x, mode='discriminator')
                loss = outputs['disc_loss']
                loss.backward()
                optimizer_D.step()

            # Train generator for 1 step
            model.set_generator_mode()
            outputs = model(x, mode='generator')
            loss = outputs['gen_loss']
            loss.backward()
            optimizer_G.step()

    Phase 2 (DKD - remaining epochs):
        Standard DKD training on entire student

        model.set_phase(2)
        model.discard_adversarial_components()

        for batch in loader:
            outputs = model(x, targets)
            loss = outputs['dkd_loss']
            loss.backward()
            optimizer_DKD.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


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


class FeatureRegressor(nn.Module):
    """
    Light 1x1 convolutional regressor to project features to a common hidden dimension.

    Args:
        in_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels for common space
    """

    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """
    Lightweight discriminator to distinguish between teacher and student features.

    Args:
        hidden_channels (int): Number of channels in the hidden feature space
    """

    def __init__(self, hidden_channels):
        super(FeatureDiscriminator, self).__init__()

        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Simple discriminator network
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input features [batch_size, hidden_channels, H, W]

        Returns:
            Tensor: Discriminator output [batch_size, 1] (probability of being teacher)
        """
        # Global average pooling
        pooled = self.global_pool(x)  # [batch_size, hidden_channels, 1, 1]

        # Discriminate
        output = self.discriminator(pooled)  # [batch_size, 1]

        return output


class DisDKD(nn.Module):
    """
    Discriminator-enhanced Decoupled Knowledge Distillation.

    Combines DKD's decoupled logit-level distillation (TCKD + NCKD) with
    discriminator-based feature alignment from DisKD.

    Training occurs in two phases:
    1. Discriminator phase: Train discriminator and teacher projector to distinguish
       teacher (real=1) from student (fake=0) features
    2. Student phase: Train student and student projector with DKD loss + adversarial loss

    Args:
        teacher (nn.Module): Pretrained teacher network
        student (nn.Module): Student network
        teacher_layer (str): Name of teacher layer for feature extraction
        student_layer (str): Name of student layer for feature extraction (layer G)
        teacher_channels (int): Number of channels in teacher feature map
        student_channels (int): Number of channels in student feature map
        hidden_channels (int): Number of channels in the common hidden space
        alpha (float): Weight for TCKD loss (DKD component)
        beta (float): Weight for NCKD loss (DKD component)
        temperature (float): Temperature for DKD softmax
    """

    def __init__(
        self,
        teacher,
        student,
        teacher_layer,
        student_layer,
        teacher_channels,
        student_channels,
        hidden_channels=256,
        alpha=1.0,
        beta=8.0,
        temperature=4.0,
    ):
        super(DisDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.hidden_channels = hidden_channels
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

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

        # Feature regressors to project to common dimension
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        for p in self.teacher_regressor.parameters():
            p.requires_grad = False
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        # Feature discriminator
        self.discriminator = FeatureDiscriminator(hidden_channels)

        # BCE loss for discriminator
        self.bce_loss = nn.BCELoss()

        # Track training mode
        self.training_mode = "student"  # 'student' or 'discriminator'

        print(
            f"Teacher regressor has {count_params(self.teacher_regressor)*1e-6:.3f}M params..."
        )
        print(
            f"Student regressor has {count_params(self.student_regressor)*1e-6:.3f}M params..."
        )
        print(
            f"Discriminator has {count_params(self.discriminator)*1e-6:.3f}M params..."
        )
        print(
            f"DKD parameters: alpha={alpha}, beta={beta}, temperature={temperature}\n"
        )

    def set_training_mode(self, mode):
        """
        Set training mode: 'student' or 'discriminator'

        Args:
            mode (str): Training mode
        """
        assert mode in [
            "student",
            "discriminator",
        ], "Mode must be 'student' or 'discriminator'"
        self.training_mode = mode

        if mode == "discriminator":
            # Phase 1: Freeze student, unfreeze discriminator and teacher projector
            for param in self.student.parameters():
                param.requires_grad = False
            for param in self.student_regressor.parameters():
                param.requires_grad = False
            for param in self.discriminator.parameters():
                param.requires_grad = True
            for param in self.teacher_regressor.parameters():
                param.requires_grad = True
        else:  # student mode
            # Phase 2: Unfreeze student, freeze discriminator and teacher projector
            for param in self.student.parameters():
                param.requires_grad = True
            for param in self.student_regressor.parameters():
                param.requires_grad = True
            for param in self.discriminator.parameters():
                param.requires_grad = False
            for param in self.teacher_regressor.parameters():
                param.requires_grad = False

    # def match_spatial_dimensions(self, student_feat, teacher_feat):
    #     """
    #     Match spatial dimensions of student features to teacher features via interpolation.

    #     Args:
    #         student_feat (Tensor): Student features
    #         teacher_feat (Tensor): Teacher features

    #     Returns:
    #         Tensor: Student features with matched spatial dimensions
    #     """
    #     if student_feat.shape[2:] != teacher_feat.shape[2:]:
    #         student_feat = F.interpolate(
    #             student_feat,
    #             size=teacher_feat.shape[2:],
    #             mode="bilinear",
    #             align_corners=False,
    #         )
    #     return student_feat

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        """
        Compute the Decoupled Knowledge Distillation loss (TCKD + NCKD).

        Args:
            logits_student (Tensor): Student logits
            logits_teacher (Tensor): Teacher logits
            target (Tensor): Ground truth labels

        Returns:
            Tensor: Combined DKD loss
        """
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)

        # Compute softmax probabilities
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        # Target Class Knowledge Distillation (TCKD)
        pred_student_tckd = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher_tckd = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student_tckd = torch.log(pred_student_tckd)

        tckd_loss = (
            F.kl_div(log_pred_student_tckd, pred_teacher_tckd, reduction="batchmean")
            * (self.temperature**2)
        )

        # Non-Target Class Knowledge Distillation (NCKD)
        pred_teacher_nckd = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_nckd = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )

        nckd_loss = (
            F.kl_div(log_pred_student_nckd, pred_teacher_nckd, reduction="batchmean")
            * (self.temperature**2)
        )

        # Combined DKD loss
        return self.alpha * tckd_loss + self.beta * nckd_loss

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def forward(self, x, targets):
        """
        Forward pass computing outputs and losses based on training mode.

        Args:
            x (Tensor): Input tensor
            targets (Tensor): Ground truth labels

        Returns:
            dict: Dictionary containing logits and losses based on training mode
        """
        batch_size = x.size(0)

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

        # Project features to common hidden dimension
        teacher_hidden = self.teacher_regressor(teacher_feat)
        student_hidden = self.student_regressor(student_feat)

        # Match spatial dimensions
        # student_hidden = self.match_spatial_dimensions(student_hidden, teacher_hidden)

        result = {"teacher_logits": teacher_logits, "student_logits": student_logits}

        if self.training_mode == "discriminator":
            # Phase 1: Train discriminator to distinguish teacher (real=1) from student (fake=0)

            # Discriminator predictions
            teacher_pred = self.discriminator(teacher_hidden)
            student_pred = self.discriminator(student_hidden.detach())

            # Real/fake labels
            real_labels = torch.ones(batch_size, 1, device=x.device)
            fake_labels = torch.zeros(batch_size, 1, device=x.device)

            # Discriminator loss
            disc_loss_real = self.bce_loss(teacher_pred, real_labels)
            disc_loss_fake = self.bce_loss(student_pred, fake_labels)
            disc_loss = (disc_loss_real + disc_loss_fake) / 2

            # Calculate discriminator accuracy
            teacher_correct = (teacher_pred > 0.5).float()  # Should predict 1 (real)
            student_correct = (student_pred <= 0.5).float()  # Should predict 0 (fake)
            disc_accuracy = (
                (teacher_correct.sum() + student_correct.sum()) / (2 * batch_size)
            ).item()

            result["discriminator_loss"] = disc_loss.item()
            result["discriminator_accuracy"] = disc_accuracy
            result["total_disc_loss"] = disc_loss

        else:  # student mode
            # Phase 2: Train student with DKD loss + adversarial loss

            # Compute DKD loss (TCKD + NCKD)
            dkd_loss = self.compute_dkd_loss(student_logits, teacher_logits, targets)

            # Adversarial loss: student wants to fool discriminator (be classified as teacher)
            student_pred = self.discriminator(student_hidden)
            real_labels = torch.ones(batch_size, 1, device=x.device)
            adversarial_loss = self.bce_loss(student_pred, real_labels)

            # Calculate fool rate (how many times student fooled discriminator)
            fool_rate = (student_pred > 0.5).float().mean().item()

            result["dkd_loss"] = dkd_loss.item()
            result["adversarial_loss"] = adversarial_loss.item()
            result["fool_rate"] = fool_rate
            result["total_student_loss"] = adversarial_loss
            result["method_specific_loss"] = dkd_loss

        # Clear features for next forward pass
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return result

    def get_optimizers(self, student_lr=1e-3, discriminator_lr=1e-4, weight_decay=1e-4):
        """
        Get separate optimizers for student and discriminator.

        Args:
            student_lr (float): Learning rate for student
            discriminator_lr (float): Learning rate for discriminator
            weight_decay (float): Weight decay

        Returns:
            tuple: (student_optimizer, discriminator_optimizer)
        """
        student_params = list(self.student.parameters()) + list(
            self.student_regressor.parameters()
        )

        discriminator_params = list(self.discriminator.parameters()) + list(
            self.teacher_regressor.parameters()
        )

        student_optimizer = torch.optim.Adam(
            student_params, lr=student_lr, weight_decay=weight_decay
        )

        discriminator_optimizer = torch.optim.Adam(
            discriminator_params, lr=discriminator_lr, weight_decay=weight_decay
        )

        return student_optimizer, discriminator_optimizer
