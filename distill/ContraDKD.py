import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.utils import get_module, count_params


class FeatureHooks:
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


class FeatureRegressor(nn.Module):
    """1x1 Conv to project features to common dimension."""

    def __init__(self, in_channels, hidden_channels):
        super(FeatureRegressor, self).__init__()
        self.regressor = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        return self.regressor(x)


class FeatureDiscriminator(nn.Module):
    """Discriminator with Global Pooling."""

    def __init__(self, hidden_channels):
        super(FeatureDiscriminator, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
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
        pooled = self.global_pool(x)
        return self.discriminator(pooled)


class L2CContrastiveLoss(nn.Module):
    """
    Conditional Contrastive Loss (L2C).
    Encourages features to cluster around learnable class proxies.
    """

    def __init__(self, num_classes, feat_dim=256, temperature=0.07):
        super(L2CContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.class_embeddings)

    def forward(self, features, labels, network_features=None):
        """
        Compute L2C loss with both data-class and data-data contrastive terms.

        Args:
            features: Query features [B, Dim] (normalized)
            labels: Class labels [B]
            network_features: Optional reference features for data-data contrast
                             If None, uses features (self-contrastive)
        """
        if network_features is None:
            network_features = features

        # Normalize
        features = F.normalize(features, dim=1)
        network_features = F.normalize(network_features, dim=1)
        proxies = F.normalize(self.class_embeddings, dim=1)

        # 1. Positive Proxy Similarity: (x_i . c_yi)
        target_proxies = proxies[labels]
        proxy_sim = (features * target_proxies).sum(
            dim=1, keepdim=True
        ) / self.temperature

        # 2. Batch Similarity Matrix: (x_i . x_k)
        batch_sim = torch.mm(features, network_features.t()) / self.temperature

        # Masks
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        same_class_mask = (labels_col == labels_row).float().fill_diagonal_(0)
        diff_mask = torch.ones_like(same_class_mask).fill_diagonal_(0)

        # L2C Numerator: exp(proxy) + sum(exp(same_class))
        exp_sim = torch.exp(batch_sim)
        same_class_sum = (exp_sim * same_class_mask).sum(dim=1, keepdim=True)
        numerator = torch.exp(proxy_sim) + same_class_sum

        # L2C Denominator: exp(proxy) + sum(exp(all_others))
        all_sum = (exp_sim * diff_mask).sum(dim=1, keepdim=True)
        denominator = torch.exp(proxy_sim) + all_sum

        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()


class ContraDKD(nn.Module):
    """
    Unified ContraDKD with corrected training objectives.

    Discriminator Phase: Train Disc + Teacher Regressor (adversarial only)
    Student Phase: Train Student + Student Regressor (DKD + Adversarial + L2C)
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
        num_classes=10,
        alpha=1.0,
        beta=8.0,
        temperature=4.0,
        l2c_weight=0.5,
        adv_weight=0.1,
    ):
        super(ContraDKD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.l2c_weight = l2c_weight
        self.adv_weight = adv_weight

        # Freeze Teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Hooks
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
        self.teacher_hooks = FeatureHooks(
            [(teacher_layer, get_module(self.teacher.model, teacher_layer))]
        )
        self.student_hooks = FeatureHooks(
            [(student_layer, get_module(self.student.model, student_layer))]
        )

        # Projectors
        self.teacher_regressor = FeatureRegressor(teacher_channels, hidden_channels)
        self.student_regressor = FeatureRegressor(student_channels, hidden_channels)

        # Discriminator
        self.discriminator = FeatureDiscriminator(hidden_channels)

        # L2C Loss Module
        self.l2c_loss_mod = L2CContrastiveLoss(num_classes, hidden_channels)

        self.bce_loss = nn.BCELoss()
        self.training_mode = "student"

        print(
            f"ContraDKD Init: Hidden={hidden_channels}, L2C_W={l2c_weight}, Adv_W={adv_weight}"
        )
        print(
            f"Params: Disc={count_params(self.discriminator)}, L2C_Proxies={count_params(self.l2c_loss_mod)}"
        )

    def initialize_class_embeddings(self, dataloader, device="cuda"):
        """Initialize L2C class proxies using Teacher's features."""
        print("Initializing ContraDKD Proxies from Teacher...")
        self.teacher.eval()
        self.teacher_regressor.to(device)
        self.teacher_regressor.eval()

        class_feats = {i: [] for i in range(self.num_classes)}

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                _ = self.teacher(inputs)
                t_feat = self.teacher_hooks.features.get(self.teacher_layer)

                t_proj = self.teacher_regressor(t_feat)
                t_vec = F.adaptive_avg_pool2d(t_proj, 1).flatten(1)
                t_vec = F.normalize(t_vec, dim=1)

                for f, l in zip(t_vec, labels):
                    class_feats[l.item()].append(f.cpu())

                self.teacher_hooks.clear()
                if i >= 50:
                    break

        for c in range(self.num_classes):
            if len(class_feats[c]) > 0:
                center = torch.stack(class_feats[c]).mean(dim=0)
                self.l2c_loss_mod.class_embeddings.data[c] = center.to(device)

        print("Proxies initialized.")

    def set_training_mode(self, mode):
        self.training_mode = mode
        if mode == "discriminator":
            # CORRECTED: Discriminator trains with Teacher Regressor for adversarial game
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.student_regressor.parameters():
                p.requires_grad = False

            for p in self.discriminator.parameters():
                p.requires_grad = True
            for p in self.teacher_regressor.parameters():
                p.requires_grad = True

            # CORRECTED: Class embeddings only updated during student phase
            self.l2c_loss_mod.class_embeddings.requires_grad = False

        else:  # Student Mode
            for p in self.student.parameters():
                p.requires_grad = True
            for p in self.student_regressor.parameters():
                p.requires_grad = True

            for p in self.discriminator.parameters():
                p.requires_grad = False
            for p in self.teacher_regressor.parameters():
                p.requires_grad = False

            # CORRECTED: Allow class embeddings to refine during student training
            self.l2c_loss_mod.class_embeddings.requires_grad = True

    def forward(self, x, targets):
        batch_size = x.size(0)

        # Forward Pass
        with torch.no_grad():
            t_logits = self.teacher(x)
        s_logits = self.student(x)

        # Extract & Project Features
        t_feat = self.teacher_hooks.features.get(self.teacher_layer)
        s_feat = self.student_hooks.features.get(self.student_layer)

        if t_feat is None or s_feat is None:
            raise ValueError("Hooks failed to capture features")

        t_proj = self.teacher_regressor(t_feat)
        s_proj = self.student_regressor(s_feat)

        # Prepare Vectors (Pool -> Flatten -> Normalize)
        t_vec = F.normalize(F.adaptive_avg_pool2d(t_proj, 1).flatten(1), dim=1)
        s_vec = F.normalize(F.adaptive_avg_pool2d(s_proj, 1).flatten(1), dim=1)

        result = {"teacher_logits": t_logits, "student_logits": s_logits}

        if self.training_mode == "discriminator":
            # --- CORRECTED: PURE ADVERSARIAL TRAINING ---
            t_pred = self.discriminator(t_proj)
            s_pred = self.discriminator(s_proj.detach())

            # Instead of hard 0/1 labels, use smoothed versions
            real_lbl = torch.ones(batch_size, 1, device=x.device) * 0.9  # 0.9 instead of 1.0
            fake_lbl = torch.ones(batch_size, 1, device=x.device) * 0.1  # 0.1 instead of 0.0

            disc_loss = 0.5 * (
                self.bce_loss(t_pred, real_lbl) + self.bce_loss(s_pred, fake_lbl)
            )

            result["discriminator_loss"] = disc_loss.item()
            result["total_disc_loss"] = disc_loss  # No L2C here

            # Metrics
            acc = ((t_pred > 0.5).float().mean() + (s_pred <= 0.5).float().mean()) / 2
            result["discriminator_accuracy"] = acc.item()

        else:
            # --- CORRECTED: STUDENT TRAINING WITH FULL OBJECTIVE ---
            # A. DKD (Logit-level)
            dkd_loss = self.compute_dkd_loss(s_logits, t_logits, targets)

            # B. Adversarial (Fool Discriminator)
            s_pred = self.discriminator(s_proj)
            real_lbl = torch.ones(batch_size, 1, device=x.device)
            adv_loss = self.bce_loss(s_pred, real_lbl)

            # C. CORRECTED: L2C with Teacher Reference (data-data + data-class)
            # Student features should align to:
            # 1. Class prototypes (via class_embeddings)
            # 2. Teacher features (via network_features parameter)
            l2c_student = self.l2c_loss_mod(s_vec, targets, t_vec.detach())

            result["dkd"] = dkd_loss.item()
            result["adversarial"] = adv_loss.item()
            result["l2c_student"] = l2c_student.item()
            result["fool_rate"] = (s_pred > 0.5).float().mean().item()

            # Total Internal Loss
            total_internal = (
                dkd_loss + self.adv_weight * adv_loss + self.l2c_weight * l2c_student
            )
            result["total_student_loss"] = total_internal
            result["method_specific_loss"] = total_internal

        self.teacher_hooks.clear()
        self.student_hooks.clear()
        return result

    def compute_dkd_loss(self, logits_student, logits_teacher, target):
        """Standard DKD implementation."""
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student + 1e-8)
        tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean") * (
            self.temperature**2
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = F.kl_div(
            log_pred_student_part2, pred_teacher_part2, reduction="batchmean"
        ) * (self.temperature**2)
        return self.alpha * tckd_loss + self.beta * nckd_loss

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()

    def _cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def get_optimizers(self, student_lr=1e-3, discriminator_lr=1e-4, weight_decay=1e-4):
        """
        CORRECTED: Separate optimizers with proper parameter grouping.
        """
        # Student: Student Net + Student Regressor + Class Embeddings
        student_params = (
            list(self.student.parameters())
            + list(self.student_regressor.parameters())
            + list(self.l2c_loss_mod.parameters())  # CORRECTED: Added here
        )

        # Discriminator: Discriminator Net + Teacher Regressor
        discriminator_params = list(self.discriminator.parameters()) + list(
            self.teacher_regressor.parameters()
        )

        opt_s = torch.optim.Adam(
            student_params, lr=student_lr, weight_decay=weight_decay
        )
        opt_d = torch.optim.Adam(
            discriminator_params, lr=discriminator_lr, weight_decay=weight_decay
        )

        return opt_s, opt_d
