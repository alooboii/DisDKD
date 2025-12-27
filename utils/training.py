import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils.utils import accuracy, AverageMeter
from utils.model_factory import create_distillation_model, print_model_parameters
from utils.checkpoint import save_checkpoint


class Trainer:
    """
    Handles model training and validation.

    Supports two modes:
    1. Standard Mode (LogitKD, DKD, CRD, FitNet): Single phase training.
    2. Two-Phase Mode (DisDKD):
       - Phase 1: Interleaved Adversarial Training (Discriminator/Generator).
       - Phase 2: Decoupled Knowledge Distillation.
    """

    def __init__(
        self, teacher, student, num_classes, criterion, loss_tracker, device, args
    ):
        self.teacher = teacher
        self.student = student
        self.num_classes = num_classes
        self.criterion = criterion
        self.loss_tracker = loss_tracker
        self.device = device
        self.args = args

        # Create distillation model
        self.model = create_distillation_model(args, teacher, student, num_classes).to(
            device
        )
        print_model_parameters(self.model, args.method)

        # Phase tracking for DisDKD
        self.current_phase = 1
        # Use args if provided, else default to 50% of epochs
        self.phase1_epochs = (
            args.disdkd_phase1_epochs
            if hasattr(args, "disdkd_phase1_epochs")
            and args.disdkd_phase1_epochs is not None
            else args.epochs // 2
        )
        if args.method != "DisDKD":
            self.phase1_epochs = 0  # Standard methods skip Phase 1

        # Setup optimizers
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Setup optimizer(s) and scheduler(s)."""
        args = self.args

        if args.method == "DisDKD":
            # --- DisDKD Phase 1 Optimizers ---
            self.discriminator_optimizer = self.model.get_discriminator_optimizer(
                lr=args.discriminator_lr, weight_decay=args.weight_decay
            )
            self.generator_optimizer = self.model.get_generator_optimizer(
                lr=args.lr, weight_decay=args.weight_decay
            )
            # Phase 2 optimizer (DKD) is initialized later upon switch
            self.dkd_optimizer = None
            self.dkd_scheduler = None
        else:
            # --- Standard Method Optimizers ---
            self.student_optimizer, self.student_scheduler = self._create_optimizer(
                self.model
            )
            self.discriminator_optimizer = None

    def _create_optimizer(self, model):
        """Create optimizer and scheduler for a model."""
        args = self.args
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
        return optimizer, scheduler

    def _switch_to_phase2(self, epoch):
        """Transition DisDKD from Phase 1 (Adversarial) to Phase 2 (DKD)."""
        print(f"\n[Phase Switch] Epoch {epoch}: Transitioning to Phase 2 (DKD)...")

        self.current_phase = 2
        self.model.set_phase(2)
        self.model.discard_adversarial_components()

        # Free Phase 1 memory
        if hasattr(self, "discriminator_optimizer"):
            del self.discriminator_optimizer
            del self.generator_optimizer
            torch.cuda.empty_cache()

        # Initialize Phase 2 Optimizer (Standard DKD)
        self.dkd_optimizer = self.model.get_dkd_optimizer(
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.dkd_scheduler = StepLR(
            self.dkd_optimizer, step_size=self.args.step_size, gamma=self.args.lr_decay
        )
        print("[Phase Switch] Optimizers reset. Ready for DKD.")

    def train(self, train_loader, val_loader):
        """Main training loop handling both standard and phased training."""
        print(f"\nStarting training for {self.args.epochs} epochs...")
        best_acc = 0.0

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # --- DisDKD Logic ---
            if self.args.method == "DisDKD":
                # Check for Phase Switch
                if self.current_phase == 1 and epoch >= self.phase1_epochs:
                    self._switch_to_phase2(epoch)

                if self.current_phase == 1:
                    train_losses, train_acc = self._train_epoch_phase1(
                        train_loader, epoch
                    )
                else:
                    train_losses, train_acc = self._train_epoch_phase2(
                        train_loader, epoch
                    )
                    self.dkd_scheduler.step()

            # --- Standard Logic ---
            else:
                train_losses, train_acc = self._train_epoch_standard(
                    train_loader, epoch
                )
                self.student_scheduler.step()

            # Validation
            val_losses, val_acc = self._validate(val_loader, epoch)

            # Logging
            # Get current learning rate
            if self.args.method == "DisDKD":
                current_lr = (
                    self.dkd_optimizer.param_groups[0]["lr"]
                    if self.current_phase == 2
                    else self.generator_optimizer.param_groups[0]["lr"]
                )
            else:
                current_lr = self.student_optimizer.param_groups[0]["lr"]

            self.loss_tracker.log_epoch(
                epoch, "train", train_losses, train_acc, current_lr
            )
            self.loss_tracker.log_epoch(epoch, "val", val_losses, val_acc, current_lr)

            # Print epoch summary
            elapsed = time.time() - start_time

            # Phase-specific printing
            if self.args.method == "DisDKD" and self.current_phase == 1:
                # Phase 1: Show all adversarial metrics
                print(
                    f"Epoch {epoch} [Phase 1 - Adversarial]: "
                    f"D_Loss={train_losses['discriminator']:.4f}, "
                    f"G_Loss={train_losses['generator']:.4f}, "
                    f"Adv_Loss={train_losses['adversarial']:.4f}, "
                    f"D_Acc={train_losses['disc_acc']:.2%}, "
                    f"Fool={train_losses['fool_rate']:.2%}, "
                    f"Val_Acc={val_acc:.2f}%, "
                    f"Time={elapsed:.1f}s"
                )
            else:
                # Phase 2 or standard methods: Show all losses and accuracies
                loss_str = ", ".join(
                    [f"{k}={v:.4f}" for k, v in train_losses.items() if v > 0]
                )
                print(
                    f"Epoch {epoch}: "
                    f"{loss_str}, "
                    f"Train_Acc={train_acc:.2f}%, "
                    f"Val_Acc={val_acc:.2f}%, "
                    f"Time={elapsed:.1f}s"
                )

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc

                # Determine which optimizer to save
                if self.args.method == "DisDKD":
                    opt = (
                        self.dkd_optimizer
                        if self.current_phase == 2
                        else self.generator_optimizer
                    )
                else:
                    opt = self.student_optimizer

                save_checkpoint(
                    self.model, opt, epoch, val_acc, self.args, is_best=True
                )

            print("-" * 80)

        return best_acc

    def _train_epoch_phase1(self, train_loader, epoch):
        """DisDKD Phase 1: Interleaved Adversarial Training."""
        self.model.train()
        self.model.set_phase(1)
        meters = self._init_meters(phase=1)

        progress_bar = tqdm(train_loader, desc=f"Ep {epoch} (Ph1: Adv)", leave=False)

        for batch_idx, (inputs, _) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)

            # 1. Train Discriminator
            self.model.set_discriminator_mode()
            self.discriminator_optimizer.zero_grad()
            disc_out = self.model(inputs, mode="discriminator")
            disc_out["disc_loss"].backward()
            self.discriminator_optimizer.step()

            # Update discriminator metrics
            meters["discriminator"].update(disc_out["disc_loss"].item(), batch_size)
            meters["disc_acc"].update(disc_out["disc_accuracy"], batch_size)

            # 2. Train Generator
            self.model.set_generator_mode()
            self.generator_optimizer.zero_grad()
            gen_out = self.model(inputs, mode="generator")
            gen_out["gen_loss"].backward()
            self.generator_optimizer.step()

            # Update generator metrics
            meters["generator"].update(gen_out["gen_loss"].item(), batch_size)
            meters["adversarial"].update(gen_out["adv_loss"], batch_size)
            meters["fool_rate"].update(gen_out["fool_rate"], batch_size)

            # Live tqdm updates: D_Loss, D_Acc after discriminator step; G_Loss, Fool after generator step
            if batch_idx % self.args.print_freq == 0:
                progress_bar.set_postfix(
                    {
                        "D_Loss": f"{meters['discriminator'].avg:.4f}",
                        "D_Acc": f"{meters['disc_acc'].avg:.2%}",
                        "G_Loss": f"{meters['generator'].avg:.4f}",
                        "Fool": f"{meters['fool_rate'].avg:.2%}",
                    }
                )

        progress_bar.close()
        # Phase 1 doesn't track classification accuracy in training
        return self._get_average_losses(meters), 0.0

    def _train_epoch_phase2(self, train_loader, epoch):
        """DisDKD Phase 2: Standard DKD Training."""
        self.model.train()
        self.model.set_phase(2)
        meters = self._init_meters(phase=2)

        progress_bar = tqdm(train_loader, desc=f"Ep {epoch} (Ph2: DKD)", leave=False)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.dkd_optimizer.zero_grad()

            # Forward
            outputs = self.model(inputs, targets)

            # Loss: Alpha * CE + DKD (which internally = dkd_alpha * TCKD + dkd_beta * NCKD)
            ce_loss = self.criterion(outputs["student_logits"], targets)
            total_loss = self.args.alpha * ce_loss + outputs["dkd_loss"]

            total_loss.backward()
            self.dkd_optimizer.step()

            # Metrics
            acc = accuracy(outputs["student_logits"], targets)[0]
            meters["total"].update(total_loss.item(), inputs.size(0))
            meters["ce"].update(ce_loss.item(), inputs.size(0))
            meters["dkd"].update(outputs["dkd_loss"].item(), inputs.size(0))
            meters["accuracy"].update(acc.item(), inputs.size(0))

            # Live tqdm updates: Total loss and accuracy
            if batch_idx % self.args.print_freq == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{meters['total'].avg:.4f}",
                        "Acc": f"{meters['accuracy'].avg:.2f}%",
                    }
                )

        progress_bar.close()

        # Print all component-wise losses at epoch end
        print(
            f"  Phase 2 Components: Total={meters['total'].avg:.4f}, "
            f"CE={meters['ce'].avg:.4f}, DKD={meters['dkd'].avg:.4f}"
        )

        return self._get_average_losses(meters), meters["accuracy"].avg

    def _train_epoch_standard(self, train_loader, epoch):
        """Standard training for non-DisDKD methods."""
        self.model.train()
        meters = self._init_meters(phase=2)  # Use standard meters

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            if self.args.method == "CRD":
                inputs, targets, indices = batch_data
                self.model.set_sample_indices(indices)
            else:
                inputs, targets = batch_data

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.student_optimizer.zero_grad()

            # --- Forward & Loss Calculation ---
            if self.args.method == "DKD":
                # DKD returns (teacher_logits, student_logits, dkd_loss)
                teacher_logits, student_logits, dkd_loss = self.model(inputs, targets)
                ce_loss = self.criterion(student_logits, targets)
                total_loss = self.args.alpha * ce_loss + dkd_loss

                # Robust logging dictionary
                losses_dict = {
                    "total": total_loss.item(),
                    "ce": ce_loss.item(),
                    "dkd": dkd_loss.item(),
                    "kd": 0.0,  # Placeholder for LossTracker
                }

                # Add DKD components for logging
                tckd, nckd = self._compute_dkd_components(
                    student_logits, teacher_logits, targets
                )
                losses_dict["tckd"] = tckd
                losses_dict["nckd"] = nckd

            else:
                # Other methods (KD, FitNet, CRD)
                if self.args.method == "Pretraining":
                    student_logits = self.model(inputs)
                    teacher_logits = None
                    ce_loss = self.criterion(student_logits, targets)
                    total_loss = ce_loss
                    losses_dict = {
                        "total": total_loss.item(),
                        "ce": ce_loss.item(),
                        "kd": 0.0,
                    }
                else:
                    teacher_logits, student_logits, method_loss = self.model(inputs)
                    ce_loss = self.criterion(student_logits, targets)
                    kd_loss = self._compute_kd_loss(teacher_logits, student_logits)
                    total_loss = self.args.alpha * ce_loss + self.args.beta * kd_loss

                    losses_dict = {"ce": ce_loss.item(), "kd": kd_loss.item()}

                    if method_loss is not None:
                        total_loss += self.args.gamma * method_loss
                        if self.args.method == "CRD":
                            losses_dict["contrastive"] = method_loss.item()
                        elif self.args.method == "FitNet":
                            losses_dict["hint"] = method_loss.item()

                    losses_dict["total"] = total_loss.item()

            total_loss.backward()
            self.student_optimizer.step()

            # Update meters
            acc = accuracy(student_logits, targets)[0]
            meters["accuracy"].update(acc.item(), inputs.size(0))
            for k, v in losses_dict.items():
                if k in meters:
                    meters[k].update(v, inputs.size(0))

            # Live tqdm updates: Total loss and accuracy
            if batch_idx % self.args.print_freq == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{meters['total'].avg:.4f}",
                        "Acc": f"{meters['accuracy'].avg:.2f}%",
                    }
                )

        progress_bar.close()
        return self._get_average_losses(meters), meters["accuracy"].avg

    def _validate(self, val_loader, epoch=None):
        """Validate the model."""
        self.model.eval()
        meters = self._init_meters(phase=2)  # Standard validation uses standard meters

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # For validation, we ONLY care about the Student's classification ability.
                # Direct call to student submodule avoids Phase 1 logic that requires 'mode'.
                if self.args.method == "DisDKD" and self.current_phase == 2:
                    outputs = self.model.student(inputs)
                elif hasattr(self.model, "student"):
                    outputs = self.model.student(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                acc = accuracy(outputs, targets)[0]

                meters["total"].update(loss.item(), inputs.size(0))
                meters["accuracy"].update(acc.item(), inputs.size(0))

        return self._get_average_losses(meters), meters["accuracy"].avg

    def _compute_kd_loss(self, teacher_logits, student_logits):
        if teacher_logits is None:
            return torch.tensor(0.0, device=self.device)
        T = self.args.tau
        return nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
        ) * (T * T)

    def _compute_dkd_components(self, student_logits, teacher_logits, targets):
        """Calculates TCKD and NCKD just for logging purposes in standard DKD."""
        with torch.no_grad():
            # This logic mimics DKD.py to extract components for the log file
            # Re-implementing briefly here to keep Trainer self-contained for logging
            # (Logic copied from DisDKD/DKD for consistency)
            gt_mask = (
                torch.zeros_like(student_logits)
                .scatter_(1, targets.unsqueeze(1), 1)
                .bool()
            )
            other_mask = (
                torch.ones_like(student_logits)
                .scatter_(1, targets.unsqueeze(1), 0)
                .bool()
            )

            p_s = F.softmax(student_logits / self.args.tau, dim=1)
            p_t = F.softmax(teacher_logits / self.args.tau, dim=1)

            # TCKD Part
            p_s_tckd = torch.cat(
                [
                    (p_s * gt_mask).sum(1, keepdims=True),
                    (p_s * other_mask).sum(1, keepdims=True),
                ],
                dim=1,
            )
            p_t_tckd = torch.cat(
                [
                    (p_t * gt_mask).sum(1, keepdims=True),
                    (p_t * other_mask).sum(1, keepdims=True),
                ],
                dim=1,
            )
            tckd = F.kl_div(
                torch.log(p_s_tckd + 1e-8), p_t_tckd, reduction="batchmean"
            ) * (self.args.tau**2)

            # NCKD Part
            p_t_nckd = F.softmax(
                teacher_logits / self.args.tau - 1000.0 * gt_mask, dim=1
            )
            log_p_s_nckd = F.log_softmax(
                student_logits / self.args.tau - 1000.0 * gt_mask, dim=1
            )
            nckd = F.kl_div(log_p_s_nckd, p_t_nckd, reduction="batchmean") * (
                self.args.tau**2
            )

            return tckd.item(), nckd.item()

    def _init_meters(self, phase):
        if phase == 1:
            return {
                "discriminator": AverageMeter(),
                "generator": AverageMeter(),
                "adversarial": AverageMeter(),
                "disc_acc": AverageMeter(),
                "fool_rate": AverageMeter(),
            }
        else:
            # Standard metrics for Phase 2 and Baseline methods
            return {
                "total": AverageMeter(),
                "ce": AverageMeter(),
                "kd": AverageMeter(),
                "dkd": AverageMeter(),
                "tckd": AverageMeter(),
                "nckd": AverageMeter(),
                "hint": AverageMeter(),
                "contrastive": AverageMeter(),
                "accuracy": AverageMeter(),
            }

    def _get_average_losses(self, meters):
        return {k: v.avg for k, v in meters.items() if k != "accuracy"}
