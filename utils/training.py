import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from utils.model_factory import create_distillation_model
from utils.checkpoint import save_checkpoint


class Trainer:
	"""
	Unified trainer for DisDKD with correct GAN-style adversarial training.

	Phase 1:
		- Interleaved discriminator / generator updates
		- Proper BN handling (D.train(), D.eval())
		- Correct optimizer construction AFTER mode selection
		- MMD tracking for feature alignment

	Phase 2:
		- Pure DKD fine-tuning
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
		self.method = args.method

		if self.method != "Pretraining":
			self.distill_model = create_distillation_model(
				args, teacher, student, num_classes
			).to(device)
		else:
			raise NotImplementedError("Pretraining not supported here")

		self.best_val_acc = 0.0

	# ==========================================================
	# Main entry
	# ==========================================================

	def train(self, train_loader, val_loader):
		if self.method != "DisDKD":
			raise NotImplementedError

		return self._train_disdkd(train_loader, val_loader)

	# ==========================================================
	# DisDKD training
	# ==========================================================

	def _train_disdkd(self, train_loader, val_loader):
		phase1_epochs = self.args.disdkd_phase1_epochs
		total_epochs = self.args.epochs
		k_disc = getattr(self.args, "disdkd_k_disc_steps", 1)

		print(f"\n{'='*60}")
		print(f"PHASE 1: Adversarial Training (Interleaved D/G)")
		print(f"  Epochs: {phase1_epochs}")
		print(f"  Discriminator steps per batch: {k_disc}")
		print(f"  Generator steps per batch: 1")
		print(f"  Discriminator LR: {self.args.disdkd_phase1_lr}")
		print(f"  Generator LR: {self.args.disdkd_phase2_lr}")
		print(f"{'='*60}\n")

		# ------------------------------------------------------
		# Phase 1: Adversarial alignment
		# ------------------------------------------------------

		self.distill_model.set_phase(1)

		# IMPORTANT: build optimizers AFTER mode selection
		self.distill_model.set_discriminator_mode()
		self.optimizer_D = self.distill_model.get_discriminator_optimizer(
			lr=self.args.disdkd_phase1_lr,
			weight_decay=self.args.weight_decay,
		)

		self.distill_model.set_generator_mode()
		self.optimizer_G = self.distill_model.get_generator_optimizer(
			lr=self.args.disdkd_phase2_lr,
			weight_decay=self.args.weight_decay,
		)

		for epoch in range(1, phase1_epochs + 1):
			base = getattr(self.args, "disdkd_mmd_weight", 0.05)
			ramp = min(1.0, epoch / max(1, phase1_epochs))

			train_stats = self._train_epoch_phase1(train_loader, epoch, k_disc)
			val_stats = self._validate_epoch_phase1(val_loader, epoch)

			# Log to tracker
			self.loss_tracker.log_epoch(
				epoch=epoch,
				phase="train",
				losses={
					"disdkd_phase": 1,
					"disc_loss": train_stats["disc_loss"],
					"disc_acc": train_stats["disc_acc"],
					"gen_loss": train_stats["gen_loss"],
					"fool_rate": train_stats["fool_rate"],
					"mmd_d": train_stats["mmd_d"],
					"mmd_g": train_stats["mmd_g"],
					"total": train_stats["disc_loss"] + train_stats["gen_loss"],
				},
				accuracy=0.0,
			)

			self.loss_tracker.log_epoch(
				epoch=epoch,
				phase="val",
				losses={
					"disdkd_phase": 1,
					"disc_loss": val_stats["disc_loss"],
					"disc_acc": val_stats["disc_acc"],
					"gen_loss": val_stats["gen_loss"],
					"fool_rate": val_stats["fool_rate"],
					"mmd_d": val_stats["mmd_d"],
					"mmd_g": val_stats["mmd_g"],
					"total": val_stats["disc_loss"] + val_stats["gen_loss"],
				},
				accuracy=0.0,
			)

			print(f"\n{'='*70}")
			print(f"Epoch {epoch}/{phase1_epochs} [Phase 1] Summary")
			print(f"{'='*70}")
			print(f"mmd_weight = {base*ramp:.4f}")
			print(
				f"Train | D_loss: {train_stats['disc_loss']:.4f}, "
				f"D_acc: {train_stats['disc_acc']:.2f}%, "
				f"G_loss: {train_stats['gen_loss']:.4f}, "
				f"Fool: {train_stats['fool_rate']:.2f}%"
			)
			print(
				f"      | MMD_D: {train_stats['mmd_d']:.4f}, "
				f"MMD_G: {train_stats['mmd_g']:.4f}, "
				f"Δ: {abs(train_stats['mmd_d'] - train_stats['mmd_g']):.4f}"
			)
			print(
				f"Val   | D_loss: {val_stats['disc_loss']:.4f}, "
				f"D_acc: {val_stats['disc_acc']:.2f}%, "
				f"G_loss: {val_stats['gen_loss']:.4f}, "
				f"Fool: {val_stats['fool_rate']:.2f}%"
			)
			print(
				f"      | MMD_D: {val_stats['mmd_d']:.4f}, "
				f"MMD_G: {val_stats['mmd_g']:.4f}, "
				f"Δ: {abs(val_stats['mmd_d'] - val_stats['mmd_g']):.4f}"
			)
			print(f"{'='*70}\n")

		# ------------------------------------------------------
		# Phase 2: DKD
		# ------------------------------------------------------

		print(f"\n{'='*60}")
		print(f"PHASE 2: DKD Fine-tuning")
		print(f"  Epochs: {total_epochs - phase1_epochs}")
		print(f"  Learning rate: {self.args.disdkd_phase3_lr}")
		print(f"  Final Phase 1 MMD: {train_stats['mmd_g']:.4f} (lower = better alignment)")
		print(f"{'='*60}\n")

		self.distill_model.set_phase(2)
		self.distill_model.discard_adversarial_components()

		self.optimizer_DKD = self.distill_model.get_dkd_optimizer(
			lr=self.args.disdkd_phase3_lr,
			weight_decay=self.args.weight_decay,
		)

		scheduler = (
			StepLR(self.optimizer_DKD, self.args.step_size, self.args.lr_decay)
			if self.args.step_size > 0
			else None
		)

		for epoch in range(phase1_epochs + 1, total_epochs + 1):
			train_stats = self._train_epoch_phase2(train_loader, epoch)
			val_stats = self._validate_epoch_phase2(val_loader, epoch)

			# Log to tracker
			self.loss_tracker.log_epoch(
				epoch=epoch,
				phase="train",
				losses={
					"disdkd_phase": 2,
					"ce": train_stats["ce_loss"],
					"dkd": train_stats["dkd_loss"],
					"total": train_stats["total_loss"],
				},
				accuracy=train_stats["accuracy"],
			)

			self.loss_tracker.log_epoch(
				epoch=epoch,
				phase="val",
				losses={
					"disdkd_phase": 2,
					"ce": val_stats["ce_loss"],
					"dkd": val_stats["dkd_loss"],
					"total": val_stats["total_loss"],
				},
				accuracy=val_stats["accuracy"],
			)

			if scheduler:
				scheduler.step()

			if val_stats["accuracy"] > self.best_val_acc:
				self.best_val_acc = val_stats["accuracy"]
				self._save_checkpoint(epoch, is_best=True)
				print(f"  ⭐ New best val accuracy: {self.best_val_acc:.2f}%")

			print(f"\n{'='*60}")
			print(f"Epoch {epoch}/{total_epochs} [Phase 2] Summary")
			print(f"{'='*60}")
			print(
				f"Train | Loss: {train_stats['total_loss']:.4f}, "
				f"CE: {train_stats['ce_loss']:.4f}, "
				f"DKD: {train_stats['dkd_loss']:.4f}, "
				f"Acc: {train_stats['accuracy']:.2f}%"
			)
			print(
				f"Val   | Loss: {val_stats['total_loss']:.4f}, "
				f"CE: {val_stats['ce_loss']:.4f}, "
				f"DKD: {val_stats['dkd_loss']:.4f}, "
				f"Acc: {val_stats['accuracy']:.2f}%"
			)
			print(f"{'='*60}\n")

		print(f"\n{'='*60}")
		print(f"Training Complete!")
		print(f"  Best validation accuracy: {self.best_val_acc:.2f}%")
		print(f"{'='*60}\n")

		return self.best_val_acc

	# ==========================================================
	# Phase 1 — training (GAN equilibrium + MMD logging)
	# ==========================================================

	def _train_epoch_phase1(self, loader, epoch, k_disc):
		self.distill_model.train()

		disc_loss_sum = 0.0
		disc_acc_sum = 0.0
		gen_loss_sum = 0.0
		fool_rate_sum = 0.0
		mmd_d_sum = 0.0
		mmd_g_sum = 0.0
		num_batches = 0

		phase1_epochs = self.args.disdkd_phase1_epochs
		base = getattr(self.args, "disdkd_mmd_weight", 0.05)
		ramp = min(1.0, epoch / max(1, phase1_epochs))   # linear ramp
		self.distill_model.mmd_weight = base * ramp

		pbar = tqdm(loader, desc=f"Epoch {epoch} [Phase 1] Train", ncols=140, leave=False)

		for x, _ in pbar:
			x = x.to(self.device)

			# ------------------ Discriminator ------------------
			self.distill_model.set_discriminator_mode()
			self.distill_model.discriminator.train()
			self.distill_model.student.eval()

			batch_disc_loss = 0.0
			batch_disc_acc = 0.0
			batch_mmd_d = 0.0

			for _ in range(k_disc):
				self.optimizer_D.zero_grad()
				out = self.distill_model(x, mode="discriminator")
				out["disc_loss"].backward()
				self.optimizer_D.step()

				batch_disc_loss += out["disc_loss"].item()
				batch_disc_acc += out["disc_accuracy"]
				batch_mmd_d += out["mmd"]

			batch_disc_loss /= k_disc
			batch_disc_acc /= k_disc
			batch_mmd_d /= k_disc

			# ------------------ Generator ------------------
			self.distill_model.set_generator_mode()
			self.distill_model.discriminator.eval()  # freeze BN
			self.distill_model.student.train()

			self.optimizer_G.zero_grad()
			out_g = self.distill_model(x, mode="generator")
			out_g["gen_loss"].backward()
			self.optimizer_G.step()

			# ------------------ Accumulate ------------------
			disc_loss_sum += batch_disc_loss
			disc_acc_sum += batch_disc_acc
			gen_loss_sum += out_g["gen_loss"].item()
			fool_rate_sum += out_g["fool_rate"]
			mmd_d_sum += batch_mmd_d
			mmd_g_sum += out_g["mmd"]

			num_batches += 1

			pbar.set_postfix(
				{
					"D_loss": f"{disc_loss_sum/num_batches:.4f}",
					"D_acc": f"{100*disc_acc_sum/num_batches:.1f}%",
					"G_loss": f"{gen_loss_sum/num_batches:.4f}",
					"Fool": f"{100*fool_rate_sum/num_batches:.1f}%",
					"MMD_D": f"{mmd_d_sum/num_batches:.4f}",
					"MMD_G": f"{mmd_g_sum/num_batches:.4f}",
				}
			)

		pbar.close()

		return {
			"disc_loss": disc_loss_sum / num_batches,
			"disc_acc": 100 * disc_acc_sum / num_batches,
			"gen_loss": gen_loss_sum / num_batches,
			"fool_rate": 100 * fool_rate_sum / num_batches,
			"mmd_d": mmd_d_sum / num_batches,
			"mmd_g": mmd_g_sum / num_batches,
		}

	# ==========================================================
	# Phase 1 — validation (diagnostic only, read-only)
	# ==========================================================

	def _validate_epoch_phase1(self, loader, epoch):
		phase1_epochs = self.args.disdkd_phase1_epochs
		base = getattr(self.args, "disdkd_mmd_weight", 0.05)
		ramp = min(1.0, epoch / max(1, phase1_epochs))
		self.distill_model.mmd_weight = base * ramp
		self.distill_model.eval()
		self.distill_model.discriminator.eval()
		self.distill_model.student.eval()

		disc_loss_sum = 0.0
		disc_acc_sum = 0.0
		gen_loss_sum = 0.0
		fool_rate_sum = 0.0
		mmd_d_sum = 0.0
		mmd_g_sum = 0.0

		num_batches = 0

		pbar = tqdm(loader, desc=f"Epoch {epoch} [Phase 1] Val", ncols=140, leave=False)

		with torch.no_grad():
			for x, _ in pbar:
				x = x.to(self.device)

				d = self.distill_model(x, mode="discriminator")
				g = self.distill_model(x, mode="generator")

				disc_loss_sum += d["disc_loss"].item()
				disc_acc_sum += d["disc_accuracy"]
				gen_loss_sum += g["gen_loss"].item()
				fool_rate_sum += g["fool_rate"]
				mmd_d_sum += d["mmd"]
				mmd_g_sum += g["mmd"]

				num_batches += 1

				pbar.set_postfix(
					{
						"D_loss": f"{disc_loss_sum/num_batches:.4f}",
						"D_acc": f"{100*disc_acc_sum/num_batches:.1f}%",
						"G_loss": f"{gen_loss_sum/num_batches:.4f}",
						"Fool": f"{100*fool_rate_sum/num_batches:.1f}%",
						"MMD_D": f"{mmd_d_sum/num_batches:.4f}",
						"MMD_G": f"{mmd_g_sum/num_batches:.4f}",
					}
				)

		pbar.close()

		return {
			"disc_loss": disc_loss_sum / num_batches,
			"disc_acc": 100 * disc_acc_sum / num_batches,
			"gen_loss": gen_loss_sum / num_batches,
			"fool_rate": 100 * fool_rate_sum / num_batches,
			"mmd_d": mmd_d_sum / num_batches,
			"mmd_g": mmd_g_sum / num_batches,
		}

	# ==========================================================
	# Phase 2 — DKD
	# ==========================================================

	def _train_epoch_phase2(self, loader, epoch):
		self.distill_model.train()

		total_loss_sum = 0.0
		ce_loss_sum = 0.0
		dkd_loss_sum = 0.0
		correct = 0
		total = 0
		num_batches = 0

		pbar = tqdm(
			loader, desc=f"Epoch {epoch} [Phase 2] Train", ncols=120, leave=False
		)

		for x, y in pbar:
			x, y = x.to(self.device), y.to(self.device)
			self.optimizer_DKD.zero_grad()

			out = self.distill_model(x, targets=y)
			ce = self.criterion(out["student_logits"], y)
			loss = self.args.alpha * ce + self.args.beta * out["dkd_loss"]
			loss.backward()
			self.optimizer_DKD.step()

			total_loss_sum += loss.item()
			ce_loss_sum += ce.item()
			dkd_loss_sum += out["dkd_loss"].item()
			correct += out["student_logits"].argmax(1).eq(y).sum().item()
			total += y.size(0)
			num_batches += 1

			# Update progress bar
			pbar.set_postfix(
				{
					"loss": f"{total_loss_sum/num_batches:.4f}",
					"ce": f"{ce_loss_sum/num_batches:.4f}",
					"dkd": f"{dkd_loss_sum/num_batches:.4f}",
					"acc": f"{100*correct/total:.2f}%",
				}
			)

		pbar.close()

		return {
			"total_loss": total_loss_sum / num_batches,
			"ce_loss": ce_loss_sum / num_batches,
			"dkd_loss": dkd_loss_sum / num_batches,
			"accuracy": 100 * correct / total,
		}

	def _validate_epoch_phase2(self, loader, epoch):
		self.distill_model.eval()

		total_loss_sum = 0.0
		ce_loss_sum = 0.0
		dkd_loss_sum = 0.0
		correct = 0
		total = 0
		num_batches = 0

		pbar = tqdm(loader, desc=f"Epoch {epoch} [Phase 2] Val", ncols=120, leave=False)

		with torch.no_grad():
			for x, y in pbar:
				x, y = x.to(self.device), y.to(self.device)
				out = self.distill_model(x, targets=y)
				ce = self.criterion(out["student_logits"], y)
				loss = self.args.alpha * ce + self.args.beta * out["dkd_loss"]

				total_loss_sum += loss.item()
				ce_loss_sum += ce.item()
				dkd_loss_sum += out["dkd_loss"].item()
				correct += out["student_logits"].argmax(1).eq(y).sum().item()
				total += y.size(0)
				num_batches += 1

				# Update progress bar
				pbar.set_postfix(
					{
						"loss": f"{total_loss_sum/num_batches:.4f}",
						"ce": f"{ce_loss_sum/num_batches:.4f}",
						"dkd": f"{dkd_loss_sum/num_batches:.4f}",
						"acc": f"{100*correct/total:.2f}%",
					}
				)

		pbar.close()

		return {
			"total_loss": total_loss_sum / num_batches,
			"ce_loss": ce_loss_sum / num_batches,
			"dkd_loss": dkd_loss_sum / num_batches,
			"accuracy": 100 * correct / total,
		}

	# ==========================================================
	# Checkpointing
	# ==========================================================

	def _save_checkpoint(self, epoch, is_best=False):
		"""Save model checkpoint."""
		# Get the student model
		if hasattr(self.distill_model, "student"):
			model = self.distill_model.student
		else:
			model = self.student

		# Get the appropriate optimizer
		if hasattr(self, "optimizer_DKD") and self.optimizer_DKD is not None:
			optimizer = self.optimizer_DKD
		elif hasattr(self, "optimizer_G"):
			optimizer = self.optimizer_G
		elif hasattr(self, "optimizer"):
			optimizer = self.optimizer
		else:
			optimizer = None

		save_checkpoint(
			model=model,
			optimizer=optimizer,
			epoch=epoch,
			accuracy=self.best_val_acc,
			args=self.args,
			is_best=is_best,
		)