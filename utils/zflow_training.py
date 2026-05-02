import csv
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from distill.flow_model import (
    PairConditionedVelocity,
    build_layer_pairs,
    compute_flow_losses_for_pairs,
    euler_integrate,
    sample_layer_pairs,
)
from distill.zero_shot_replacement import ResNetSegmentRunner
from distill.zspace import LayerConditionedZSpace
from utils.utils import accuracy, get_module


class MultiLayerHooks:
    def __init__(self, named_layers: Iterable[Tuple[str, torch.nn.Module]]):
        self.features = OrderedDict()
        self.hooks = []

        def hook_fn(name):
            def _hook(module, _input, output):
                self.features[name] = output

            return _hook

        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class CSVLogger:
    def __init__(self, path: Path, headers: Sequence[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = list(headers)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def write(self, row: Dict):
        row_out = {k: row.get(k, "") for k in self.headers}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row_out)


class ZFlowRunner:
    """Stage-wise training/eval runner for ZFlow."""

    def __init__(self, args, teacher, train_loader, val_loader, device):
        self.args = args
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.layer_names = list(args.z_layers)
        self.layer_to_idx = {name: i for i, name in enumerate(self.layer_names)}

        named_layers = [
            (layer_name, get_module(self.teacher.model, layer_name))
            for layer_name in self.layer_names
        ]
        self.teacher_hooks = MultiLayerHooks(named_layers)

        self.zspace = LayerConditionedZSpace(
            layer_names=self.layer_names,
            z_dim=args.z_dim,
            z_tokens=args.z_tokens,
            z_tokens_mode=args.z_tokens_mode,
            use_token_norm=args.z_use_norm,
            mlp_expansion=args.z_mlp_expansion,
        ).to(device)

        self.flow = PairConditionedVelocity(
            z_dim=args.z_dim,
            num_layers=len(self.layer_names),
            hidden_dim=args.zflow_hidden_dim,
            num_blocks=args.zflow_num_blocks,
        ).to(device)

        self.rand = random.Random(args.seed)

        self.save_root = Path(args.save_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.zflow_dir = self.save_root / "zflow"
        self.zflow_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.teacher_hooks.remove()

    def run(self):
        stage = self.args.zflow_stage

        try:
            if stage == "zspace":
                self.train_zspace_stage()
                return

            if stage == "flow":
                if not self.args.zspace_ckpt:
                    raise ValueError("--zspace_ckpt is required for --zflow_stage flow")
                self.train_flow_stage(Path(self.args.zspace_ckpt))
                return

            if stage == "eval":
                if not self.args.zspace_ckpt or not self.args.flow_ckpt:
                    raise ValueError(
                        "--zspace_ckpt and --flow_ckpt are required for --zflow_stage eval"
                    )
                self.evaluate_zero_shot(Path(self.args.zspace_ckpt), Path(self.args.flow_ckpt))
                return

            if stage == "all":
                zspace_ckpt = self.train_zspace_stage()
                flow_ckpt = self.train_flow_stage(zspace_ckpt)
                self.evaluate_zero_shot(zspace_ckpt, flow_ckpt)
                return

            raise ValueError(
                f"Unknown zflow_stage '{stage}'. Use zspace|flow|eval|all"
            )
        finally:
            self.close()

    def _extract_inputs_targets(self, batch):
        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def _teacher_forward(self, inputs: torch.Tensor):
        self.teacher_hooks.clear()
        logits = self.teacher(inputs)
        features = OrderedDict(
            (name, self.teacher_hooks.features[name].detach()) for name in self.layer_names
        )
        self.teacher_hooks.clear()
        return logits.detach(), features

    def _warmup_zspace(self, loader):
        batch = next(iter(loader))
        inputs, _ = self._extract_inputs_targets(batch)
        with torch.no_grad():
            _, features = self._teacher_forward(inputs)
        self.zspace.ensure_from_features(features)

    def _weighted_update(self, meter_sums: Dict[str, float], metrics: Dict[str, float], weight: int):
        for k, v in metrics.items():
            meter_sums[k] += float(v) * weight

    def _finalize_weighted(self, meter_sums: Dict[str, float], total_weight: int):
        if total_weight <= 0:
            return {k: 0.0 for k in meter_sums}
        return {k: v / total_weight for k, v in meter_sums.items()}

    def _save_zspace_checkpoint(self, path: Path, epoch: int, metric: float, optimizer):
        payload = {
            "epoch": epoch,
            "metric": metric,
            "layer_names": self.layer_names,
            "zspace_state_dict": self.zspace.state_dict(),
            "layer_shape_templates": self.zspace.layer_shape_templates,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(self.args),
        }
        torch.save(payload, path)

    def _load_zspace_checkpoint(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "layer_names" in ckpt and list(ckpt["layer_names"]) != self.layer_names:
            raise ValueError(
                f"Layer mismatch in zspace ckpt. ckpt={ckpt['layer_names']}, current={self.layer_names}"
            )

        self._warmup_zspace(self.train_loader)
        self.zspace.load_state_dict(ckpt["zspace_state_dict"], strict=False)
        if "layer_shape_templates" in ckpt:
            self.zspace.layer_shape_templates.update(ckpt["layer_shape_templates"])

    def _save_flow_checkpoint(self, path: Path, epoch: int, metric: float, optimizer):
        payload = {
            "epoch": epoch,
            "metric": metric,
            "layer_names": self.layer_names,
            "flow_state_dict": self.flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(self.args),
        }
        torch.save(payload, path)

    def _load_flow_checkpoint(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "layer_names" in ckpt and list(ckpt["layer_names"]) != self.layer_names:
            raise ValueError(
                f"Layer mismatch in flow ckpt. ckpt={ckpt['layer_names']}, current={self.layer_names}"
            )
        self.flow.load_state_dict(ckpt["flow_state_dict"], strict=False)

    def train_zspace_stage(self) -> Path:
        print("\n=== ZFlow Stage 1: Z-space Learning ===")
        self.zspace.train()
        self.flow.eval()

        self._warmup_zspace(self.train_loader)

        optimizer = torch.optim.Adam(
            self.zspace.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.step_size, gamma=self.args.lr_decay
        )

        transition_headers = []
        for a, b in zip(self.layer_names[:-1], self.layer_names[1:]):
            transition_headers.append(f"trans_dist_{a}_to_{b}")
            transition_headers.append(f"trans_angle_{a}_to_{b}")

        stage1_headers = [
            "epoch",
            "phase",
            "total",
            "rec",
            "dist",
            "angle",
            "trans_dist",
            "trans_angle",
            "top1",
            "lr",
        ]
        for layer in self.layer_names:
            stage1_headers.extend([f"rec_{layer}", f"dist_{layer}", f"angle_{layer}"])
        stage1_headers.extend(transition_headers)

        logger = CSVLogger(self.zflow_dir / "zspace_log.csv", headers=stage1_headers)

        best_val = float("inf")
        best_ckpt = self.zflow_dir / "zspace_best.pth"
        last_ckpt = self.zflow_dir / "zspace_last.pth"

        for epoch in range(self.args.epochs):
            start_t = time.time()
            train_metrics = self._run_zspace_epoch(
                loader=self.train_loader,
                optimizer=optimizer,
                train=True,
            )
            val_metrics = self._run_zspace_epoch(
                loader=self.val_loader,
                optimizer=None,
                train=False,
            )

            lr = optimizer.param_groups[0]["lr"]
            train_row = {"epoch": epoch, "phase": "train", "lr": lr, **train_metrics}
            val_row = {"epoch": epoch, "phase": "val", "lr": lr, **val_metrics}
            logger.write(train_row)
            logger.write(val_row)

            scheduler.step()
            elapsed = time.time() - start_t

            print(
                f"Epoch {epoch}: Z train total={train_metrics.get('total', 0.0):.4f}, "
                f"val total={val_metrics.get('total', 0.0):.4f}, "
                f"val rec={val_metrics.get('rec', 0.0):.4f}, "
                f"val trans={val_metrics.get('trans_dist', 0.0):.4f}/{val_metrics.get('trans_angle', 0.0):.4f}, "
                f"time={elapsed:.1f}s"
            )

            self._save_zspace_checkpoint(last_ckpt, epoch, val_metrics["total"], optimizer)
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                self._save_zspace_checkpoint(best_ckpt, epoch, best_val, optimizer)
                print(f"[ZFlow] New best zspace checkpoint: {best_ckpt}")

        print(f"[ZFlow] Stage 1 complete. Best ckpt: {best_ckpt}")
        return best_ckpt

    def _run_zspace_epoch(self, loader, optimizer, train: bool):
        if train:
            self.zspace.train()
        else:
            self.zspace.eval()

        metric_sums = defaultdict(float)
        total_samples = 0

        desc = "ZSpace Train" if train else "ZSpace Val"
        pbar = tqdm(loader, desc=desc, leave=False)

        for batch_idx, batch in enumerate(pbar):
            inputs, targets = self._extract_inputs_targets(batch)
            batch_size = inputs.size(0)

            with torch.no_grad():
                teacher_logits, features = self._teacher_forward(inputs)

            if train:
                optimizer.zero_grad()

            total_loss, metrics, _ = self.zspace.compute_z_losses(
                features,
                lambda_rec=self.args.lambda_rec,
                lambda_dist=self.args.lambda_dist,
                lambda_angle=self.args.lambda_angle,
                lambda_trans_dist=self.args.lambda_trans_dist,
                lambda_trans_angle=self.args.lambda_trans_angle,
            )

            if train:
                total_loss.backward()
                optimizer.step()

            top1 = accuracy(teacher_logits, targets, topk=(1,))[0].item()
            metrics = dict(metrics)
            metrics["top1"] = top1

            self._weighted_update(metric_sums, metrics, batch_size)
            total_samples += batch_size

            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                pbar.set_postfix(
                    {
                        "total": f"{(metric_sums['total']/max(total_samples,1)):.4f}",
                        "rec": f"{(metric_sums['rec']/max(total_samples,1)):.4f}",
                    }
                )

        return self._finalize_weighted(metric_sums, total_samples)

    def train_flow_stage(self, zspace_ckpt: Path) -> Path:
        print("\n=== ZFlow Stage 2: Flow Learning ===")

        self._load_zspace_checkpoint(zspace_ckpt)
        self.zspace.eval()
        for p in self.zspace.parameters():
            p.requires_grad = False

        self.flow.train()

        optimizer = torch.optim.Adam(
            self.flow.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.step_size, gamma=self.args.lr_decay
        )

        stage2_headers = [
            "epoch",
            "phase",
            "total",
            "fm",
            "path",
            "end",
            "pairs_used",
            "top1",
            "lr",
        ]
        logger = CSVLogger(self.zflow_dir / "flow_log.csv", headers=stage2_headers)

        best_val = float("inf")
        best_ckpt = self.zflow_dir / "flow_best.pth"
        last_ckpt = self.zflow_dir / "flow_last.pth"

        for epoch in range(self.args.epochs):
            start_t = time.time()
            train_metrics = self._run_flow_epoch(
                loader=self.train_loader,
                optimizer=optimizer,
                train=True,
            )
            val_metrics = self._run_flow_epoch(
                loader=self.val_loader,
                optimizer=None,
                train=False,
            )

            lr = optimizer.param_groups[0]["lr"]
            train_row = {"epoch": epoch, "phase": "train", "lr": lr, **train_metrics}
            val_row = {"epoch": epoch, "phase": "val", "lr": lr, **val_metrics}
            logger.write(train_row)
            logger.write(val_row)

            scheduler.step()
            elapsed = time.time() - start_t

            print(
                f"Epoch {epoch}: Flow train total={train_metrics.get('total', 0.0):.4f}, "
                f"val total={val_metrics.get('total', 0.0):.4f}, "
                f"val fm/path/end={val_metrics.get('fm', 0.0):.4f}/"
                f"{val_metrics.get('path', 0.0):.4f}/{val_metrics.get('end', 0.0):.4f}, "
                f"time={elapsed:.1f}s"
            )

            self._save_flow_checkpoint(last_ckpt, epoch, val_metrics["total"], optimizer)
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                self._save_flow_checkpoint(best_ckpt, epoch, best_val, optimizer)
                print(f"[ZFlow] New best flow checkpoint: {best_ckpt}")

        print(f"[ZFlow] Stage 2 complete. Best ckpt: {best_ckpt}")
        return best_ckpt

    def _run_flow_epoch(self, loader, optimizer, train: bool):
        if train:
            self.flow.train()
        else:
            self.flow.eval()

        metric_sums = defaultdict(float)
        total_samples = 0

        desc = "Flow Train" if train else "Flow Val"
        pbar = tqdm(loader, desc=desc, leave=False)

        for batch_idx, batch in enumerate(pbar):
            inputs, targets = self._extract_inputs_targets(batch)
            batch_size = inputs.size(0)

            with torch.no_grad():
                teacher_logits, features = self._teacher_forward(inputs)
                z_by_layer = self.zspace.encode_feature_dict(features)

            pairs = sample_layer_pairs(
                layer_names=self.layer_names,
                mode=self.args.zflow_pair_mode,
                num_pairs=self.args.zflow_num_pairs,
                rng=self.rand,
            )

            if train:
                optimizer.zero_grad()

            total_loss, metrics = compute_flow_losses_for_pairs(
                velocity_model=self.flow,
                z_by_layer=z_by_layer,
                layer_names=self.layer_names,
                pairs=pairs,
                steps=self.args.zflow_steps,
                lambda_fm=self.args.lambda_fm,
                lambda_path=self.args.lambda_path,
                lambda_end=self.args.lambda_end,
            )

            if train:
                total_loss.backward()
                optimizer.step()

            top1 = accuracy(teacher_logits, targets, topk=(1,))[0].item()
            metrics = dict(metrics)
            metrics["top1"] = top1
            metrics["pairs_used"] = float(len(pairs))

            self._weighted_update(metric_sums, metrics, batch_size)
            total_samples += batch_size

            if batch_idx % max(1, self.args.print_freq // 10) == 0:
                pbar.set_postfix(
                    {
                        "total": f"{(metric_sums['total']/max(total_samples,1)):.4f}",
                        "fm": f"{(metric_sums['fm']/max(total_samples,1)):.4f}",
                        "path": f"{(metric_sums['path']/max(total_samples,1)):.4f}",
                    }
                )

        return self._finalize_weighted(metric_sums, total_samples)

    def _kl_to_teacher(self, pred_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.args.tau
        return F.kl_div(
            F.log_softmax(pred_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

    def evaluate_zero_shot(self, zspace_ckpt: Path, flow_ckpt: Path):
        print("\n=== ZFlow Stage 3: Zero-Shot Replacement Evaluation ===")

        self._load_zspace_checkpoint(zspace_ckpt)
        self._load_flow_checkpoint(flow_ckpt)

        self.zspace.eval()
        self.flow.eval()
        for p in self.zspace.parameters():
            p.requires_grad = False
        for p in self.flow.parameters():
            p.requires_grad = False

        runner = ResNetSegmentRunner(self.teacher)

        pair_indices = build_layer_pairs(self.layer_names, self.args.zflow_pair_mode)
        if not pair_indices:
            raise ValueError("No valid layer pairs for evaluation")

        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        zspace_params = sum(p.numel() for p in self.zspace.parameters())
        flow_params = sum(p.numel() for p in self.flow.parameters())
        params_by_baseline = {
            "full_teacher": teacher_params,
            "skip_block": teacher_params,
            "z_checkpoint": teacher_params + zspace_params,
            "linear_z_oracle": teacher_params + zspace_params,
            "flow_replacement": teacher_params + zspace_params + flow_params,
        }

        eval_headers = [
            "baseline",
            "pair",
            "k",
            "available",
            "num_params",
            "accuracy",
            "kl_to_teacher",
            "endpoint_latent_error",
            "decoded_hidden_error",
            "latency_ms",
        ]
        logger = CSVLogger(self.zflow_dir / "eval_metrics.csv", headers=eval_headers)

        stats = defaultdict(lambda: defaultdict(float))

        def add_stat(key, metric_dict, batch_size, available=True):
            stats[key]["seen"] += 1.0
            if available:
                stats[key]["available"] += 1.0
                stats[key]["count"] += float(batch_size)
                for mk, mv in metric_dict.items():
                    stats[key][mk] += float(mv) * float(batch_size)
            else:
                stats[key]["count"] += 0.0

        pbar = tqdm(self.val_loader, desc="ZFlow Eval", leave=False)
        with torch.no_grad():
            for batch in pbar:
                inputs, targets = self._extract_inputs_targets(batch)
                bsz = inputs.size(0)

                teacher_logits, features = self._teacher_forward(inputs)
                z_by_layer = self.zspace.encode_feature_dict(features)

                # 1) Full teacher baseline.
                t0 = time.perf_counter()
                logits_full = teacher_logits
                latency_ms = (time.perf_counter() - t0) * 1000.0
                acc_full = accuracy(logits_full, targets, topk=(1,))[0].item()
                add_stat(
                    ("full_teacher", "-", 0),
                    {
                        "accuracy": acc_full,
                        "kl_to_teacher": 0.0,
                        "endpoint_latent_error": 0.0,
                        "decoded_hidden_error": 0.0,
                        "latency_ms": latency_ms,
                    },
                    bsz,
                    available=True,
                )

                for start_idx, end_idx in pair_indices:
                    start_layer = self.layer_names[start_idx]
                    end_layer = self.layer_names[end_idx]
                    pair_name = f"{start_layer}->{end_layer}"

                    h_a = features[start_layer]
                    h_b = features[end_layer]
                    z_a = z_by_layer[start_layer]
                    z_b = z_by_layer[end_layer]

                    # 2) skip baseline if shape-compatible.
                    key_skip = ("skip_block", pair_name, 0)
                    if h_a.shape == h_b.shape:
                        t0 = time.perf_counter()
                        logits_skip = runner.forward_from_layer_output(end_layer, h_a)
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        acc = accuracy(logits_skip, targets, topk=(1,))[0].item()
                        kl = self._kl_to_teacher(logits_skip, teacher_logits).item()
                        add_stat(
                            key_skip,
                            {
                                "accuracy": acc,
                                "kl_to_teacher": kl,
                                "endpoint_latent_error": F.mse_loss(z_a, z_b).item(),
                                "decoded_hidden_error": F.mse_loss(h_a, h_b).item(),
                                "latency_ms": latency_ms,
                            },
                            bsz,
                            available=True,
                        )
                    else:
                        add_stat(key_skip, {}, bsz, available=False)

                    # 3) Z checkpoint replacement without flow: E_a -> D_b.
                    t0 = time.perf_counter()
                    h_hat_direct = self.zspace.decode(
                        z_a,
                        end_layer,
                        self.zspace.layer_shape_templates[end_layer],
                    )
                    logits_direct = runner.forward_from_layer_output(end_layer, h_hat_direct)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    acc = accuracy(logits_direct, targets, topk=(1,))[0].item()
                    kl = self._kl_to_teacher(logits_direct, teacher_logits).item()
                    add_stat(
                        ("z_checkpoint", pair_name, 0),
                        {
                            "accuracy": acc,
                            "kl_to_teacher": kl,
                            "endpoint_latent_error": F.mse_loss(z_a, z_b).item(),
                            "decoded_hidden_error": F.mse_loss(h_hat_direct, h_b).item(),
                            "latency_ms": latency_ms,
                        },
                        bsz,
                        available=True,
                    )

                    # 4) Linear interpolation diagnostic (oracle z_b).
                    if self.args.zflow_diagnostic:
                        t0 = time.perf_counter()
                        h_hat_lin = self.zspace.decode(
                            z_b,
                            end_layer,
                            self.zspace.layer_shape_templates[end_layer],
                        )
                        logits_lin = runner.forward_from_layer_output(end_layer, h_hat_lin)
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                        acc = accuracy(logits_lin, targets, topk=(1,))[0].item()
                        kl = self._kl_to_teacher(logits_lin, teacher_logits).item()
                        add_stat(
                            ("linear_z_oracle", pair_name, 0),
                            {
                                "accuracy": acc,
                                "kl_to_teacher": kl,
                                "endpoint_latent_error": 0.0,
                                "decoded_hidden_error": F.mse_loss(h_hat_lin, h_b).item(),
                                "latency_ms": latency_ms,
                            },
                            bsz,
                            available=True,
                        )

                    # 5) Flow replacement baseline with K sweep.
                    for k in self.args.zflow_eval_steps:
                        t0 = time.perf_counter()
                        z_hat = euler_integrate(
                            velocity_model=self.flow,
                            z0=z_a,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            steps=k,
                        )
                        h_hat = self.zspace.decode(
                            z_hat,
                            end_layer,
                            self.zspace.layer_shape_templates[end_layer],
                        )
                        logits_flow = runner.forward_from_layer_output(end_layer, h_hat)
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                        acc = accuracy(logits_flow, targets, topk=(1,))[0].item()
                        kl = self._kl_to_teacher(logits_flow, teacher_logits).item()

                        add_stat(
                            ("flow_replacement", pair_name, int(k)),
                            {
                                "accuracy": acc,
                                "kl_to_teacher": kl,
                                "endpoint_latent_error": F.mse_loss(z_hat, z_b).item(),
                                "decoded_hidden_error": F.mse_loss(h_hat, h_b).item(),
                                "latency_ms": latency_ms,
                            },
                            bsz,
                            available=True,
                        )

        rows = []
        for key, agg in stats.items():
            baseline, pair_name, k = key
            seen = max(1.0, agg.get("seen", 1.0))
            avail = agg.get("available", 0.0)
            count = agg.get("count", 0.0)
            available_flag = 1 if avail > 0 else 0

            if count > 0:
                row = {
                    "baseline": baseline,
                    "pair": pair_name,
                    "k": int(k),
                    "available": available_flag,
                    "num_params": params_by_baseline.get(baseline, 0),
                    "accuracy": agg.get("accuracy", 0.0) / count,
                    "kl_to_teacher": agg.get("kl_to_teacher", 0.0) / count,
                    "endpoint_latent_error": agg.get("endpoint_latent_error", 0.0)
                    / count,
                    "decoded_hidden_error": agg.get("decoded_hidden_error", 0.0)
                    / count,
                    "latency_ms": agg.get("latency_ms", 0.0) / count,
                }
            else:
                row = {
                    "baseline": baseline,
                    "pair": pair_name,
                    "k": int(k),
                    "available": available_flag,
                    "num_params": params_by_baseline.get(baseline, 0),
                    "accuracy": "",
                    "kl_to_teacher": "",
                    "endpoint_latent_error": "",
                    "decoded_hidden_error": "",
                    "latency_ms": "",
                }

            rows.append(row)
            logger.write(row)

        print(f"[ZFlow] Evaluation metrics saved to {self.zflow_dir / 'eval_metrics.csv'}")
        # Print a compact summary for quick visibility.
        for row in rows[:10]:
            print(
                f"{row['baseline']} | {row['pair']} | K={row['k']} | "
                f"acc={row['accuracy']} | kl={row['kl_to_teacher']}"
            )
