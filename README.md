# Knowledge Distillation Toolbox

Compact PyTorch training repo for experimenting with multiple KD methods, now including **FlowKD** (flow matching over teacher logits).

## Methods (`--method`)
- `Pretraining`: teacher pretraining (CE only).
- `HardCE`: student CE baseline (hard labels only).
- `LogitKD`: standard KD (CE + KL with temperature).
- `LogitMSE`: CE + MSE between student and teacher logits.
- `FlowKD`: CE + optional KL + flow-matching velocity loss over logits/probabilities.
- `ZFlow`: teacher-only 3-stage pipeline (Z-space learning, flow learning, zero-shot block replacement eval).
- Existing methods: `DKD`, `DisDKD`, `FitNet`, `CRD`, `ContraDKD`.

## FlowKD Objective
For input `x`:
- `z_t = teacher(x).detach()`
- `z_0 ~ base` where `base ∈ {zeros, gaussian}`
- `tau ~ Uniform(0,1)`
- `z_tau = (1 - tau) * z_0 + tau * z_t`
- `v_star`:
  - `logits`: `z_t - z_0`
  - `probabilities`: `softmax(z_t/T) - softmax(z_0/T)`
- `v_pred = v_phi(h_s(x), z_tau, tau_embed(tau))`

Loss:
`L = alpha * CE + (beta * KL if use_kl else 0) + lambda_fm * MSE(v_pred, v_star)`

## Key FlowKD Flags
- `--base_logits zeros|gaussian`
- `--lambda_fm <float>`
- `--use_kl true|false`
- `--flow_target logits|probabilities`
- `--temperature <float>` (alias of `--tau`)

## Metrics Reported
Per epoch CSV now logs:
- `accuracy` / `top1`
- `nll`
- `ece` (15-bin)
- `kl_to_teacher`
- `logit_mse`

Checkpoints are saved to `--save_dir` using method/teacher/student/dataset naming.

## Quick Start (CIFAR-10)
### 1) Hard-label CE baseline
```bash
python main.py \
  --method HardCE \
  --teacher resnet18 \
  --student resnet18 \
  --dataset CIFAR10 \
  --epochs 1 \
  --batch_size 128 \
  --save_dir ./checkpoints/hardce
```

### 2) Standard KD baseline
```bash
python main.py \
  --method LogitKD \
  --teacher resnet50 \
  --student resnet18 \
  --dataset CIFAR10 \
  --epochs 1 \
  --tau 4.0 \
  --alpha 1.0 \
  --beta 0.4 \
  --save_dir ./checkpoints/logitkd
```

### 3) Logit matching baseline
```bash
python main.py \
  --method LogitMSE \
  --teacher resnet50 \
  --student resnet18 \
  --dataset CIFAR10 \
  --epochs 1 \
  --alpha 1.0 \
  --gamma 1.0 \
  --save_dir ./checkpoints/logitmse
```

### 4) FlowKD
```bash
python main.py \
  --method FlowKD \
  --teacher resnet50 \
  --student resnet18 \
  --student_layer layer2 \
  --dataset CIFAR10 \
  --epochs 1 \
  --alpha 1.0 \
  --beta 0.4 \
  --lambda_fm 1.0 \
  --use_kl true \
  --base_logits zeros \
  --flow_target logits \
  --temperature 4.0 \
  --save_dir ./checkpoints/flowkd
```

## Ablation Examples
### Base logits
```bash
python main.py --method FlowKD --dataset CIFAR10 --base_logits zeros --save_dir ./checkpoints/flowkd_zeros
python main.py --method FlowKD --dataset CIFAR10 --base_logits gaussian --save_dir ./checkpoints/flowkd_gaussian
```

### KL on/off
```bash
python main.py --method FlowKD --dataset CIFAR10 --use_kl true --beta 0.4 --save_dir ./checkpoints/flowkd_kl
python main.py --method FlowKD --dataset CIFAR10 --use_kl false --save_dir ./checkpoints/flowkd_nokl
```

### Target space
```bash
python main.py --method FlowKD --dataset CIFAR10 --flow_target logits --save_dir ./checkpoints/flowkd_logits
python main.py --method FlowKD --dataset CIFAR10 --flow_target probabilities --temperature 4.0 --save_dir ./checkpoints/flowkd_probs
```

## Notes
- Use `python main.py --help` for full CLI options.
- Existing OOD/domain-based dataset support remains unchanged.

## ZFlow (Teacher-Only Stages)
Run with `--method ZFlow` and pick a stage:
- `--zflow_stage zspace`: learn layer-conditioned encoder/decoder in common Z-space.
- `--zflow_stage flow`: learn pair-conditioned velocity field in Z-space.
- `--zflow_stage eval`: evaluate zero-shot block replacement baselines.
- `--zflow_stage all`: run all three stages sequentially.

Example:
```bash
python main.py \
  --method ZFlow \
  --zflow_stage all \
  --teacher resnet50 \
  --dataset CIFAR10 \
  --z_layers layer1 layer2 layer3 layer4 \
  --z_dim 128 \
  --z_tokens_mode spatial \
  --z_tokens 4 \
  --lambda_rec 1.0 \
  --lambda_dist 1.0 \
  --lambda_angle 1.0 \
  --lambda_trans_dist 1.0 \
  --lambda_trans_angle 1.0 \
  --lambda_fm 1.0 \
  --lambda_path 1.0 \
  --lambda_end 1.0 \
  --zflow_pair_mode mixed \
  --zflow_num_pairs 4 \
  --zflow_steps 8 \
  --zflow_eval_steps 1 2 4 8 \
  --save_dir ./checkpoints/zflow_run
```

Artifacts:
- `.../zflow/zspace_best.pth`, `zspace_last.pth`
- `.../zflow/flow_best.pth`, `flow_last.pth`
- `.../zflow/zspace_log.csv`, `flow_log.csv`, `eval_metrics.csv`
