# Motion Prior Distillation (Dual Teacher)

Distill **two** frozen teachers — a Teleopit `TemporalCNN` motion-tracking
actor and an mjlab `Velocity-Rough-Unitree-G1` MLP locomotion actor — into
**one** student that, at deployment, consumes only proprioception
(`projected_gravity`, `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`,
`last_action`, each with `history_length=4`).

Two flavors of student are supported:

| Variant | Latent | Algorithm | Task ID |
|---|---|---|---|
| VAE | continuous (32-d) Gaussian | KL to `motion_prior` head | `Mjlab-MotionPrior-Flat-Unitree-G1` |
| VQ-VAE | discrete (2048 × 64-d codebook) | EMA codebook + commit loss | `Mjlab-MotionPrior-VQ-Flat-Unitree-G1` |

A separate `Mjlab-MotionPrior-Rough-Unitree-G1` task is registered for
single-teacher (teacher_b only) experiments on the rough env. The dual-env
distillation always uses the **flat** task ID as the primary; the runner
spins up the rough env internally — see "Launch" below.

---

## Architecture

```
flat env  ──► obs.student        (372)  ─┐
              obs.teacher_a      (166)  ─┤  encoder_a → z_a ──┐
              obs.teacher_a_history (10×166) ─► teacher_a ──► action_a (target)
                                                              │
rough env ──► obs.student        (372)  ─┐                    │
              obs.teacher_b      (286)  ─┤  encoder_b → z_b ──┤
                                              teacher_b ────► action_b (target)
                                                              │
                                                              ▼
                                          decoder([prop, z]) → student_action
                                          motion_prior(prop) → mp_z (deploy)
```

* **`student` schema is identical on both envs** so a single network covers both.
* **Teachers are frozen.** They produce the per-step regression target
  used by behavior loss only.
* **Shared `decoder`, `motion_prior`** force the two encoders to settle
  into a coherent latent distribution.
* **VAE** picks a continuous latent; the deploy path is
  `prop → motion_prior MLP → mp_mu Linear → decoder`.
* **VQ-VAE** picks the nearest codebook entry; the deploy path is
  `prop → motion_prior MLP → decoder`.

---

## Prerequisites

| Asset | Path | Note |
|---|---|---|
| teacher_a checkpoint | `~/zcy/Teleopit/track.pt` | Teleopit `TemporalCNN`, 27 tensors, 166-d obs + 10-frame history |
| teacher_b checkpoint | `~/zcy/mjlab-prior/logs/model_21000.pt` | mjlab velocity-rough actor, plain MLP, 286-d obs |
| motion file | `~/zcy/Teleopit/data/one_motion_for_debug.npz` (or your own `.npz` / W&B registry) | Drives the flat env's `MotionCommand` |

---

## Launch

The runner integrates with the standard mjlab training entry point. It
takes the **flat** task ID as primary; the rough env is built internally.

### Quick smoke (small scale, GPU)

```bash
uv run python -m mjlab.scripts.train \
  --task Mjlab-MotionPrior-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ~/zcy/Teleopit/data/one_motion_for_debug.npz \
  --env.scene.num-envs 64 \
  --agent.secondary-num-envs 64 \
  --agent.max-iterations 50 \
  --agent.num-steps-per-env 8 \
  --no-video
```

This launches dual-env rollout (64 + 64 envs), 50 iterations, ≈45 s on a
single RTX 4090. Loss prints every iter; checkpoint + ONNX land under
`logs/rsl_rl/g1_motion_prior/<timestamp>/`.

### Full training

```bash
uv run python -m mjlab.scripts.train \
  --task Mjlab-MotionPrior-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/your_motion.npz \
  --env.scene.num-envs 4096 \
  --agent.secondary-num-envs 4096 \
  --agent.max-iterations 100000
```

The two envs split GPU memory; if you OOM, lower both `num-envs`
together (typical: 2048 + 2048 on a 24 GB card).

### Override teacher checkpoints

Defaults are baked into `RslRlMotionPriorRunnerCfg` (paths above). To
point at different ckpts:

```bash
  --agent.teacher-a-policy-path /path/to/track.pt \
  --agent.teacher-b-policy-path /path/to/velocity_rough.pt
```

### Tune the loss

```bash
  --agent.algorithm.behavior-weight-a 1.0 \
  --agent.algorithm.behavior-weight-b 0.5 \
  --agent.algorithm.align-loss-coeff 0.01    # VAE only; off by default
  --agent.algorithm.kl-loss-coeff-max 0.01   # VAE annealing endpoints
  --agent.algorithm.kl-loss-coeff-min 0.001
```

### VQ variant

```bash
uv run python -m mjlab.scripts.train \
  --task Mjlab-MotionPrior-VQ-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/your_motion.npz \
  --env.scene.num-envs 4096 \
  --agent.secondary-num-envs 4096 \
  --agent.max-iterations 100000
```

Same env, different runner / algorithm / policy classes (selected by
`runner_cls=MotionPriorVQOnPolicyRunner` at task registration). Codebook
size is 2048 × 64 by default.

### Resume

```bash
  --agent.resume \
  --agent.load-run "<timestamp_or_regex>" \
  --agent.load-checkpoint "model_.*.pt"
```

The frozen teachers are **not** stored in the checkpoint — they reload
from the configured paths every time. Only the trainable submodules
(`encoder_a`, `encoder_b`, `decoder`, `motion_prior`, the VAE μ/σ heads
or the VQ codebook) and the optimizer state are persisted.

### Play / visualize

```bash
uv run python -m mjlab.scripts.play \
  --task Mjlab-MotionPrior-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/your_motion.npz \
  --agent.load-run "<timestamp>" \
  --agent.load-checkpoint "model_<iter>.pt"
```

`play.py` calls `runner.get_inference_policy()` which returns a callable
that runs the **deploy path only** (proprioception → `motion_prior` →
`decoder` → action). Teacher obs are ignored even though the env still
produces them — this is the same behavior the on-robot ONNX exhibits.

---

## Loss components

### VAE (`DistillationMotionPrior`)

```
behavior = w_a · MSE(student_a, teacher_a) + w_b · MSE(student_b, teacher_b)
ar1      = mu_regu_loss_coeff · (||Δμ_a|| + ||Δμ_b||)        # episode-boundary masked
kl       = kl_coeff · ( KL(N(μ_a, σ_a²) ‖ N(μ_mp_a, σ_mp_a²))
                       + KL(N(μ_b, σ_b²) ‖ N(μ_mp_b, σ_mp_b²)) )
align    = align_loss_coeff · ( ‖mean(μ_a) - mean(μ_b)‖²           # default 0
                               + ‖mean(logσ_a²) - mean(logσ_b²)‖² )
total    = behavior + ar1 + kl + align
```

* `kl_coeff` linearly anneals from `kl_loss_coeff_max` (default 0.01) to
  `kl_loss_coeff_min` (default 0.001) between iters
  `anneal_start_iter` (default 2500) and `anneal_end_iter` (default 5000).
* `align_loss` is first-moment matching across teachers (off by default;
  enable when the two encoders' latents collapse onto the same region).

### VQ-VAE (`DistillationMotionPriorVQ`)

```
behavior = w_a · MSE(student_a, teacher_a) + w_b · MSE(student_b, teacher_b)
ar1      = mu_regu_loss_coeff · (||Δenc_a|| + ||Δenc_b||)    # raw encoder output, not quantized
commit   = commit_loss_coeff · (commit_a + commit_b)         # β=0.25 default
mp       = mp_loss_coeff · ( MSE(mp_code_a, q_a.detach())
                            + MSE(mp_code_b, q_b.detach()) )
total    = behavior + ar1 + commit + mp
```

* `commit_*` come straight from `EMAQuantizer.forward(training=True)`;
  they are MSE between the encoder output and its quantized lookup.
* `mp` supervises the `motion_prior` head to predict (in `code_dim`) the
  quantized code the encoder would have chosen — the deploy path target.
* No KL, no `align_loss`: the shared codebook already pins both encoders
  into one discrete dictionary.

Per-submodule gradient clipping (default `max_grad_norm=1.0`) is applied
to `encoder_a / encoder_b / decoder / motion_prior` (plus the VAE μ/σ
heads). The frozen teachers and the VQ codebook (no trainable params)
are skipped automatically.

---

## Differences from the upstream `motionprior` reference

* **teacher_b is a plain MLP, not a TemporalCNN**. The upstream design
  assumed both teachers were Teleopit-style TemporalCNNs. mjlab's
  `Velocity-Rough-Unitree-G1` actor (286-d obs → MLP[512,256,128] → 29
  dof, scalar Gaussian std) is loaded by a separate
  `load_velocity_teacher` and exposes only the 1-D obs path.
* **Two physical envs, not one mixed-terrain env**. Each teacher runs in
  its own env (flat for teacher_a, rough for teacher_b) so each sees its
  training distribution. The runner steps both envs every iteration,
  shares the student, and routes behavior loss per-env to the matching
  teacher.
* **Rollout / epoch decoupling**. Upstream stores student outputs (with
  autograd graphs attached) and re-backwards them across
  `num_learning_epochs` — this trips
  `RuntimeError: Trying to backward through the graph a second time`.
  This implementation runs rollout under `torch.no_grad`, stores only
  detached inputs + frozen teacher targets, and re-forwards trainable
  submodules each epoch for a fresh graph.
* **VQ algorithm has actual VQ losses**. Upstream
  `distillation_motion_prior_vq.py` is a copy-paste of the VAE algorithm
  that references VAE-only fields (`es_mu`, `mp_mu`, etc.) and never
  applies the commit loss. This implementation supplies a real VQ loss
  pack (commit + AR(1) on raw encoder + motion_prior code regression).
* **EMA quantizer state is checkpointed**. Upstream stores `code_sum` and
  `code_count` as plain attributes, so they don't follow `.to(device)`
  and aren't saved in `state_dict`. They are buffers here.

---

## File layout

```
src/mjlab/tasks/motion_prior/
├── motion_prior_env_cfg.py             — base env factory
├── observations_cfg.py                 — student / teacher_a / teacher_a_history / teacher_b builders
├── onnx.py                             — VAE & VQ deploy wrappers + ONNX export
├── rl_cfg.py                           — typed RslRlMotionPriorRunnerCfg / Policy / Algo
├── mdp/
│   ├── __init__.py                     — re-exports tracking mdp + new ref_*_b
│   └── observations.py                 — ref_base_lin_vel_b / ang_vel / projected_gravity_b
├── teacher/
│   ├── loader.py                       — Teleopit TemporalCNN loader
│   ├── velocity_loader.py              — mjlab velocity MLP loader
│   ├── temporal_cnn_model.py           — TemporalCNNModel (port from Teleopit)
│   └── conv1d_encoder.py               — Conv1dEncoder
├── rl/
│   ├── runner.py                       — MotionPriorOnPolicyRunner (dual env)
│   ├── policies/
│   │   ├── motion_prior_policy.py      — VAE policy (dual encoder + shared decoder)
│   │   ├── motion_prior_vq_policy.py   — VQ policy (shared codebook + decoder)
│   │   └── quantizer.py                — EMAQuantizer
│   └── algorithms/
│       ├── distillation_motion_prior.py    — VAE loss + AR(1) + KL anneal + align
│       └── distillation_motion_prior_vq.py — VQ loss + commit + AR(1) + mp regression
└── config/g1/
    ├── env_cfgs.py                     — flat (teacher_a) + rough (teacher_b) variants
    └── rl_cfg.py                       — registered runner cfg per task

tests/
├── test_motion_prior_teacher_load.py   — Teleopit ckpt parity (8 tests)
├── test_motion_prior_policy.py         — VAE policy invariants (8 tests)
├── test_motion_prior_algorithm.py      — VAE algorithm math (8 tests)
├── test_motion_prior_runner.py         — runner wiring with fake env (4 tests)
├── test_motion_prior_e2e.py            — real GPU env smoke (slow, 1 test)
├── test_motion_prior_vq_policy.py      — VQ policy + EMAQuantizer (11 tests)
├── test_motion_prior_vq_algorithm.py   — VQ algorithm math (6 tests)
└── test_motion_prior_onnx.py           — VAE/VQ deploy + ONNX parity (8 tests)
```
