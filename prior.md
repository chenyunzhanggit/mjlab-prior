# Motion Prior 蒸馏迁移计划（isaaclab → mjlab）

> 把 isaaclab 工程 `~/zcy/motionprior` 中的 `g1_motion_prior` /
> `g1_motion_prior_vq` 两个 task 的策略蒸馏流程，迁移到本仓库 mjlab 环境。
> **Dual-Teacher 蒸馏**：
> - `teacher_a` = Teleopit `track.pt`（TemporalCNN, 166 obs + 10 帧 history → 29 dof）
>   → ckpt: `~/zcy/Teleopit/track.pt`
> - `teacher_b` = mjlab `Mjlab-Velocity-Rough-Unitree-G1` 训出的 actor（**plain MLP**, 286 obs → 29 dof）
>   → ckpt: `~/zcy/mjlab-prior/logs/model_21000.pt`

---

## 实施进度（截至 2026-04-29，14 个 task 全部完成）

**完成（task #1–#14）**

| TODO# | 状态 | 文件 / 关键产出 |
|---|---|---|
| #1 | ✅ | Teleopit ckpt 内部结构梳理（`actor_state_dict` 27 tensor 1:1 对应 `TemporalCNNModel`） |
| #2 | ✅ | `src/mjlab/tasks/motion_prior/{__init__,motion_prior_env_cfg,mdp,rl,config}` 骨架 |
| #3 | ✅ | `mdp/observations.py`（`ref_base_lin_vel_b/ref_base_ang_vel_b/ref_projected_gravity_b`）+ `observations_cfg.py`（student / teacher_a / teacher_a_history / teacher_b builder）+ `config/g1/env_cfgs.py` 双 env 变体 |
| #4 | ✅ | flat env reward 裁到只剩 `motion_global_anchor_pos`；rough env 保留全套 velocity reward / curriculum |
| #5 | ✅ | `tests/test_motion_prior_teacher_load.py`（8 tests，PyTorch ↔ ONNX 1e-5 一致） |
| #6 | ✅ | `rl/policies/motion_prior_policy.py` `MotionPriorPolicy` + `teacher/velocity_loader.py`（teacher_b loader）+ `tests/test_motion_prior_policy.py`（8 tests） |
| #7 | ✅ | `rl/algorithms/distillation_motion_prior.py`（dual-teacher behavior + AR(1) + 退火 KL + 可选 align_loss + 子模块 grad clip）+ `tests/test_motion_prior_algorithm.py`（8 tests） |
| #8 | ✅ | `rl/runner.py` `MotionPriorOnPolicyRunner` 双 vec env 串联 rollout；`_collect_rollout` 只存 detached inputs，`_epoch_step` 每 epoch 重新前向构图 + `tests/test_motion_prior_runner.py`（4 tests） |
| #9 | ✅ | `rl_cfg.py` typed `RslRlMotionPriorRunnerCfg / PolicyCfg / AlgoCfg`；`config/g1/rl_cfg.py` 用 typed cfg |
| #10 | ✅ | `tests/test_motion_prior_e2e.py`（slow，GPU 真盘 32+32 envs × 5 iter，46s 通过） |
| #11 | ✅ | `rl/policies/quantizer.py` `EMAQuantizer`（`code_sum/count` 改 buffer，device-agnostic）+ `motion_prior_vq_policy.py` `MotionPriorVQPolicy`（双 encoder + 共享 codebook/decoder/motion_prior）+ `tests/test_motion_prior_vq_policy.py`（11 tests） |
| #12 | ✅ | `rl/algorithms/distillation_motion_prior_vq.py`（behavior + AR(1) on raw enc + commit + mp code-prediction，无 KL/align）+ `tests/test_motion_prior_vq_algorithm.py`（6 tests） |
| #13 | ✅ | `onnx.py`（`MotionPriorVAEDeployModel` / `MotionPriorVQDeployModel` Path 3 + `export_motion_prior_to_onnx`）+ runner `get_inference_policy` + `export_policy_to_onnx`，`save()` 自动落盘 ONNX + `tests/test_motion_prior_onnx.py`（8 tests，PyTorch ↔ ONNX 1e-5 一致） |
| #14 | ✅ | `docs/source/changelog.rst` 新增 Added 条目；`src/mjlab/tasks/motion_prior/README.md` 含架构图、launch 命令、loss 公式、与 upstream 差异说明 |

**已注册任务 ID**

- `Mjlab-MotionPrior-Flat-Unitree-G1` — flat env + teacher_a + VAE 蒸馏（**双 teacher 训练入口**：runner 内部建 rough env）
- `Mjlab-MotionPrior-Rough-Unitree-G1` — rough env + teacher_b（单 teacher baseline）
- `Mjlab-MotionPrior-VQ-Flat-Unitree-G1` — VQ-VAE 蒸馏，env 同 flat 版

**测试总数**：54（53 fast + 1 slow GPU e2e）。`make check`（ruff format / check / ty）全清。

**部署 path**：ONNX 导出 + `runner.get_inference_policy()` 都走 **Path 3** —
`prop_obs (372) → motion_prior → [mp_mu | code_dim] → decoder → action (29)`。
机器人侧没有 motion command / ref velocities / height_scan，所以 teacher 路径只在仿真评估时用，不参与部署。

启动命令、loss 详解、与 upstream 差异等见 `src/mjlab/tasks/motion_prior/README.md`。

**关键设计修正（区别于最初设计）**

1. **teacher_b 不是另一个 TemporalCNN**。它来自 `Mjlab-Velocity-Rough-Unitree-G1`，是 plain MLP（286 → MLP[512,256,128] → 29，scalar Gaussian std）。
   原 prior.md 中假定"teacher_b 同理"那段是错的；teacher_b 没有 `actor_history` 路径，只吃 1-D obs。
2. **双 vec env 而非单 env mixed terrain**。teacher_a 期望 flat 地面，teacher_b 期望 rough 地形 — 用两个 `ManagerBasedRlEnv` 在同一 GPU 进程并行 rollout（runner 内部各 step 一次），共享一份 student policy。
   - 要点：`student` obs schema 在两 env 上严格一致（5 项 proprio × history=4 = 372 维），保证 student 网络在两边都能跑。
   - GPU 显存：`secondary_num_envs` 默认与 primary 同尺寸，单卡足够；缩 num_envs 即可让出空间。
3. **Rollout / epoch 解耦**。原 motionprior 在 rollout 时直接存 student 输出（autograd graph），多 epoch 重 backward 会撞 `RuntimeError`。我们改为：rollout 全 `torch.no_grad`，只存 detached inputs；每 epoch 用存好的输入重新前向，graph 全新。
4. **ckpt 路径**：原文档写 `~/project/...`，实际仓库都在 `~/zcy/...`。

**所有 task 已完成。** 后续工作：跑长 horizon 训练 + sim2real 验证。

---

## Teleopit Teacher 实情（关键，影响整个迁移）

> 之前默认 Teacher 是普通 MLP 是错的。Teleopit 的 Teacher 是 **TemporalCNN**：
> 包含 1-D obs 的 `actor` 组（MLP 路径）+ 多帧 history 的 `actor_history`
> 组（Conv1D 路径），两路 latent 拼接后过 MLP 产出 action。这一点会**整体
> 改写** task #1, #3, #5, #11 的具体实现。

### Teacher 架构（`Teleopit/train_mimic/tasks/tracking/rl/temporal_cnn_model.py`
+ `conv1d_encoder.py`）

```
actor (B, D_actor)              ──► [obs_normalizer] ──► latent_1d
actor_history (B, T=10, D_actor)──► [per-group EmpiricalNorm] ──► permute(B,D,T)
                                ──► Conv1dEncoder(out=(128,64,32), k=3, ELU,
                                    AdaptiveAvgPool1d(1)) ──► latent_3d (B,32)
[latent_1d, latent_3d]          ──► MLP(1024,512,256,256,128) ──► actions
                                ──► GaussianDistribution(init_std=1.0, scalar)
```

关键超参（`config/rl.py:13-44`）：
```
hidden_dims      = (1024, 512, 256, 256, 128)
cnn_output_chs   = (128, 64, 32)
kernel_size      = 3
activation       = "elu"
global_pool      = "avg"
obs_normalization = True            # 包括 1-D 与 3-D 两路
distribution     = GaussianDistribution(init_std=1.0, scalar)
```

### Teacher Actor obs（**这就是我们 Teacher Encoder 的输入**）

由两个 obs group 拼成（**顺序依 mjlab `obs_groups` 配置**）：

**1) `actor` 组**（1-D, 当前帧）— 来源 `tracking_env_cfg.py:44-77` +
`config/env.py:154-166` 的覆盖：

```python
actor = {
    # 来自 base tracking_env_cfg，但 General-Tracking-G1 删除了
    # "motion_anchor_pos_b" 和 "base_lin_vel" 两项 (env.py:154-158)
    "command":             generated_commands(motion),
    "motion_anchor_ori_b": noise=Unoise(±0.05),
    "base_ang_vel":        builtin_sensor("robot/imu_ang_vel"), noise=±0.2,
    "joint_pos":           joint_pos_rel(biased=True), noise=±0.01,
    "joint_vel":           joint_vel_rel,              noise=±0.5,
    "actions":             last_action,
    # 然后由 General-Tracking-G1 注入 _VELCMD_ACTOR_TERMS (env.py:69-86):
    "projected_gravity":     noise=±0.05,
    "ref_base_lin_vel_b":    params={"command_name": "motion"},
    "ref_base_ang_vel_b":    params={"command_name": "motion"},
    "ref_projected_gravity_b": params={"command_name": "motion"},
}
# concatenate_terms=True, enable_corruption=True
```

**2) `actor_history` 组**（3-D, `(B, T=10, D)`）— `config/env.py:50-59`：

```python
cfg.observations["actor_history"] = ObservationGroupCfg(
    terms=deepcopy(cfg.observations["actor"].terms),  # 与 actor 同 terms
    concatenate_terms=True,
    enable_corruption=cfg.observations["actor"].enable_corruption,
    history_length=10,
    flatten_history_dim=False,                        # 保留 (B,T,D)
)
```

**关键事实**：`actor_history` 与 `actor` 的 term 集合完全相同，只是
history_length=10 且不展平。所以 mjlab 这边**只要构造出与 `actor` 相同的
union obs**，就能同时给 Teacher 喂 `actor` (1-D) 与 `actor_history` (3-D)。

### Teacher checkpoint 形式（Step 1 实测确认）

ckpt 顶层 key（`scripts/inspect_teleopit_ckpt.py` 输出）：
```
['actor_state_dict', 'critic_state_dict', 'optimizer_state_dict',
 'iter'=30000, 'infos'={'env_state': {'common_step_counter': 720000}}]
```

> ⚠️ 修正：**不是 `model_state_dict`，是 `actor_state_dict` 顶层一层**，
> 直接 `model.load_state_dict(ckpt['actor_state_dict'])` 即可，**不需要前缀
> 过滤或 remap**。

`actor_state_dict` 内部分组（n_tensors=27）：

| 子模块 | tensor | shape | 说明 |
|---|---|---|---|
| `obs_normalizer` | `_mean / _var / _std` | `(1, 166)` | 1-D actor obs running stats |
| `obs_normalizer` | `count` | `()` | 已观测样本数 |
| `obs_normalizers_3d.actor_history` | `_mean / _var / _std` | `(1, 166)` | 3-D history running stats（同样 166 维） |
| `cnn_encoders.actor_history.net` | `0.weight / 0.bias` | `(128, 166, 3) / (128,)` | Conv1D in_channels=166 |
| `cnn_encoders.actor_history.net` | `2.weight / 2.bias` | `(64, 128, 3)` | Conv1D 中间层 |
| `cnn_encoders.actor_history.net` | `4.weight / 4.bias` | `(32, 64, 3)` | Conv1D 末层 → 32 维 latent |
| `mlp` | `0.weight / 0.bias` | `(1024, 198) / (1024,)` | **首层 in=198 = 166（1D）+ 32（CNN latent）** |
| `mlp` | `2/4/6/8.weight` | `(512,1024) / (256,512) / (256,256) / (128,256)` | hidden_dims=(1024,512,256,256,128) |
| `mlp` | `10.weight / 10.bias` | `(29, 128) / (29,)` | 输出 = G1 29 dof |
| `distribution` | `std_param` | `(29,)` | GaussianDistribution scalar std |

**ONNX schema**（`track.onnx`）：
```
inputs:  obs          shape=[1, 166]
         obs_history  shape=[1, 10, 166]
outputs: actions      shape=[1, 29]
```

> 喂 ONNX 时 `obs_history` 是 `(B, T, D)`，**不是** `(B, D, T)`。模型内部
> 自己 `permute(0, 2, 1)` 转 channels-first 给 Conv1D，外部不要预先转。

零输入推理输出 norm = `4.055428`（保留作为 PyTorch 对照基线，task #11 用）。

**critic** state_dict 不会被 motion prior 训练使用（蒸馏不需要 value），
仅作记录：critic obs dim=298（多 132 维特权 body_pos / body_ori），其余结构同 actor。

**迁移结论**：
1. MotionPriorPolicy 不再持有 `nn.Sequential teacher`，改持有
   `TemporalCNNModel`（或 inference 子集 `_TorchTemporalCNNModel`）。
2. `evaluate(actor_obs, actor_history)` → 调用其 `forward(obs, obs_history)`，
   返回 action（deterministic 路径用 `mean`）。
3. ckpt 加载方式：
   ```python
   ckpt = torch.load(path, map_location=device, weights_only=False)
   model.load_state_dict(ckpt["actor_state_dict"], strict=True)
   ```
   `strict=True` 安全——ckpt 27 个 tensor 与 `TemporalCNNModel` 应当 1:1
   对应（含 `distribution.std_param`，蒸馏不用但保留无害）。

---

## motionprior 工程参考实现要点（迁移可直接照抄/对照）

## motionprior 工程参考实现要点（迁移可直接照抄/对照）

下面所有引用都基于 `~/zcy/motionprior/source/whole_body_tracking/` 子树。

### A. Teacher 观测组成（`envs/base/base_config.py:118-156`）

`TeacherObservationsCfg.PolicyCfg` 顺序为：

```
command (motion)                          # mdp.generated_commands
projected_gravity        history_length=4 noise=Unoise(±0.05)
motion_ref_ang_vel                        noise=Unoise(±0.05)
base_ang_vel             history_length=4 noise=Unoise(±0.2)
joint_pos_rel            history_length=4 noise=Unoise(±0.01)
joint_vel_rel            history_length=4 noise=Unoise(±1.0)
last_action              history_length=4
base_lin_vel                              # 特权：本体线速度
anchor_pos_diff                           # 特权：anchor pos 误差
key_body_pos_diff                         # 特权：关键 body 相对 pos 误差
key_body_rot_diff                         # 特权：关键 body 相对 ori 误差
anchor_height                             # 特权：anchor 高度（仅 flat env）
motion_anchor_pos_b_future noise=Unoise(±0.25)
motion_anchor_ori_b_future noise=Unoise(±0.05)
__post_init__: enable_corruption=True, concatenate_terms=True
```

**迁移备注**：mjlab 的 `Mjlab-Tracking-Flat-Unitree-G1` 已有大部分等价 obs
term；缺失的（`anchor_height`、`relative_body_pos_error` /
`relative_body_orientation_error`、`motion_anchor_pos_b_future`）需在
`mjlab/tasks/tracking/mdp/observations.py` 补齐或映射到现有名字。

### B. Student 观测组成（`envs/unitree_g1/g1_motion_prior_cfg.py:60-65`）

```python
class StudentObsCfg(ObsGroup):
    projected_gravity = ObsTerm(..., noise=Unoise(±0.05), history_length=4)
    base_ang_vel      = ObsTerm(..., noise=Unoise(±0.2),  history_length=4)
    joint_pos         = ObsTerm(joint_pos_rel, noise=Unoise(±0.01), history_length=4)
    joint_vel         = ObsTerm(joint_vel_rel, noise=Unoise(±1.0),  history_length=4)
    actions           = ObsTerm(last_action,   history_length=4)
```

**注意**：student 完全是本体感知，**不含 motion command**，部署时不需要
teacher_obs；这是蒸馏的核心价值。

### C. G1 关键 body / anchor 列表（`g1_motion_prior_cfg.py:108-124`）

```python
commands.motion.anchor_body_name = "torso_link"
commands.motion.body_names = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
```

mjlab 的 `unitree_g1_flat_tracking_env_cfg`
（`src/mjlab/tasks/tracking/config/g1/env_cfgs.py:38-56`）这一段**完全一致**，
直接复用。

### D. Reward / 终止条件（`g1_motion_prior_cfg.py:46-53`）

蒸馏阶段**只保留一项 reward**：

```python
motion_global_anchor_pos = RewTerm(
    func=mdp.motion_global_anchor_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.3},
)
```

虽然蒸馏 loss 不依赖 reward，但环境仍需 reward 用于 termination 监控、
episode log、curriculum 触发，不能直接删空。

### E. MotionPriorPolicy 架构（`my_modules/motion_prior_policy.py`）

| 子模块 | 输入 | 输出 | 默认 hidden |
|---|---|---|---|
| `teacher` (frozen) | teacher_obs | num_actions | `[512,256,128]` |
| `encoder` | teacher_obs | latent_z_dims=32 | `[512,256,128]` |
| `es_mu`, `es_var` | latent_z_dims | latent_z_dims | 单层 Linear |
| `motion_prior` | prop_obs (= num_prior_obs) | latent_z_dims | `[512,256,128]` |
| `mp_mu`, `mp_var` | latent_z_dims | latent_z_dims | 单层 Linear |
| `decoder` | `[latent_z, prop_obs]` (latent + num_prior_obs) | num_actions | `[512,256,128]` |

关键接口（迁移时**保持同名**，方便 inference / downstream 复用）：

- `act(prop_obs, teacher_obs)` → `(enc_mu, enc_log_var, latent_z, recons_actions, mp_mu, mp_log_var)`
- `evaluate(teacher_obs)` → 冻结 teacher 的 action（teacher 调 `with torch.no_grad()`）
- `policy_inference(prop_obs, teacher_obs)` → `recons_actions`（推理时 `latent_z = enc_mu`，**不采样**）
- `encoder_inference(teacher_obs)` → `(mu, log_var)`
- `decoder_inference(prop_obs, latent_z)` → action
- `motion_prior_inference(prop_obs)` → `mp_mu`（部署时实际用这个）
- `reparameterize(mu, log_var)` → `mu + eps * exp(0.5 * log_var)`

**Teacher key remap**（关键陷阱，原代码 `motion_prior_policy.py:62-65`）：

```python
teacher_state_dict = teacher_policy_dict['model_state_dict']
actor_state_dict = {
    k.replace('actor.', ''): v
    for k, v in teacher_state_dict.items()
    if 'actor' in k
}
self.teacher.load_state_dict(actor_state_dict, strict=False)
```

mjlab 这边 tracking ckpt 的 actor 命名要先打印一下确认前缀（可能是
`policy.actor.0.weight` 或别的），按实际改 prefix；建议在 task #11 单独
写个测试覆盖。

### F. Loss 实现细节（`my_algorithms/distillation_motion_prior.py:115-200`）

**1) Behavior loss**：
```python
behavior_loss = self.loss_fn(actions_teacher_buffer, actions_student_buffer)
# loss_fn ∈ {mse, huber}, 默认 mse
```

**2) AR(1) 平滑先验**（处理 episode 边界）：
```python
phi = 0.99
time_zs = enc_mu_buffer_time_stack  # (num_envs, seq_len, latent_dim)
error = time_zs[:, 1:] - time_zs[:, :-1] * phi
# episode 边界 mask（progress 不连续 / episode 开头 idx<=2）
not_consecs = ((idxes[:, 1:] - idxes[:, :-1]) != 1).view(-1)
error[not_consecs] = 0
starteres = ((idxes <= 2)[:, 1:] + (idxes <= 2)[:, :-1]).view(-1)
error[starteres] = 0
ar1_prior = torch.norm(error, dim=-1).mean()
mu_regu_loss_coeff = 0.01
```

mjlab 端要从 `env.episode_length_buf` 在 rollout 每步 clone 出
`progress_buf`，然后 stack 成 `(num_envs, seq_len, 1)`。

**3) KL（VAE 版）**：
```python
kl_loss = 0.5 * torch.mean(
    mp_log_var - enc_log_var
    + (torch.exp(enc_log_var) + (enc_mu - mp_mu) ** 2) / torch.exp(mp_log_var)
    - 1
)
# 退火：anneal_start_iter=2500, anneal_end_iter=5000, max=0.01, min=0.001
```

**4) 总 loss**：
```python
loss = behavior_loss + 0.01 * ar1_prior + kl_loss_coeff * kl_loss
```

**5) Grad clip**（**对每个子模块单独 clip，不是整体 clip**）：
```python
for m in [encoder, es_mu, es_var, decoder, motion_prior, mp_mu, mp_var]:
    nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)  # 1.0
```

### G. Runner 训练循环（`my_runner/on_policy_runner_mp.py:106-289`）

每个 iteration 内：

```python
for _ in range(num_steps_per_env):  # 24
    enc_mu, enc_log_var, latent_z, actions, mp_mu, mp_log_var = policy.act(obs, priv_obs)
    priv_act = policy.evaluate(priv_obs)            # teacher action
    obs, rew, dones, infos = env.step(actions.detach())
    priv_obs = infos["observations"]["teacher"]      # ← isaac 特有

    actions_teacher_buffer.append(priv_act)
    actions_student_buffer.append(actions)
    enc_mu_buffer.append(enc_mu); enc_log_var_buffer.append(enc_log_var)
    mp_mu_buffer.append(mp_mu);   mp_log_var_buffer.append(mp_log_var)
    progress_buf.append(env.episode_length_buf.clone().detach())

# 时间维 stack 后 cat batch 维
enc_mu_buffer_time_stack = torch.stack(enc_mu_buffer, dim=1)  # (envs, T, dim)
progress_buf             = torch.stack(progress_buf, dim=1)   # (envs, T)
... (其余 .cat(dim=0))
loss_dict = alg.compute_loss_one_batch(..., progress_buf=progress_buf, cur_iter_num=it)
```

**mjlab 适配点**：把 `infos["observations"]["teacher"]` 替换为
`obs_dict["teacher_a"]` / `obs_dict["teacher_b"]`（来自
`RslRlVecEnvWrapper.get_observations()` 的 `TensorDict`）。

### H. 训练超参（`g1_motion_prior_cfg.py:127-149`）

```
num_steps_per_env  = 24
max_iterations     = 100000
save_interval      = 500
runner_class_name  = "MotionPriorOnPolicyRunner"
experiment_name    = "motion_prior"
empirical_normalization = False

policy.init_noise_std            = 1.0      # 蒸馏其实没用到 std 采样，留着兼容
policy.teacher_hidden_dims       = [512,256,128]
policy.encoder_hidden_dims       = [512,256,128]
policy.latent_z_dims             = 32
policy.decoder_hidden_dims       = [512,256,128]
policy.motion_prior_hidden_dims  = [512,256,128]
policy.activation                = "elu"

algorithm.num_learning_epochs = 5
algorithm.learning_rate       = 5.0e-4
algorithm.max_grad_norm       = 1.0
algorithm.gradient_length     = 1
```

### I. VQ 版差异（`g1_motion_prior_vq_cfg.py` + `vq_policy/*.py`）

```
runner_class_name  = "MotionPriorVQOnPolicyRunner"
experiment_name    = "motion_prior_vq"
policy.class_name  = "MotionPriorVQPolicy"
policy.code_dim    = 64
policy.num_code    = 2048
algorithm.class_name           = "DistillationMotionPriorVQ"
algorithm.use_q_encoder_output = False   # 是否把量化后的 code 输出当 encoder mu
```

VQ 用最近邻 `argmin ||z - codebook||²` 得到 code index，量化后通过
straight-through estimator 反传梯度；motion_prior 头改为预测 code 的
**类别分布**（cross-entropy 监督）。

### J. 双 Teacher 在本工程的具体落地建议

回到你的两个方案：

**方案 1：同 Encoder + 占位填 `-100`**

```
union_teacher_obs = concat(teacher_a_obs, teacher_b_specific_obs)
                    或 padded_obs[mask_a] = teacher_a_obs;
                       padded_obs[mask_b] = teacher_b_obs (其余 -100)

encoder(union_teacher_obs) → (μ, log σ²)
behavior_loss = MSE(student_action, teacher_a_action) * mask_a
              + MSE(student_action, teacher_b_action) * mask_b
```

- 优点：**单 encoder**、参数少、推理路径简单；下游 motion_prior 头天然就是
  对 union 空间建模。
- 风险：`-100` 是一个魔数；encoder 的第一层权重要学会"识别占位"，需要保证
  occlusion mask 在 batch 中两 teacher 比例均衡，否则会被多数 teacher
  主导。
- 工程提示：env 这一侧要稳定输出固定维度 `union_teacher_obs`，建议把
  "占位填充"做在 env 的 obs func 内部（写一个 wrapper obs term），
  而不是在 policy 里临时拼。

**方案 2（推荐先走通）：双 Encoder + 共享 motion_prior**

```
encoder_a(teacher_a_obs) → (μ_a, log σ²_a)
encoder_b(teacher_b_obs) → (μ_b, log σ²_b)
motion_prior(prop_obs)   → (μ_mp, log σ²_mp)   # 唯一一份

behavior_loss = w_a · MSE(student_action_a, teacher_a_action)
              + w_b · MSE(student_action_b, teacher_b_action)
kl_loss       = KL(N(μ_a, σ_a²) || N(μ_mp, σ_mp²))
              + KL(N(μ_b, σ_b²) || N(μ_mp, σ_mp²))   # 两份都拉向同一 mp
```

`student_action_*` 走各自 `decoder(latent_z_*, prop_obs)`，但 **decoder
共享一份**（这是把"两 teacher 压缩到同一 motion prior"的关键，
共享 decoder 强制两 latent 落到同一动作流形）。

> **关于你草稿中的疑问**："两个 encoder 输出的 mu / std 是否要加 loss 约束
> 距离，否则不同动作分布会重合？"
>
> 我的判断：
> 1. 共享 motion_prior + 共享 decoder 已经隐式拉近两 latent 分布，
>    通常**不需要额外 align**。
> 2. 如果观察到训练初期两 latent 子空间互相覆盖（=动作混淆），开一个
>    `align_loss = ||μ_a − μ_b||² + ||log σ_a² − log σ_b²||²`，
>    系数 0.001~0.01；但**只有在两 teacher 输入到同一 batch 元素**时才有
>    意义（比如同一帧 motion 同时被两个 teacher 评估）。如果实际 batch
>    里 a / b 来自不同 env，就不能直接做点对点 align，要改成分布层面的
>    moment matching（mean/var 一阶统计对齐）。
> 3. **不希望对齐**的情况也存在：如果两 teacher 本来就负责不同动作族
>    （比如 walking vs climbing），强行对齐反而损失多样性。这时只靠
>    `kl(*, mp)` 拉到同一 mp 就够了，**故意让 μ_a / μ_b 在 latent 空间分开**
>    才是合理的。

**结论**：先实现方案 2 + 共享 decoder + 共享 motion_prior + `align_loss_coeff=0`
跑 baseline；再开 `align_loss_coeff=0.01` 做对照；方案 1 作为后续 ablation。

### K. mjlab 端可直接复用的现成件（不要重写）

- `mjlab.tasks.tracking.tracking_env_cfg.make_tracking_env_cfg`
  → motion_prior env_cfg 直接基于它继承
- `mjlab.tasks.tracking.mdp.MotionCommandCfg`
  → commands.motion 配置
- `mjlab.tasks.tracking.config.g1.env_cfgs.unitree_g1_flat_tracking_env_cfg`
  → G1 关键参数（action_scale、anchor_body_name、body_names、self_collision
  传感器、ee_body_pos termination）一字不改可用
- `mjlab.rl.runner.MjlabOnPolicyRunner`
  → save / ONNX 导出框架；只重写 `learn()`
- `mjlab.rl.vecenv_wrapper.RslRlVecEnvWrapper.get_observations`
  → 已经返回 TensorDict，多 obs group 天然友好

---

## 核心理解

`g1_motion_prior` 与 `g1_motion_prior_vq` 不是普通 RL 任务，而是
**VAE / VQ-VAE 形式的策略蒸馏**：

```
teacher_obs (含 motion ref/anchor 误差)  ──► [冻结的 teacher MLP]  ──► teacher_action
                                       └─► [encoder] ─► (μ_e, log σ²_e) ──► z
prop_obs (本体感知, student 用)         ──► [motion_prior MLP] ──► (μ_mp, log σ²_mp)
[z, prop_obs]                            ──► [decoder] ──► student_action

loss = MSE(teacher_action, student_action)             # behavior cloning
     + λ_AR1 · ||z_t - 0.99·z_{t-1}||                   # 时序平滑先验
     + λ_KL  · KL( N(μ_e, σ²_e) || N(μ_mp, σ²_mp) )    # 让 motion_prior 头逼近 encoder
```

VQ 版把 encoder 输出量化到 codebook（2048×64），KL 换成离散 cross-entropy
+ commitment / VQ loss。

蒸馏完后产物有三：可单独使用的 **encoder**（提 z）、**motion_prior 头**
（无监督下也能给 z）、**decoder**（z + prop_obs → action），后续 downstream
任务用得上。

## 关键迁移差异（mjlab vs isaaclab）

| 维度 | isaaclab | mjlab | 影响 |
|---|---|---|---|
| 多组 obs | `extras["observations"]["teacher"]` | `TensorDict`，所有 group 一起返回 | runner 取 obs 的方式要改 |
| Runner 基类 | `OnPolicyRunner` | `MjlabOnPolicyRunner`（含 onnx 导出 / wandb） | 自定义 runner 时尽量复用 save / load |
| ckpt 命名 | `actor.0.weight` 等 | mjlab 走 `ActorCritic`，需核对前缀 | teacher 权重加载要 remap key（task #11 专门验证） |
| `history_length` | obs term 原生支持 | 需先确认 mjlab 是否支持 | 影响 student / teacher obs 形状 |
| 课程 / reward | tracking 任务有完整 motion reward | 直接复用 | 蒸馏阶段只保留必要项 |

## 已知坑（提前避开）

- **`ObservationTermCfg.history_length`** — mjlab 目前未确认是否原生支持，
  如缺失需要在 manager 层加，或在 obs func 内部维护 ring buffer。
- **`teacher_policy_path` 走 rl_cfg.policy 字段**，与 isaaclab 端保持一致，
  不要塞进 env_cfg。
- **不要继承 PPO 的 learn()** — 蒸馏没有 critic / GAE，
  `MotionPriorOnPolicyRunner` 继承 `MjlabOnPolicyRunner` 但
  **重写 learn()**，避免被 PPO 的 storage 接口拖累。
- **rsl_rl 5.2.0** — 最近 bump（commit `33fc155e`），如果原 isaac 版依赖较老
  `Distillation` 基类，要核对当前 rsl_rl 还有没有此类；没有就独立写。

---

## TODO List

执行建议顺序：**1 → 2 → 3 → 4 → 11 → 5 → 7 → 9 → 10 → 12 → 6 → 8 → 13 → 14**。

### 1. 梳理 Teacher checkpoint 接口 & 对齐观测（**Teleopit Teacher**）
Teacher 来自 `~/zcy/Teleopit`：`track.pt`（含 obs_normalizer + Conv1D
encoder + MLP）。具体行动：

1. 在 mjlab 端把 Teleopit 的 `tasks/tracking/rl/temporal_cnn_model.py` 与
   `conv1d_encoder.py` 整段拷过来（或加 `Teleopit` 为可编辑依赖），
   保留同名以便 ckpt key 对得上。
2. 比对 mjlab `Mjlab-Tracking-Flat-Unitree-G1` 与 Teleopit
   `make_general_tracking_env_cfg` 的 obs term 差异（参见上方
   "Teleopit Teacher 实情" 节列表）。**对齐 actor obs 是关键**——
   Teleopit 删掉了 `motion_anchor_pos_b` / `base_lin_vel`，但加入了
   `_VELCMD_ACTOR_TERMS` 四项 ref 信号。我们的 motion_prior env 必须
   生成这一**完全一致**的 actor obs schema 给 Teacher，否则 ckpt 推理
   全错。
3. 加 `actor_history` 组（`history_length=10`, `flatten_history_dim=False`），
   含义与 actor 同 terms 但 3-D。
4. 写一个 `_load_temporal_cnn_teacher(path) -> TemporalCNNModel` 工具函数：
   - 从 ckpt 读 `model_state_dict` 中 actor 部分（取决于
     `RslRlOnPolicyRunnerCfg` 的保存格式，通常是 `actor.*` 顶层）
   - 直接 `model.load_state_dict(actor_state, strict=True)`
   - **不要**走原 motionprior 的 `'actor.' replace` 字符串过滤逻辑

**产出**：teacher obs schema 对照表 + 一段可运行的 teacher 加载冒烟脚本
（验证 1e-6 推理一致性，见 task #11）。
**双 Teacher 适配**：列出 *两套* teacher obs schema（teacher_a / teacher_b），
分别记录 actor / actor_history 维度和差集；为方案 1（同 Encoder + 占位填充）
准备 union schema。

### 2. 在 mjlab 新建 motion_prior 任务骨架
参考 `src/mjlab/tasks/tracking/` 的目录结构，新增 `src/mjlab/tasks/motion_prior/`
子包，含：

```
motion_prior_env_cfg.py
mdp/__init__.py
rl/__init__.py
rl/runner.py
config/g1/{__init__.py, env_cfgs.py, rl_cfg.py}
```

在 `config/g1/__init__.py` 用 `register_mjlab_task` 注册两个任务 ID：
- `Mjlab-MotionPrior-Flat-Unitree-G1` （VAE 蒸馏）
- `Mjlab-MotionPrior-VQ-Flat-Unitree-G1` （VQ 蒸馏）

分别对应 isaaclab 端 `g1_motion_prior` 与 `g1_motion_prior_vq`。
`env_cfg` 直接基于 tracking g1 env_cfg 改造（继承 + 替换 obs group），
**不要从零写**。

### 3. 定义 student / 双 teacher 观测组（**含 1-D + 3-D 两路**）
在 `motion_prior_env_cfg` 中替换默认 actor / critic obs，改为：

- **student**：`projected_gravity`, `base_ang_vel`, `joint_pos_rel`,
  `joint_vel_rel`, `last_action`（每项 `history_length=4`），加 `Unoise` 噪声，
  参照 `g1_motion_prior_cfg.py:60-65`。
- **teacher_a / teacher_b**：每个 teacher **各两个 group**：
  - `teacher_a` (1-D, `concatenate_terms=True, enable_corruption=True`) ——
    与 Teleopit `make_general_tracking_env_cfg` 的 actor obs 完全一致
    （含 `_VELCMD_ACTOR_TERMS`，删 `motion_anchor_pos_b` /
    `base_lin_vel`）
  - `teacher_a_history` (3-D, `history_length=10, flatten_history_dim=False`) ——
    `terms=deepcopy(teacher_a.terms)`
  - teacher_b 同理

mjlab 的 `ObservationGroupCfg` 已支持 `history_length` 与
`flatten_history_dim`（确认自 Teleopit `_add_history_obs_groups`
直接调用），无需改 manager。

注：mjlab 的 obs 是 `TensorDict`，多组 obs 会同时进入
`RslRlVecEnvWrapper.get_observations()`，无需改 wrapper。

**双 Teacher 方案抉择（影响 task #5/#7）**：
- **方案 1（同 Encoder + 占位）**：构造 `union_teacher_obs` (+ `_history`)，
  缺失维度填 `-100`；env 端额外暴露 `teacher_mask` 作为 observation。
  注意 1-D 与 3-D 都要 union。
- **方案 2（双 Encoder）**：env 端直接出 4 个 group：`teacher_a`,
  `teacher_a_history`, `teacher_b`, `teacher_b_history`；不需要占位。

> 推荐先按**方案 2** 走通（实现直观、debug 容易），跑通后再做方案 1
> 的对比实验。

### 4. 移植/复用 motion command 与对应 reward / termination
蒸馏阶段仍要驱动机器人跟随 motion，因此 `commands.motion`
+ 关键 reward（`motion_global_anchor_pos` 等） + `ee_body_pos` termination
必须保留。直接复用 `src/mjlab/tasks/tracking/mdp/commands.py` 中的
`MotionCommandCfg` 与 `anchor_body_name='torso_link'`、`body_names` 列表
（与 isaaclab 端一致）。把 `g1_motion_prior_cfg.py` 里的 `RewardsCfg`
（只保留 `motion_global_anchor_pos`，weight=0.5）映射过来，或继承 tracking
完整 reward set 再裁剪。
**产出**：`motion_prior_env_cfg` 的 commands / rewards / terminations 定义。

### 5. 实现 MotionPriorPolicy（VAE 版，双 Teleopit Teacher）网络模块
把 `motionprior/.../my_modules/motion_prior_policy.py` 整体搬到
`mjlab/tasks/motion_prior/rl/policies/motion_prior_policy.py`，但 **teacher
子模块从 `nn.Sequential` 升级为 `TemporalCNNModel`**。

**单 Teacher 基线结构**：teacher TemporalCNN（冻结）+ encoder + es_mu / es_var
+ decoder（输入 `[latent_z, prop_obs]`） + motion_prior MLP + mp_mu / mp_var。

**双 Teacher 扩展**（按 task #3 选定方案改，方案 2 推荐）：
- 持有 `self.teacher_a: TemporalCNNModel`, `self.teacher_b: TemporalCNNModel`
  两个冻结模型（构造时分别 load `track.pt` 与第二个 teacher 的 ckpt）
- 方案 1（同 Encoder + 占位）：encoder 输入 1-D union obs；
  history 路要么 union 后过单 Conv1D，要么直接拼两路 latent
- 方案 2（双 Encoder）：`encoder_a / encoder_b` 各自独立——可以是 MLP
  （只看 1-D actor obs），也可以同样上 TemporalCNN（吃 history）。
  推荐先用 MLP 简单起步，**因为 encoder 的目标是把 teacher_obs 压成
  latent z，不需要时序卷积**；TemporalCNN 仅用于 frozen teacher。

关键接口（保持与原 motionprior 同名）：
- `act(prop_obs, teacher_a_obs, teacher_a_hist, teacher_b_obs, teacher_b_hist)`
- `evaluate(teacher_obs, teacher_history)` — 调用对应 TemporalCNN，
  注意要传**两个张量**（actor 与 actor_history），返回 deterministic action
  （不采样 std；`as_deterministic_output_module` 路径或直接取 `mlp(latent)`）
- `policy_inference` / `encoder_inference` / `decoder_inference` /
  `motion_prior_inference` / `load_state_dict`

`load_state_dict` 不再做 `'actor.'` 字符串过滤；改为：
```python
ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
actor_state = {
    k.removeprefix("actor."): v
    for k, v in ckpt["model_state_dict"].items()
    if k.startswith("actor.")
}
self.teacher_a.load_state_dict(actor_state, strict=True)
for p in self.teacher_a.parameters(): p.requires_grad = False
self.teacher_a.eval()
```

`teacher_a_policy_path` / `teacher_b_policy_path` 通过 rl_cfg 传入。
**需要在 task #11 用 ONNX (`track.onnx`) 当 ground truth 验证 1e-6 一致性**——
ONNX 输入是 `(obs, obs_history)`，正好对应 `(actor, actor_history)` 两组。

### 6. 实现 MotionPriorVQPolicy（VQ 版）网络模块
搬运 `motionprior/.../my_modules/vq_policy/motion_prior_vq_policy.py` 与
`quantize_cnn.py` 到 `mjlab/tasks/motion_prior/rl/policies/`。
VQ 版与 VAE 版的差异：
- encoder 输出送入 codebook（`num_code=2048`, `code_dim=64`）做最近邻量化
- decoder 输入 `[code, prop_obs]`
- motion_prior 头预测 code 分布

需要保留 codebook EMA 更新（如果原实现走的是 EMA）以及 commitment / VQ loss
项。先搬过来不动，留待与 algorithm 对齐时再调。
**双 Teacher 提示**：codebook 共享一份；按方案 2 时两 encoder 写入同一
codebook，相当于天然学到一个跨 teacher 的统一动作字典。

### 7. 实现 DistillationMotionPrior 算法（含 loss 组合，双 Teacher）
在 `mjlab/tasks/motion_prior/rl/algorithms/distillation_motion_prior.py`
复刻 isaaclab 端 `distillation_motion_prior.py`：

- `compute_loss_one_batch`：
  - `behavior_loss`：MSE，**对两 teacher 分别算后加权求和**
    （默认 `w_a = w_b = 1.0`，cfg 暴露权重）
  - `ar1_prior`：`mu_regu_loss_coeff=0.01`，对 `enc_mu` 在时间序列上做
    AR(1) 平滑（双 encoder 时各算各的，再相加）
  - `kl_loss`：`kl_loss_coeff` 在 `anneal_start_iter=2500..anneal_end_iter=5000`
    之间从 `0.01 → 0.001` 退火
    - 单 encoder：`KL(N(μ_e, σ_e) || N(μ_mp, σ_mp))`
    - 双 encoder：两份 `KL(N(μ_{e,*}, σ_{e,*}) || N(μ_mp, σ_mp))` 相加
  - **新增 `align_loss`（仅方案 2，可选但推荐）**：直接回应草稿中的疑问
    "两个 encoder 输出的 mu / std 是否要加 loss 约束其距离"。给一个
    `align_loss_coeff`（默认 0.0，先关掉做对比实验，再开 0.01 起步），
    实现为 `||μ_a - μ_b||² + ||log σ_a² - log σ_b²||²`，避免不同动作分布
    在同一 latent 空间发生不期望的重合 / 漂移。

- 对 `encoder` / `es_mu` / `es_var` / `decoder` / `motion_prior` / `mp_mu` /
  `mp_var` 分别做 grad clip
- 不需要 PPO 的 critic / GAE，可不继承 mjlab 现有 PPO，按需写独立 algorithm 类

注意：原实现里 `progress_buf` 用于屏蔽 episode 边界处的 AR(1) 项，
要从 `env.episode_length_buf` 取并 stack。

### 8. 实现 DistillationMotionPriorVQ 算法
复刻 isaaclab 端 `distillation_motion_prior_vq.py`。相对 VAE 版的差异：
- 把 KL loss 替换为 cross-entropy（motion_prior 预测 code 索引）
- 加 commitment loss + codebook loss
- 保留 `behavior_loss`（双 teacher 加权）与 AR(1)
- 同样对各子模块单独 grad clip
- `use_q_encoder_output` 等开关从 cfg 读取
- 双 Teacher 在 VQ 下推荐方案 2 + 共享 codebook，无需 align_loss
  （codebook 量化已经强约束了 latent 空间）

### 9. 实现 MotionPriorOnPolicyRunner（含 rollout 循环）
在 `mjlab/tasks/motion_prior/rl/runner.py` 写 `MotionPriorOnPolicyRunner`，
继承 mjlab 现有 `MjlabOnPolicyRunner` 但**重写 `learn()` 与初始化**，因为
蒸馏的 rollout 与 PPO 完全不同（无 critic、无 GAE、按时间堆 buffer 后整体
BPTT-like 更新）。

要点：
1. 从 `env.get_observations()` 取出 `TensorDict`，按 group 名拆出 `student`,
   `teacher_a`, `teacher_b`（如方案 1，再额外取 `teacher_mask`）
2. 每个 iteration 内 rollout `num_steps_per_env` 步，按时间维 stack：
   `actions_teacher_a/b`、`actions_student`、`enc_mu_a/b`、`enc_log_var_a/b`、
   `mp_mu / log_var`、`progress_buf`
3. 调 `alg.compute_loss_one_batch`；按 `save_interval` 保存
4. 保存格式与 mjlab `MjlabOnPolicyRunner.save` 兼容（含 ONNX 导出可选）
5. 同名注册一个 `MotionPriorVQOnPolicyRunner` 子类（或参数化），供 VQ 任务使用

### 10. 写 MotionPriorDistillationCfg（rl_cfg）
在 `config/g1/rl_cfg.py` 仿照 isaaclab 端
`G1MotionPriorDistillationCfg` / `G1MotionPriorDistillationVQCfg` 写 dataclass：

- `num_steps_per_env=24`, `max_iterations=100000`, `save_interval=500`
- **policy**: `teacher_hidden_dims` / `encoder_hidden_dims` /
  `decoder_hidden_dims` / `motion_prior_hidden_dims=[512,256,128]`,
  `latent_z_dims=32`（VQ 版换成 `num_code=2048`, `code_dim=64`），
  `activation='elu'`,
  **`teacher_a_policy_path` + `teacher_b_policy_path`（必填）**,
  **`teacher_fusion_mode: Literal["shared_encoder", "dual_encoder"]`**,
  **`teacher_a_obs_group` / `teacher_b_obs_group`**（指向 env 中的 obs group 名）
- **algorithm**: `num_learning_epochs=5`, `lr=5e-4`, `max_grad_norm=1.0`,
  `gradient_length=1`,
  **`behavior_loss_weights=(1.0, 1.0)`**, **`align_loss_coeff=0.0`**

确保字段命名与 mjlab 的 `RslRlOnPolicyRunnerCfg` 等价物兼容（参考
`src/mjlab/rl/config.py`），或单独定义新的 cfg 类型并在 `register_mjlab_task`
时透传。

### 11. Teleopit Teacher 加载 & ONNX 一致性冒烟测试
单独写一个 pytest（`tests/test_motion_prior_teacher_load.py`）：
1. 加载 `~/zcy/Teleopit/track.pt` 到 `TemporalCNNModel`，加载
   `~/zcy/Teleopit/track.onnx` 到 `onnxruntime.InferenceSession`
2. 构造 dummy 输入：`obs (1, D_actor)`, `obs_history (1, 10, D_actor)`，
   值用 `torch.randn(seed=42)` 固定
3. **PyTorch 推理**：`teacher_a(obs, obs_history)`（走 deterministic 路径）
4. **ONNX 推理**：`session.run([dummy_obs, dummy_obs_history])`
5. 断言两路输出在 `atol=1e-5` 内相等
6. 重复对 teacher_b（如有第二个 ckpt）
7. 额外：验证 encoder / decoder 等 trainable 部分 grad 正常 backward；
   verify `teacher.parameters()` 全部 `requires_grad=False`

这一步是 train 跑通的硬前置——一旦发现 obs 顺序 / norm stats / cnn key 不对，
后面所有 loss 都是无意义的。

**坑提醒**：rsl_rl 的 `EmpiricalNormalization` 包含 `running_mean` /
`running_var` / `count` 三个 buffer（非 trainable），ckpt 一定要把这些
load 进去；否则 obs_normalizer 是恒等，推理结果与 ONNX 偏差非常大。

### 12. 小规模端到端跑通蒸馏（num_envs 小，迭代少）
用 `num_envs=64`、`max_iterations=50` 跑 `Mjlab-MotionPrior-Flat-Unitree-G1`，
确认：
- env reset / step 正常
- student / teacher_a / teacher_b obs 维度匹配各自 MLP 输入
- rollout 能堆出 buffer，loss 各项（含 `behavior_a`, `behavior_b`,
  `kl_a`, `kl_b`, `ar1`, `align`）都非 NaN
- `behavior_loss` 在前 50 iter 有下降趋势
- `save_interval` 触发的 ckpt 能成功落盘并加载 resume

然后用同样规模跑 VQ 版。这一步只验流程，不追指标。

### 13. play / inference 脚本与 ONNX 导出
在 `src/mjlab/tasks/motion_prior/scripts/` 加 play 脚本（参考
`tracking/scripts/evaluate.py`），用 `policy_inference`（VAE）或
`decoder + sampled code`（VQ）做推理可视化。同时给 runner 加
`export_policy_to_onnx`，仅导出 student 推理路径
（encoder→decoder 或 mp→decoder），便于后续 sim2real 与 downstream 任务。
**双 Teacher 提示**：导出时只保留 motion_prior 头 + decoder，**不导**
teacher / encoder（部署时没有 teacher_obs）。

### 14. 文档 & changelog 更新
在 `docs/source/changelog.rst` "Upcoming version (not yet released)" 下加
`Added` 条目说明新增 `Mjlab-MotionPrior-Flat-Unitree-G1` / `-VQ` 任务及蒸馏
流程。在 `src/mjlab/tasks/motion_prior/README.md`（或对应 doc 页）记录：
- 两个 teacher ckpt 怎么传
- 双 Teacher 两种融合方案的差异与开关
- obs 组件
- loss 各项含义与超参（含 `align_loss_coeff` 的实验建议）
- 与 isaaclab 实现的差异点

最后跑 `make check` 与 `make test` 确认全绿。

---

## 用户原始草稿（保留）

先完成上述的内容，但和 ~/zcy/motionprior 不同的是，我这个工程中的网络会接收两个 Teacher，并把他们压缩到同一 motion prior 中，有两种方法：
1. 用同一个 Encoder, obs 维度的不同之处填充 -100 作为占位； 2. 两个 Teacher 用两个 Encoder，但这就要考虑这两个 encoder 输出的 mu and std 是不是要加 loss 约束其距离？ 否则会出现不同动作分布重合的情况？
(先采用两个encoder 的方式把)
