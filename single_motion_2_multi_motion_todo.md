# Single Motion → Multi Motion 接入 motion_prior 的 Todo

## 背景

motion_prior 蒸馏（task = `Mjlab-MotionPrior-Flat-Unitree-G1`）目前的 flat env
继承自 `unitree_g1_flat_tracking_env_cfg`，使用单 motion 的
`MotionCommandCfg`（`src/mjlab/tasks/tracking/mdp/commands.py`），CLI 只能传
单个 `.npz` 文件作为 motion 来源。

我们的目标只有一个：

> **让 flat env 在蒸馏时从一个 motion 目录抽样多段 motion，而不是只认一个
> .npz 文件。** rough env 不动；motion_prior 算法 / encoder / decoder 不动。

参考实现：`~/project/tracking_bfm` 的 `MultiMotionCommand` —— 它已经有：

- 目录自动扫描（`os.walk` 找所有 `.npz`）
- isaaclab → mujoco 关节 / body 重排（你的 dataset
  `/home/lenovo/g1_retargeted_data/npz/Data10k` 是 isaaclab 顺序）
- per-env `motion_idx + time_steps`，越界自动 resample
- 与单 motion `MotionCommand` 完全一致的 property 接口（`joint_pos`、
  `body_pos_w`、`anchor_*` 等）—— 下游 reward / termination / observation 零改动

不要走 `~/project/mjlab` 上游 / `~/project/Teleopit`。理由：tracking_bfm 是从
mjlab 上游精简过来的，AMP buffer / clip metadata 都已注释掉，最贴合
motion_prior 的需求。

---

## 关键参考文件（动手前先读一遍）

| 角色 | 路径 |
|---|---|
| 参考的 multi_commands 完整实现 | `~/project/tracking_bfm/src/mjlab/tasks/tracking/mdp/multi_commands.py` |
| 参考的 task config（如何把 cfg 类参数化） | `~/project/tracking_bfm/src/mjlab/tasks/tracking/config/g1/env_cfgs.py` |
| 参考的 task 注册 | `~/project/tracking_bfm/src/mjlab/tasks/tracking/config/g1/__init__.py` |
| 当前 fork 的单 motion 实现 | `src/mjlab/tasks/tracking/mdp/commands.py`（保留，不动） |
| motion_prior flat env 入口 | `src/mjlab/tasks/motion_prior/config/g1/env_cfgs.py` 的 `unitree_g1_flat_motion_prior_env_cfg` |
| train CLI 入口 | `src/mjlab/scripts/train.py`（关键判断在 `is_tracking_task` 那块） |
| 数据集 | `/home/lenovo/g1_retargeted_data/npz/Data10k/`（9840 个子目录，每个有 `motion.npz`） |

---

## 数据集 schema（已确认）

每个 `motion.npz` 包含：

```
fps: (1,) int64
joint_pos: (T, 29) float32
joint_vel: (T, 29) float32
body_pos_w: (T, 30, 3) float32
body_quat_w: (T, 30, 4) float32
body_lin_vel_w: (T, 30, 3) float32
body_ang_vel_w: (T, 30, 3) float32
```

关节 / body 顺序是 **isaaclab**，需要 `_ISAACLAB_TO_MUJOCO_*_REINDEX`
重排（tracking_bfm 已实现）。

---

## Todo（按顺序执行）

### 1. 盘点 motion_prior 下游消费 `MotionCommand` 的哪些 property

目标：搞清楚一旦换 cfg 后哪些下游会被影响。

```sh
grep -rn "command_name.*motion\|cfg.commands\[.motion.\]" src/mjlab/tasks/motion_prior/
grep -rn "anchor_lin_vel_w\|anchor_ang_vel_w\|anchor_pos_w\|anchor_quat_w\|joint_pos\|body_pos_w" src/mjlab/tasks/motion_prior/
```

特别留意 **`anchor_lin_vel_w` / `anchor_ang_vel_w`** —— tracking_bfm 的实现里
这两个 property 会带 `history_steps + 1 + future_steps` 拼接，shape 是
`(num_envs, (h+1+f)*3)`；单 motion 的实现是 `(num_envs, 3)`。**形状变了下游会崩。**

记录每个消费点 → 写在 `single_motion_migration_audit.md`（一次性产出物，
不需 commit），方便阶段 3 决定 history/future_steps 怎么设。

### 2. 整文件复制 tracking_bfm 的 multi_commands.py 到 fork

```sh
cp ~/project/tracking_bfm/src/mjlab/tasks/tracking/mdp/multi_commands.py \
   src/mjlab/tasks/tracking/mdp/multi_commands.py
```

复制后立即运行：

```sh
uv run python -c "from mjlab.tasks.tracking.mdp import multi_commands; print('ok')"
```

如果 import 失败，按报错补缺失依赖（`mjlab.utils.lab_api.math` 应该有；
其他若缺，对照 tracking_bfm 看是否漏了 import）。

**不要做的事**：

- 不要裁掉 `fall_recovery_*` 字段、`future_steps` / `history_steps` buffer。
  它们和 cfg、`_resample_command`、`_get_reference_time_steps` 深度耦合，
  强行删除会引入 bug。**用默认值禁用** 而不是 **删除**。
- 不要改 `_ISAACLAB_TO_MUJOCO_*_REINDEX` 常量。

### 3. 在 motion_prior flat env 接入 MultiMotionCommandCfg

文件：`src/mjlab/tasks/motion_prior/config/g1/env_cfgs.py`

改 `unitree_g1_flat_motion_prior_env_cfg(...)`：

- 现在它继承 `unitree_g1_flat_tracking_env_cfg`（单 motion）。
- 新做法：在拿到 `cfg` 后，**把 `cfg.commands["motion"]` 替换成
  `MultiMotionCommandCfg`** 实例，复用原来的 `anchor_body_name / body_names /
  pose_range / velocity_range / joint_position_range`，新增以下字段：

  ```python
  motion_path="",                # 由 CLI 注入
  motion_type="isaaclab",        # 你的 dataset 顺序
  history_steps=0,               # 关闭 history（保形状回 (N,3)）
  future_steps=0,                # 关闭 future（同上）
  fall_recovery_ratio=0.0,       # 关闭 fall recovery（默认就是 0，显式写）
  sampling_mode="adaptive",      # 或 "uniform"，看你训练偏好
  if_log_metrics=True,
  ```

- 导入用别名避免和老 cfg 冲突：

  ```python
  from mjlab.tasks.tracking.mdp.multi_commands import (
      MotionCommandCfg as MultiMotionCommandCfg,
  )
  ```

**rough env 函数 `unitree_g1_rough_motion_prior_env_cfg` 完全不动。**

### 4. 修 `scripts/train.py` 的兼容性

文件：`src/mjlab/scripts/train.py`

两处要改：

- **`is_tracking_task` 判断**（约 `train.py:74`）：
  现在是 `isinstance(motion_cmd, MotionCommandCfg)`，需要扩成兼容
  `MultiMotionCommandCfg`：

  ```python
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg as _SingleMotionCfg
  from mjlab.tasks.tracking.mdp.multi_commands import (
      MotionCommandCfg as _MultiMotionCfg,
  )
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
      cfg.env.commands["motion"], (_SingleMotionCfg, _MultiMotionCfg)
  )
  ```

- **motion_file / motion_path 加载分支**（紧跟 `is_tracking_task` 之后那一坨）：
  原代码假设 cfg 字段是 `motion_file`。`MultiMotionCommandCfg` 的字段是
  `motion_path`。加一个分支：若 cfg 是 `MultiMotionCfg`，跳过 wandb artifact
  / 单文件检查，直接信任 CLI 传进来的 `motion_path`（它会自然透传到 cfg）。

  最简实现：用 `hasattr(motion_cmd, "motion_path")` 判断走哪条路径。

`scripts/play.py` 同理（约 `play.py:76`），如果你想 play 也支持 multi
motion，照样改一遍；只跑 train 的话 play 可以延后。

### 5. 改 `run_distaill-mp.sh`

把：

```sh
--env.commands.motion.motion-file /home/lenovo/project/Teleopit/data/datasets/seed/train/one_motion_for_debug.npz
```

换成：

```sh
--env.commands.motion.motion-path /home/lenovo/g1_retargeted_data/npz/Data10k
```

字段名 `motion-file` → `motion-path`（tyro 会从 `motion_path: str` 自动推导）。

### 6. 冒烟测试 1：5 段 motion + num_envs=4

先建一个临时目录软链 5 段 motion，避免一上来就吃 9840 段：

```sh
mkdir -p /tmp/motions_tiny
for d in $(ls /home/lenovo/g1_retargeted_data/npz/Data10k | head -5); do
  ln -sf /home/lenovo/g1_retargeted_data/npz/Data10k/$d /tmp/motions_tiny/$d
done
```

跑一个 max-iter=2、num_envs=4 的微型训练：

```sh
uv run python -m mjlab.scripts.train Mjlab-MotionPrior-Flat-Unitree-G1 \
  --env.commands.motion.motion-path /tmp/motions_tiny \
  --env.scene.num-envs 4 --agent.secondary-num-envs 4 \
  --agent.max-iterations 2 \
  --agent.teacher-a-policy-path /home/lenovo/project/Teleopit/track.pt \
  --agent.teacher-b-policy-path /home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt
```

通过标准：

- 启动不报错
- 看到 `[MotionPrior] building secondary env` 日志
- 至少看到 1 行 `[it 1] behavior_a=... behavior_b=...`

### 7. 冒烟测试 2：完整 9840 段 + num_envs=64 + max-iter=20

正式数据集冒烟：

```sh
uv run python -m mjlab.scripts.train Mjlab-MotionPrior-Flat-Unitree-G1 \
  --env.commands.motion.motion-path /home/lenovo/g1_retargeted_data/npz/Data10k \
  --env.scene.num-envs 64 --agent.secondary-num-envs 64 \
  --agent.max-iterations 20 \
  --agent.teacher-a-policy-path /home/lenovo/project/Teleopit/track.pt \
  --agent.teacher-b-policy-path /home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt
```

监控点（开新终端）：

```sh
watch -n 1 nvidia-smi
```

通过标准：

- 启动加载 9840 段 motion 不 OOM（如果 OOM，先用 dataset 子集）
- 至少跑完 20 iter 不崩
- `behavior_a / behavior_b` 量级与单 motion 基线接近（不要求一样，几倍以内）

如果 OOM：先用 1000 段子集（`ls Data10k | head -1000` 软链到子目录）跑通，
显存优化作为后续 PR，不在本次范围。

---

## 不做的事（明确范围）

以下都和 "让 flat env 加载多 motion" 无关，不在本次范围内：

- ❌ 注册新的 `Mjlab-Tracking-Multi-Flat-Unitree-G1` task
- ❌ 显存优化（CPU 缓存按需上 GPU、fp16 motion data）
- ❌ 加载并行化（多线程 np.load）
- ❌ 启用 adaptive sampling 调参
- ❌ 改 rough env / motion_prior policy / runner / 蒸馏算法
- ❌ 给 `MotionCommand`（单 motion 老类）加 deprecation warning

---

## 风险点（动手时注意）

1. **anchor_lin_vel_w / anchor_ang_vel_w 的形状语义**
   - tracking_bfm 默认 `history_steps=5, future_steps=5`，形状是 `(N, 33)`。
   - 我们设 `history_steps=0, future_steps=0`，形状回到 `(N, 3)`。
   - 务必在阶段 1 audit 时确认 motion_prior 下游消费这俩的形状期望是
     `(N, 3)`；否则要把 history/future 改成那边的期望，或者改下游。

2. **CLI 字段名歧义**
   - 老的：`--env.commands.motion.motion-file`
   - 新的：`--env.commands.motion.motion-path`
   - 改 `run_distaill-mp.sh` 时不要漏。tyro 不会自动 fallback，错了直接报
     "unrecognized argument"。

3. **数据集大小**
   - 9840 段一次加载，单卡 24 GB 估计可能爆。先用子集起步。

4. **不要碰 motion_prior policy**
   - `motion_prior_policy.py` 里 `prop_obs_dim` 是从 env 动态推断的，多
     motion 不会改变 student obs 维度（依然 559，因为我们之前加了
     height_scan）。policy 完全无感。

---

## 验收

跑通 todo #7 的全量冒烟（max-iter=20）就算完成。提交一次 commit
（不要 amend），消息形如：

```
feat(motion_prior): support multi-motion dataset for flat distillation env

Port MultiMotionCommand from ~/project/tracking_bfm with history_steps=0
and future_steps=0 to keep anchor_*_w shapes (N, 3). Flat env now accepts
--env.commands.motion.motion-path pointing to a directory of motion.npz
files. Rough env and motion_prior policy unchanged.
```
