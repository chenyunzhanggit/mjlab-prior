# Single Motion → Multi Motion 接入 motion_prior 的 Todo

## 背景

motion_prior 蒸馏（task = `Mjlab-MotionPrior-Flat-Unitree-G1`）的 flat env
当前用单 motion 的 `MotionCommandCfg`（`src/mjlab/tasks/tracking/mdp/commands.py`），
CLI 只接受单个 `.npz` 文件。

我们要做的事只有一件：

> **让 flat env 在蒸馏时能从一个 motion 目录加载多段 motion。**
> rough env 不动；motion_prior 算法 / encoder / decoder 不动。

---

## 参考实现：`~/project/motionprior`

主参考是 `~/project/motionprior`，它本身就是吃 `Data10k` 这种目录结构的项目，
multi-motion 命令实现成熟、已在生产中跑通。

**唯一要嫁接的是 `~/project/tracking_bfm` 中的 isaaclab → mujoco 关节/body
reindex 常量**，因为：

- `~/project/motionprior` 跑在 isaaclab 仿真器上，关节顺序是 isaaclab，
  loader 不做 reindex。
- 我们 `mjlab_prior` 跑在 mujoco_warp 上，关节顺序是 mujoco，
  必须把 dataset 里的 isaaclab 顺序映射到 mujoco。
- `~/project/tracking_bfm` 已经定义了完整的 reindex 常量
  `_ISAACLAB_TO_MUJOCO_JOINT_REINDEX` / `_ISAACLAB_TO_MUJOCO_BODY_REINDEX`，
  以及 isaaclab / mujoco 关节和 body 名字列表。

参考文件：

| 角色 | 路径 |
|---|---|
| 主参考：multi-motion 命令实现（含 buffer / sampling / `_resample_command`） | `~/project/motionprior/source/whole_body_tracking/whole_body_tracking/mdp/commands_multi_ada.py` |
| 主参考：目录扫描成 `motion_files` 的 helper | `~/project/motionprior/scripts/env_based_rsl_rl/play_mp.py:65-97` (`get_data10K_motion_files`) |
| 反 hard-code 的 reindex 常量 + 名字列表 | `~/project/tracking_bfm/src/mjlab/tasks/tracking/mdp/multi_commands.py:32-167` |
| 反 hard-code 的 reindex 写法（`MotionLoader.__init__` 里如何应用） | `~/project/tracking_bfm/src/mjlab/tasks/tracking/mdp/multi_commands.py:170-211` |
| 我们的单 motion 实现（保留不动） | `src/mjlab/tasks/tracking/mdp/commands.py` |
| motion_prior flat env 入口 | `src/mjlab/tasks/motion_prior/config/g1/env_cfgs.py` 的 `unitree_g1_flat_motion_prior_env_cfg` |
| train CLI 入口 | `src/mjlab/scripts/train.py` |
| 数据集 | `/home/lenovo/g1_retargeted_data/npz/Data10k/`（9840 段） |

**两个不要参考**：

- ❌ `~/project/mjlab` 上游：AMP buffer / fall_recovery / clip metadata 太重
- ❌ `~/project/tracking_bfm` 的 `MultiMotionCommand` 完整实现：
  > 仅借它的 reindex 常量。它的 `MultiMotionCommand` 与 motionprior 的实现
  > 重叠度高、且带 history_steps / future_steps 等 motion_prior 不需要的字段。

---

## 数据集 schema

`/home/lenovo/g1_retargeted_data/npz/Data10k/<clip_name>/motion.npz`：

```
fps: (1,) int64
joint_pos: (T, 29) float32
joint_vel: (T, 29) float32
body_pos_w: (T, 30, 3) float32
body_quat_w: (T, 30, 4) float32
body_lin_vel_w: (T, 30, 3) float32
body_ang_vel_w: (T, 30, 3) float32
```

**关节 / body 顺序是 isaaclab**，必须 reindex。

---

## 接口契约（这个错了下游就崩）

新写的 `MultiMotionCommand` 必须暴露与现有 `MotionCommand`
（`src/mjlab/tasks/tracking/mdp/commands.py:63-260`）**完全一致**的 property：

| Property | 形状 | 语义 |
|---|---|---|
| `command` | `(N, 2*num_joints)` | `cat([joint_pos, joint_vel])` |
| `joint_pos` | `(N, num_joints)` | 当前帧关节角 |
| `joint_vel` | `(N, num_joints)` | 当前帧关节速度 |
| `body_pos_w` | `(N, num_bodies, 3)` | 当前帧 body 位置 + `env_origins` |
| `body_quat_w` | `(N, num_bodies, 4)` | 当前帧 body 朝向 |
| `body_lin_vel_w` | `(N, num_bodies, 3)` | 当前帧 body 线速度 |
| `body_ang_vel_w` | `(N, num_bodies, 3)` | 当前帧 body 角速度 |
| `anchor_pos_w` | `(N, 3)` | 当前帧 anchor body 位置 + `env_origins` |
| `anchor_quat_w` | `(N, 4)` | 当前帧 anchor 朝向 |
| `anchor_lin_vel_w` | `(N, 3)` | 当前帧 anchor 线速度 |
| `anchor_ang_vel_w` | `(N, 3)` | 当前帧 anchor 角速度 |
| `robot_*` 系列 | 同上 | 直接转发 `self.robot.data.*` |

**关键**：anchor_lin_vel_w / anchor_ang_vel_w 必须是 `(N, 3)` 不带 history /
future。motionprior 那份代码默认 `future_steps=1` → `motion_anchor_pos`
等 future 系列才有 `(N, 3*K)`，**纯 anchor_*_w 仍是 `(N, 3)`**，符合契约。
我们沿用这个语义，**不引入** `motion_joint_pos / motion_anchor_pos` 等 future
property（motion_prior 下游不消费它们）。

---

## Todo（按顺序执行）

### 1. 盘点 motion_prior 下游消费 `MotionCommand` 的哪些 property

```sh
grep -rn 'command_name="motion"\|cfg.commands\["motion"\]' src/mjlab/tasks/motion_prior/
grep -rn 'anchor_lin_vel_w\|anchor_ang_vel_w\|anchor_pos_w\|anchor_quat_w' src/mjlab/tasks/motion_prior/ src/mjlab/tasks/tracking/mdp/rewards.py src/mjlab/tasks/tracking/mdp/terminations.py src/mjlab/tasks/tracking/mdp/observations.py
```

写一份 `single_motion_migration_audit.md`（不 commit），记录每个消费点
+ 期望 shape。**确认全部 anchor_*_w 期望都是 `(N, 3)`** —— 这是契约能成立的前提。

如果发现期望不是 `(N, 3)`，立刻停下来回我，不要自行扩展实现。

### 2. 新建 `src/mjlab/tasks/tracking/mdp/multi_commands.py`

**整体策略**：从 `~/project/motionprior` 移植，把 isaaclab 调用改成 mjlab 调用，
插入 reindex 逻辑。**不要整文件复制**（它带 isaaclab import、buffer 太重等问题）。
**逐段移植**，每移一段 import 检查一次。

按这个顺序写：

#### 2.1 顶部 import 与 reindex 常量

从 `~/project/tracking_bfm/src/mjlab/tasks/tracking/mdp/multi_commands.py:32-167`
**完整复制** 4 个名字列表 + 2 个 reindex 常量：

```python
_ISAACLAB_JOINT_NAMES = [...]
_MUJOCO_JOINT_NAMES   = [...]
_ISAACLAB_BODY_NAMES  = [...]
_MUJOCO_BODY_NAMES    = [...]
_ISAACLAB_TO_MUJOCO_JOINT_REINDEX = [_ISAACLAB_JOINT_NAMES.index(n) for n in _MUJOCO_JOINT_NAMES]
_ISAACLAB_TO_MUJOCO_BODY_REINDEX  = [_ISAACLAB_BODY_NAMES.index(n) for n in _MUJOCO_BODY_NAMES]
```

import 用 mjlab 的版本（参考现有 `commands.py` 顶部）：

```python
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply, quat_error_magnitude, quat_from_euler_xyz,
    quat_inv, quat_mul, sample_uniform, yaw_quat,
)
```

#### 2.2 `MultiMotionLoader` 类

从 `~/project/motionprior/.../commands_multi_ada.py:79-166` 移植 `MultiMotionLoader`，
做 3 处修改：

- **加 `motion_type` 参数**：默认 `"isaaclab"`。
- **每读完一段 motion 就 reindex**（参考 `tracking_bfm:202-209`）：

  ```python
  if motion_type == "isaaclab":
      jp = jp[:, _ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
      jv = jv[:, _ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
      bp = bp[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]
      bq = bq[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]
      blv = blv[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]
      bav = bav[:, _ISAACLAB_TO_MUJOCO_BODY_REINDEX, :]
  ```

- **`body_indexes` 用 `torch.Tensor`**（与现有 `commands.py:55-59` 对齐），
  motionprior 用的是 list[int]。

`get_motion_data_batch` 保留不动，按原 motionprior 实现走。

**不要移植** `HybridMultiMotionLoader`（用不上）。

#### 2.3 `MultiMotionCommand` 类

从 `~/project/motionprior/.../commands_multi_ada.py:275` 起的 `MotionCommand`
移植，只保留 motion_prior 蒸馏需要的部分：

**保留**：

- `__init__`（loader、buffer、time_steps、motion_idx、motion_length、metrics）
- `_init_buffers` / `_update_buffers`（buffer 是 multi-motion 的核心机制，必须留）
- `joint_pos / joint_vel / body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w`
  这 6 个 property
- `anchor_pos_w / anchor_quat_w / anchor_lin_vel_w / anchor_ang_vel_w` 4 个
  property，**确保 shape 是 `(N, 3)` / `(N, 4)`**
- `robot_*` 转发 property
- `_update_metrics`（按现有 motion_prior 用到的项保留）
- `_resample_command`（对应 RSI / motion 抽样）
- `_update_command`（time_steps += 1，越界 resample）
- `update_relative_body_poses`（如果 motion_prior 下游 termination 用到）
- `command` property
- 必要的 `__init__` 字段：`motion`、`time_steps`、`motion_idx`、`motion_length`、
  `buffer_*`、`body_pos_relative_w`、`body_quat_relative_w`

**不要移植**：

- `motion_joint_pos / motion_joint_vel / motion_anchor_pos / motion_anchor_quat`
  这一套 future N-step lookahead property（motion_prior 不消费）
- adaptive sampling 复杂逻辑（`enable_adaptive_sampling=False` 默认走
  `_uniform_sampling` 即可；`_adaptive_sampling` 函数体可以留但默认不启用）
- `random_static_prob` 相关分支（默认 0 即关闭）
- ghost / debug visualizer 相关（如果 base class 不要求，不写；要求就留空实现）
- isaaclab API：`Articulation` → 改用 `mjlab.entity.Entity`；
  `write_joint_state_to_sim` / `write_root_state_to_sim` → 用 mjlab 现有
  `commands.py:447-477` 的 `_write_reference_state_to_sim` 实现

#### 2.4 `MultiMotionCommandCfg`

参考 `~/project/motionprior/.../commands_multi_ada.py:1015-1051`，
但**用 mjlab 的 dataclass 风格**（参考现有 `commands.py:581-605` 的 `MotionCommandCfg`）：

```python
@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
    motion_files: list[str] = field(default_factory=list)
    motion_path: str = ""           # 二选一: 优先 motion_path
    motion_type: Literal["isaaclab", "mujoco"] = "isaaclab"
    entity_name: str
    anchor_body_name: str
    body_names: tuple[str, ...]
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    enable_adaptive_sampling: bool = False
    start_from_zero_step: bool = False
    if_log_metrics: bool = True
    # 不写 future_steps / random_static_prob / 各种 adaptive_* —— 暂不需要
    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
```

**支持 `motion_path` 字段**：在 `MultiMotionCommand.__init__` 起手处加目录展开，
直接复制 `play_mp.py:65-97` 的 `get_data10K_motion_files` 逻辑（用
`glob.glob(pattern, recursive=True)`），结果赋给 `cfg.motion_files`：

```python
if cfg.motion_path:
    assert not cfg.motion_files, "motion_path 与 motion_files 不能同时指定"
    cfg.motion_files = _expand_motion_dir(cfg.motion_path)
```

#### 2.5 末尾导出别名

参考 `~/project/tracking_bfm/.../multi_commands.py:1559-1561`：

```python
# 与单 motion 公共接口对齐 — 允许 train.py / play.py 用 isinstance 同时识别
MotionCommand = MultiMotionCommand
MotionCommandCfg = MultiMotionCommandCfg
```

### 3. 在 motion_prior flat env 接入 `MultiMotionCommandCfg`

文件：`src/mjlab/tasks/motion_prior/config/g1/env_cfgs.py` 的
`unitree_g1_flat_motion_prior_env_cfg`。

它现在继承 `unitree_g1_flat_tracking_env_cfg`，那个函数 hard-coded
单 motion 的 `MotionCommandCfg`。**不要去改 tracking task 自己**——
在拿到 `cfg` 之后，**就地替换** `cfg.commands["motion"]` 为
`MultiMotionCommandCfg`：

```python
from mjlab.tasks.tracking.mdp.multi_commands import (
    MotionCommandCfg as MultiMotionCommandCfg,
)

# 在拿到 cfg 之后、设置 obs 之前
old_motion = cfg.commands["motion"]
cfg.commands["motion"] = MultiMotionCommandCfg(
    entity_name=old_motion.entity_name,
    anchor_body_name=old_motion.anchor_body_name,
    body_names=old_motion.body_names,
    pose_range=old_motion.pose_range,
    velocity_range=old_motion.velocity_range,
    joint_position_range=old_motion.joint_position_range,
    motion_path="",            # 由 CLI 注入
    motion_type="isaaclab",
    enable_adaptive_sampling=False,
    start_from_zero_step=False,
    if_log_metrics=True,
)
```

**rough env 函数完全不动。**

### 4. 修 `scripts/train.py` 的兼容性

文件：`src/mjlab/scripts/train.py`。

两处要改：

#### 4.1 `is_tracking_task` isinstance 判断

约 `train.py:74`，改为同时识别两种 cfg：

```python
from mjlab.tasks.tracking.mdp import MotionCommandCfg as _SingleMotionCfg
from mjlab.tasks.tracking.mdp.multi_commands import (
    MotionCommandCfg as _MultiMotionCfg,
)

is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], (_SingleMotionCfg, _MultiMotionCfg)
)
```

#### 4.2 motion_file / motion_path 的 CLI 注入

紧跟 `is_tracking_task` 判断那一坨 motion_file 处理逻辑：

- 若 cfg 是 `_MultiMotionCfg`（用 `hasattr(motion_cmd, "motion_path")` 判断）：
  - 直接信任 CLI 传进来的 `motion_path`（tyro 已经把 CLI 值注入到 cfg），
    跳过 wandb artifact 那套
  - 如果 `motion_path` 为空字符串，报错提示用户传
    `--env.commands.motion.motion-path`
- 若 cfg 是 `_SingleMotionCfg`：保持原行为不动

`scripts/play.py` 暂不改（只跑 train 即可）。

### 5. 改 `run_distaill-mp.sh`

把：

```sh
--env.commands.motion.motion-file <single.npz>
```

换成：

```sh
--env.commands.motion.motion-path /home/lenovo/g1_retargeted_data/npz/Data10k
```

### 6. 冒烟测试 1：5 段 motion + num_envs=4

```sh
mkdir -p /tmp/motions_tiny
for d in $(ls /home/lenovo/g1_retargeted_data/npz/Data10k | head -5); do
    ln -sf /home/lenovo/g1_retargeted_data/npz/Data10k/$d /tmp/motions_tiny/$d
done

uv run python -m mjlab.scripts.train Mjlab-MotionPrior-Flat-Unitree-G1 \
    --env.commands.motion.motion-path /tmp/motions_tiny \
    --env.scene.num-envs 4 --agent.secondary-num-envs 4 \
    --agent.max-iterations 2 \
    --agent.teacher-a-policy-path /home/lenovo/project/Teleopit/track.pt \
    --agent.teacher-b-policy-path /home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt
```

通过标准：

- 启动不报错，能看到 `[MotionPrior] building secondary env` 日志
- 看到至少 1 行 `[it 1] behavior_a=... behavior_b=...`
- log 里 `len_a` > 0（说明 motion 加载到 buffer，agent 在动）

### 7. 冒烟测试 2：完整 9840 段 + num_envs=64 + max-iter=20

```sh
uv run python -m mjlab.scripts.train Mjlab-MotionPrior-Flat-Unitree-G1 \
    --env.commands.motion.motion-path /home/lenovo/g1_retragetted_data/npz/Data10k \
    --env.scene.num-envs 64 --agent.secondary-num-envs 64 \
    --agent.max-iterations 20 \
    --agent.teacher-a-policy-path /home/lenovo/project/Teleopit/track.pt \
    --agent.teacher-b-policy-path /home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt
```

通过标准：

- 9840 段加载不 OOM。
- 跑完 20 iter 不崩。
- `behavior_a / behavior_b` 量级与单 motion 基线接近（几倍以内）。

如果 OOM：用 1000 段子集（`ls Data10k | head -1000` 软链）跑通即可。
**不要在本次范围内做显存优化**。

---

## 不做的事（明确范围）

- ❌ 注册新的 tracking-multi task
- ❌ 显存优化（CPU 缓存按需上 GPU、fp16）
- ❌ np.load 并行化
- ❌ `random_static_prob` / `motion_files_hybrid` / `HybridMultiMotionLoader`
- ❌ `motion_joint_pos` / `motion_anchor_pos` 等 future N-step property
- ❌ 改 rough env / motion_prior policy / runner / 蒸馏算法
- ❌ 改 play.py
- ❌ deprecate / 删除单 motion 老 `MotionCommand`

---

## 风险点

1. **isaaclab → mujoco reindex 错位 → 训练 silent 失败**
   - 关节 / body 顺序错了不会立刻 crash，但 teacher_a 输入分布会全错，
     `behavior_a` 会爆炸。冒烟时**手动 spot-check**：`MultiMotionLoader` 加载
     第一段 motion 后，比较 `loader.joint_pos[0]` 和原始 npz 的 `joint_pos[0]`
     是否在 reindex 后与机器人 `self.robot.joint_names` 对齐。

2. **`anchor_*_w` shape 不是 `(N, 3)` → 下游 reward / obs 维度错**
   - 必须在阶段 1 audit 里确认。

3. **dataset 9840 段 + num_envs=64 显存预算**
   - 单卡 24 GB 估计可能爆。先用子集起步，**不要现场优化**。

4. **CLI 字段名变更**
   - 老：`--env.commands.motion.motion-file`
   - 新：`--env.commands.motion.motion-path`
   - 老命令会报 unrecognized argument。

5. **`build()` 里目录展开时机**
   - cfg 实例化时 `motion_path` 可能还是空（CLI 还没注入），所以**不要在 cfg
     `__post_init__` 里展开**。展开必须在 `MultiMotionCommand.__init__` 起手处，
     此时 CLI 值已经注入到 cfg。

---

## 验收

跑通 todo #6（小冒烟）+ todo #7 至少能起来 5 iter 即可。
**一次性 commit**，消息：

```
feat(motion_prior): support multi-motion dataset for flat distillation env

Port MultiMotionCommand from ~/project/motionprior with isaaclab→mujoco
reindex constants from ~/project/tracking_bfm. Flat env now accepts
--env.commands.motion.motion-path pointing to a directory of motion.npz
files (defaults to isaaclab joint/body order). Rough env, motion_prior
policy, and single-motion MotionCommand all unchanged.
```
