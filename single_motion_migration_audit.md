# single → multi motion: motion_prior 下游消费 audit

不 commit。仅供 todo #2/#3 实施时对照"契约不能破"。

## 1. `cfg.commands["motion"]` 类型识别点

| 文件 | 行 | 用途 |
|---|---|---|
| `src/mjlab/tasks/motion_prior/config/g1/env_cfgs.py` | 68 | 构造 teacher_a obs group 时取 `command_name="motion"` |
| `src/mjlab/scripts/train.py` | ~74 | `is_tracking_task` isinstance 判断 + motion_file 注入分支 |

motion_prior 自身代码里**只有一处** `command_name="motion"`（在 obs 配置里），所以替换 `cfg.commands["motion"]` 不会触发其他改动。

## 2. anchor_*_w / robot_anchor_*_w 消费点

| 文件 | 行 | 表达式 | 期望 shape |
|---|---|---|---|
| `tasks/tracking/mdp/rewards.py` | 31 | `torch.square(command.anchor_pos_w - command.robot_anchor_pos_w)` 后 `dim=-1` | `(N, 3)` |
| `tasks/tracking/mdp/rewards.py` | 40 | `quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)` | `(N, 4)` |
| `tasks/tracking/mdp/terminations.py` | 23 | `torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1)` | `(N, 3)` |
| `tasks/tracking/mdp/terminations.py` | 32 | `command.anchor_pos_w[:, -1]`（取 z 分量） | `(N, 3)` |
| `tasks/tracking/mdp/terminations.py` | 44 | `quat_apply(... , command.anchor_quat_w, ...)` | `(N, 4)` |
| `tasks/tracking/mdp/observations.py` | 22-25 | `subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w)` | pos `(N, 3)` quat `(N, 4)` |
| `tasks/tracking/mdp/metrics.py` | 31 | `command.anchor_pos_w.unsqueeze(1)` 注释 `(num_envs, 1, 3)` | `(N, 3)` |
| `tasks/motion_prior/mdp/observations.py` | 33 | `quat_apply_inverse(robot_anchor_quat_w, anchor_lin_vel_w)` | quat `(N, 4)`, vec `(N, 3)` |
| `tasks/motion_prior/mdp/observations.py` | 38 | `quat_apply_inverse(robot_anchor_quat_w, anchor_ang_vel_w)` | 同上 |
| `tasks/motion_prior/mdp/observations.py` | 44 | `quat_apply_inverse(anchor_quat_w, asset.data.gravity_vec_w)` | quat `(N, 4)`, vec `(num_envs, 3)` |

**结论**：所有 `anchor_*_w` 消费点期望都是 `(N, 3)`（pos / lin_vel / ang_vel）或 `(N, 4)`（quat）。**契约成立**，可以放心走 todo #2 的"`anchor_*_w` shape 是 (N, 3)"路线。

## 3. body_*_w / joint_pos / joint_vel / command 消费点

| 文件 | 行 | 表达式 | 期望 shape |
|---|---|---|---|
| `tasks/tracking/mdp/observations.py` | 47-67 | `subtract_frame_transforms` 用 `robot_body_pos_w` / `robot_body_quat_w` | `(N, num_bodies, 3)` / `(N, num_bodies, 4)` |
| `tasks/motion_prior/observations_cfg.py` | 128 | `mdp.generated_commands` 拿到 `command` 属性 | `(N, 2*num_joints)` (cat joint_pos/vel) |

`command` 属性继续走 `cat([joint_pos, joint_vel])`（单 motion 现状），multi motion 实现要保留这个语义。

## 4. 与 todo "接口契约" 表对账

| Property | 期望 (todo) | 实测消费 | 一致 |
|---|---|---|---|
| `command` | `(N, 2*num_joints)` | obs 直接返回 → `(N, 58)` for G1 | ✓ |
| `joint_pos` / `joint_vel` | `(N, num_joints)` | `MotionCommand.command` 内部 cat | ✓ |
| `body_pos_w` / `body_quat_w` / `body_lin_vel_w` / `body_ang_vel_w` | `(N, num_bodies, 3/4)` | tracking observations.py 的 robot_body_*_w 镜像消费 | ✓ |
| `anchor_pos_w` / `anchor_quat_w` | `(N, 3)` / `(N, 4)` | rewards / terminations / observations / metrics | ✓ |
| `anchor_lin_vel_w` / `anchor_ang_vel_w` | `(N, 3)` | motion_prior obs `ref_base_*_b` | ✓ |
| `robot_*` | 同上 | tracking observations 用 robot_body_pos_w / robot_anchor_pos_w / robot_anchor_quat_w | ✓ |

## 5. 风险点 #2 验证结果

> "anchor_*_w shape 不是 (N, 3) → 下游 reward / obs 维度错"

**已确认所有消费点期望均为 (N, 3) / (N, 4)**。可继续执行 todo #2，并在实现时严格保证 anchor_*_w 不带 future / history 维度。
