# AMP 迁移到 mjlab_prior 调研报告

**目标任务**:`Mjlab-AMP-Velocity-Rough-Unitree-G1`
**源仓库**:`~/project/AMP_mjlab`(vendored rsl_rl + 自定义 AMP 任务)
**目标仓库**:`~/project/mjlab_prior`(rsl-rl-lib==5.2.0,mjlab 当前主干)
**关键约束**:rough 地形,**不**迁移 recovery / delayed-termination 机制,reset 用 mjlab 默认 `reset_root_state_uniform + reset_joints_by_offset`(即站立默认姿态)

---

## 一、差异清单(Source vs Target)

### 1. rsl_rl 版本与 API 模型(最大风险点)

| 维度 | AMP_mjlab (vendored rsl_rl) | mjlab_prior (rsl-rl-lib 5.2.0) |
|---|---|---|
| 观测数据结构 | `obs: Tensor`,`critic_obs: Tensor`,`amp_obs: Tensor`,分别通过 `extras["observations"][group]` 取出 | 统一 `TensorDict`,带 `obs_groups` 配置(`{"actor": ("actor",), "critic": ("critic",)}`)由 `resolve_obs_groups` 拼接到 actor/critic |
| 模型类 | `ActorCritic`(单类,actor+critic 都在一个 module) | `MLPModel`(actor 与 critic 是两个独立 `MLPModel` 实例),支持 `CNNProjModel` 处理 height_scan |
| Algorithm.act 签名 | `act(obs, critic_obs, amp_obs) -> action`,把 amp_obs 存入 `amp_transition.observations` | `act(obs: TensorDict) -> action`(不显式区分 critic / amp,通过 `obs_groups` 让 critic 读取自己的字段) |
| Algorithm.process_env_step | `process_env_step(rewards, dones, infos, amp_obs)` | `process_env_step(obs, rewards, dones, extras)` |
| Storage.Transition | 自定义 `RolloutStorage.Transition`,在 amp_ppo 里手动塞 `amp_transition` | `RolloutStorage.Transition`,obs 已是 TensorDict;Batch yield 不需要 hidden_states 解构 |
| save/load 格式 | v5 自定义 dict(`actor_state_dict` / `critic_state_dict` / `discriminator_state_dict` / `amp_normalizer`),`amp_ppo.save()` 返回 dict | 5.2 默认 `{actor_state_dict, critic_state_dict, optimizer_state_dict, rnd_*}`,`mjlab.rl.runner.MjlabOnPolicyRunner.load` 还做 4.x → 5.x 兼容迁移 |
| RolloutStorage 构造 | `init_storage(training_type, num_envs, num_steps, actor_obs_shape, critic_obs_shape, actions_shape, rnd_state_shape, device)` | `RolloutStorage("rl", num_envs, num_steps, obs, [num_actions], device)`(obs 是 TensorDict,自动按 group shape 分配) |
| KL 自适应 lr | 手动在 amp_ppo.update() 计算 | `actor.get_kl_divergence(old_params, new_params)` 接口 |

**结论**:**不能照搬 AMPPPO 类**,必须重写继承自 5.2 的 PPO,让 amp obs 走 `obs_groups["amp"]` 通道。

### 2. Runner 入口与注册

| 维度 | AMP_mjlab | mjlab_prior |
|---|---|---|
| Runner 注册 | `register_task_to_gym(...)` + 直接 import | `register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)` → `load_runner_cls(task_id)` 在 `scripts/train.py` 注入 |
| Runner 基类 | `AMPOnPolicyRunner(OnPolicyRunner)`(vendored) | `VelocityOnPolicyRunner(MjlabOnPolicyRunner)`(继承 5.2 的 OnPolicyRunner) |
| Env wrapper | 自带 vendored `VecEnv` wrapper | `RslRlVecEnvWrapper`,`step` 返回 `(TensorDict, rew, dones, extras)`,obs 来自 `observation_manager.compute()` |
| 训练入口 | `scripts/train.py` 自己写 | `mjlab/scripts/train.py`,通过 `tyro.cli` + registry 派发 |

**结论**:**采用方案 B(模块化适配)**——新建 `AmpVelocityOnPolicyRunner(MjlabOnPolicyRunner)`,内部组装 PPO+Discriminator,通过 `load_runner_cls` 注入。**不**复制 vendored rsl_rl。

参考实现:`src/mjlab/tasks/motion_prior/rl/runner.py:70` 中 `MotionPriorOnPolicyRunner` 就是一个**完全独立**的 runner(连 `OnPolicyRunner` 都不继承),证明 mjlab 训练入口允许任意自定义 runner,只要实现 `learn / save / load / add_git_repo_to_log` 即可。

### 3. AMP 观测组

| 维度 | AMP_mjlab | mjlab_prior |
|---|---|---|
| 配置位置 | `amp_env_cfg.py` 里第三个 `ObservationGroupCfg`,`history_length=1` | 新增到 `velocity_env_cfg`,以 `amp` 为 group 名,`history_length=1`(只需要单帧 (s, s')) |
| `history_ordering` | AMP_mjlab 打了 `mjlab_patch`,本仓库 mjlab 的 `ObservationGroupCfg` **没有** `history_ordering` 字段(实测 `grep history_ordering ~/.../observation_manager.py` → 0 hits) | **不引入这个 patch**——AMP 组只用 history_length=1,无需 ordering 选项 |
| 观测内容 | `body_pos_b / body_ori_b(6维,rot_mat 前 2 列) / body_lin_vel_b / body_ang_vel_b`,以 `torso_link` 为 anchor,对 14 个 body 取局部坐标 | 完全照搬,字段名一致,使用 `mjlab.utils.lab_api.math.{subtract_frame_transforms, matrix_from_quat, quat_apply_inverse}`(已存在) |
| AMP obs 在 runner 中的取用 | `extras["observations"]["amp"]` | `RslRlVecEnvWrapper.step` 返回 obs `TensorDict`,直接 `obs["amp"]` 即可(不需要从 extras 取) |

**关键**:`obs_groups` 配置中 actor / critic **不要**把 amp group 塞进去,只在 `runner._collect_rollout` 里从 obs TensorDict 取出 `obs["amp"]` 喂给判别器。

### 4. Motion 数据 schema

| 字段 | AMP_mjlab npz | mjlab_prior csv_to_npz | 一致 |
|---|---|---|---|
| `fps` | ✅ | ✅ | ✅ |
| `joint_pos` | ✅ | ✅ | ✅ |
| `joint_vel` | ✅ | ✅ | ✅ |
| `body_pos_w` | ✅ (num_frames, num_bodies, 3) | ✅ | ✅ |
| `body_quat_w` | ✅ (num_frames, num_bodies, 4) | ✅ | ✅ |
| `body_lin_vel_w` | ✅ | ✅ | ✅ |
| `body_ang_vel_w` | ✅ | ✅ | ✅ |

数据格式**完全兼容**。AMP_mjlab 已有的 `~/project/AMP_mjlab/src/assets/motions/g1/amp/WalkandRun/*.npz` 可直接复用——但需确认 body 顺序(npz 里 `body_pos_w[:, body_idx, :]` 的 idx 与 `get_g1_robot_cfg()` 解析出的 body 顺序是否一致)。

### 5. G1 body / joint 命名

mjlab_prior 中 `tracking/config/g1/env_cfgs.py:42-57` 列出的 14 个 body **与 AMP_mjlab 完全一致**:
```
pelvis, left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link,
torso_link, left_shoulder_roll_link, left_elbow_link, left_wrist_yaw_link,
right_shoulder_roll_link, right_elbow_link, right_wrist_yaw_link
```
anchor = `torso_link`,viewer body = `torso_link`,site_names = `("left_foot", "right_foot")`。

### 6. Reset 机制(用户明确要求简化)

| 维度 | AMP_mjlab | 本任务 |
|---|---|---|
| Reset 来源 | RSI 从 ref motion 抽帧(walk/run 池 + recovery 池) | **不做 RSI**,沿用 `velocity_env_cfg` 的 `reset_root_state_uniform`(默认站立位 + 微小扰动)+ `reset_joints_by_offset`(默认 joint pos / vel) |
| Delayed termination | `DelayedTerminationManager` 包裹 base manager | **不需要**,直接用标准 `TerminationManager` |
| Recovery 数据 | `Recovery/*.npz` | **不加载** |
| reward delay scaling | `_apply_delay_env_reward_scaling` 钩子 | **删除** |
| Motion 数据用途 | (a) reset 抽帧 (b) AMP expert | 只用于 (b) AMP expert |

**简化后好处**:
- 不需要 `MotionResetManager` 单例 / `DelayedTerminationManager` 子类
- 不需要在 reward term 里嵌入 delay-aware 逻辑
- velocity_env_cfg 的现有 reward / termination / curriculum / push_robot 全部保留

### 7. AMP Reward 注入方式

AMP_mjlab 的做法:**runner 在每个 env step 后把 AMP reward 直接累加到 PPO 的 transition.rewards**,task reward 通过 `task_reward_lerp` 线性混合(或并存)。在源码里就是:

```python
# amp_on_policy_runner.py
amp_reward, _ = self.alg.discriminator.predict_amp_reward(
    amp_obs, next_amp_obs_with_term, rewards, normalizer=...
)
# 然后 process_env_step(rewards, dones, infos, next_amp_obs_with_term)
# 里面: self.transition.rewards = rewards.clone()  ← rewards 已被 lerp
```

注意:AMP reward 与 task reward 的混合在 `discriminator.predict_amp_reward` 内部完成。

**本仓库的实施方式**(三个选项):
- **方案 R1**:在 runner 里跑判别器,把 AMP reward **加到 env 返回的 rewards** 之后再喂给 PPO。需要在 collect_rollout 里多一行 `rewards = lerp(amp_reward, rewards)`。最贴近 AMP_mjlab,**推荐**。
- **方案 R2**:把 AMP reward 注册为 RewardTermCfg,由 reward_manager 计算。问题:reward_manager 在 env.step 内部计算,无法访问 actor 之外的 discriminator;需要把 discriminator 作为 env 的一个 hook,复杂度高。**不推荐**。
- **方案 R3**:env wrapper 层加 AMP reward。可行但与 mjlab 设计哲学冲突。**不推荐**。

**结论**:用 R1。

### 8. ONNX 导出

| 维度 | AMP_mjlab | mjlab_prior |
|---|---|---|
| 导出路径 | actor-only(discriminator 不导出) | `MjlabOnPolicyRunner.export_policy_to_onnx`,`policy.as_onnx()` |
| AMP 特殊性 | 部署不需要 AMP obs(它只在训练用) | 直接复用 `VelocityOnPolicyRunner.save` 的 ONNX 导出逻辑;actor obs_groups 不含 amp,导出自然只导 actor 那部分 |

**结论**:零额外工作。

---

## 二、迁移方案决策汇总

| 决策项 | 选择 | 理由 |
|---|---|---|
| rsl_rl 迁移策略 | **方案 B**:在 mjlab 内新写 AMP 模块,适配 5.2 接口 | 避免 vendored rsl_rl 与现有 motion_prior / velocity / tracking 任务冲突 |
| AMP reward 注入 | **方案 R1**:runner 内 lerp 后塞进 PPO transition | 最贴近原实现,实现成本最低 |
| Reset 机制 | **完全沿用 velocity_env_cfg 默认 reset** | 用户明确要求,且大幅简化 |
| AMP obs group | 新增 `amp` group,history_length=1,不进 actor/critic | mjlab 5.2 obs_groups 机制天然支持 |
| Motion 数据复用 | 软链或拷贝 AMP_mjlab 的 `WalkandRun/*.npz` 到 `assets/motions/g1/amp/walk_run/` | 数据 schema 完全兼容,无需重新采集 |
| Runner 注册 | 新 task_id `Mjlab-AMP-Velocity-Rough-Unitree-G1`,`runner_cls=AmpVelocityOnPolicyRunner` | 不影响现有 `Mjlab-Velocity-Rough-Unitree-G1` |
| Discriminator 数据归一化 | 复用 AMP_mjlab 的 `Normalizer`(running mean/std) | 单文件复制即可,无依赖纠葛 |

---

## 三、新增/修改文件清单

### 新增

```
src/mjlab/tasks/amp_velocity/
├── __init__.py
├── amp_velocity_env_cfg.py        # 在 velocity_env_cfg 基础上加 amp obs group
├── mdp/
│   ├── __init__.py
│   └── amp_observations.py        # robot_body_pos_b / ori_b / lin_vel_b / ang_vel_b
├── rl/
│   ├── __init__.py
│   ├── runner.py                  # AmpVelocityOnPolicyRunner
│   ├── amp_ppo.py                 # AMPPPO(继承 rsl_rl 5.2 的 PPO)
│   ├── discriminator.py           # Discriminator(从 AMP_mjlab 拷贝,几乎无改动)
│   ├── motion_loader.py           # AMPLoader(从 AMP_mjlab 拷贝,简化掉 recovery 部分)
│   ├── replay_buffer.py           # ReplayBuffer(从 AMP_mjlab 拷贝)
│   └── normalizer.py              # Normalizer(从 AMP_mjlab 拷贝)
└── config/
    ├── __init__.py
    └── g1/
        ├── __init__.py            # register_mjlab_task
        ├── env_cfgs.py            # g1_amp_velocity_rough_env_cfg
        └── rl_cfg.py              # RslRlAmpRunnerCfg
```

### 修改

- `src/mjlab/tasks/__init__.py` — import `amp_velocity` 让 registry 生效
- `docs/source/training/rsl_rl.rst:33` 附近 — 加 `Mjlab-AMP-Velocity-Rough-Unitree-G1` 条目

### 数据资源

- 复制或软链:`AMP_mjlab/src/assets/motions/g1/amp/WalkandRun/*.npz` → `mjlab_prior/src/mjlab/assets/motions/g1/amp/walk_run/`

---

## 四、TODO List(按依赖顺序)

### Phase A:基础设施搬运(无逻辑改动)

- [ ] **A1** 建立目录骨架 `src/mjlab/tasks/amp_velocity/{,mdp/,rl/,config/g1/}` + `__init__.py`
- [ ] **A2** 复制 `discriminator.py`(AMP_mjlab → amp_velocity/rl/),保持接口不变
- [ ] **A3** 复制 `normalizer.py`(AMP_mjlab → amp_velocity/rl/)
- [ ] **A4** 复制 `replay_buffer.py`(AMP_mjlab → amp_velocity/rl/)
- [ ] **A5** 复制 `motion_loader.py` 的 `AMPLoader` 类,**删除** recovery_dir 相关参数
- [ ] **A6** 复制并轻度改造 `mdp/observations.py` → `amp_observations.py`(只保留 4 个 amp 用 obs 函数,改导入路径)

### Phase B:Env 配置层

- [ ] **B1** 写 `amp_velocity_env_cfg.py`:`make_amp_velocity_env_cfg()` 调用 `make_velocity_env_cfg()` 后追加:
  - 新增 `amp` ObservationGroupCfg(`body_pos_b / ori_b / lin_vel_b / ang_vel_b`,history_length=1,enable_corruption=False)
- [ ] **B2** 写 `config/g1/env_cfgs.py`:`g1_amp_velocity_rough_env_cfg(play=False)` 调用 B1,并:
  - 绑定 anchor = `torso_link`,body_names = 14-body 列表
  - **不**注册任何 motion reset 事件(沿用 base 的 `reset_base` / `reset_robot_joints`)
- [ ] **B3** 写 `config/g1/rl_cfg.py`:`g1_amp_velocity_ppo_runner_cfg()` 在现有 `unitree_g1_ppo_runner_cfg()` 基础上加 AMP 字段:
  - `amp_motion_dir`(指向 npz 目录)
  - `amp_reward_coef`、`amp_grad_pen_lambda`、`amp_task_reward_lerp`、`amp_replay_buffer_size`
  - `amp_discriminator_hidden=(1024, 512)`
  - `obs_groups` **保持** `{"actor": ("actor", "height"), "critic": ("critic", "height")}`(amp group 走 runner 直读,不进 obs_groups)
- [ ] **B4** 写 `config/g1/__init__.py`:`register_mjlab_task("Mjlab-AMP-Velocity-Rough-Unitree-G1", ...)`
- [ ] **B5** 在 `src/mjlab/tasks/__init__.py` 加 `from mjlab.tasks.amp_velocity.config import g1 as _amp_g1`

### Phase C:Runner / Algorithm 适配

- [ ] **C1** 写 `amp_ppo.py`:`AMPPPO(PPO)` 子类:
  - 接 5.2 的 `__init__(actor, critic, storage, ...)`,额外参数 `discriminator, amp_data, amp_normalizer, amp_replay_buffer_size, task_reward_lerp`
  - 重写 `update()`:在原 PPO mini-batch 循环里同步拉取 amp_policy / amp_expert,加 discriminator loss + grad penalty
  - 重写 `save()` / `load()` 增加 discriminator + normalizer 字段
  - 保留 `construct_algorithm` 静态方法的接口签名(供 runner 调用)
- [ ] **C2** 写 `runner.py`:`AmpVelocityOnPolicyRunner(MjlabOnPolicyRunner)`:
  - `__init__` 中读 `train_cfg["amp_*"]` 字段,加载 `AMPLoader`,构造 `Discriminator + Normalizer`,把它们注入 AMPPPO
  - 重写 `learn()`:沿用基类骨架,但在 rollout 内加:
    1. `amp_obs = obs["amp"]`(从 TensorDict 取)
    2. step 后 `next_amp_obs = obs["amp"]`,在 done 的 env 上 `next_amp_obs_with_term[reset_ids] = amp_obs[reset_ids]`
    3. `amp_reward, _ = discriminator.predict_amp_reward(amp_obs, next_amp_obs_with_term, rewards, normalizer)`
    4. `lerp_rewards = (1-α) * amp_reward + α * rewards`,送入 `alg.process_env_step`
    5. 把 `(amp_obs, next_amp_obs_with_term)` 推入 `alg.amp_storage`
  - `save / load` 继承父类 + 加 discriminator 字段(参考 motion_prior runner 的写法)
  - `export_policy_to_onnx` 直接继承(actor 不含 amp 输入,无需改动)
- [ ] **C3** 配置类 `RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg)` 加 AMP 专用字段(放在 `src/mjlab/tasks/amp_velocity/rl/__init__.py` 或独立 `cfg.py`)

### Phase D:Motion 数据接入

- [ ] **D1** 将 `~/project/AMP_mjlab/src/assets/motions/g1/amp/WalkandRun/` 拷贝到 `~/project/mjlab_prior/src/mjlab/assets/motions/g1/amp/walk_run/`(或建软链)
- [ ] **D2** 验证 body 顺序:写一段一次性脚本,加载第一个 npz,打印 `body_pos_w.shape[1]`(应等于 G1 总 body 数),并验证 idx=0 是 pelvis,idx for torso_link 一致
- [ ] **D3** `rl_cfg.py` 里 `amp_motion_dir` 默认指向 D1 的绝对路径(用 `pathlib` 拼)

### Phase E:跑通端到端

- [ ] **E1** Smoke test:`uv run train Mjlab-AMP-Velocity-Rough-Unitree-G1 --env.scene.num-envs=64 --agent.max-iterations=10 --gpu-ids=[0]`,验证不崩
- [ ] **E2** 检查 logs:`amp_loss`、`amp_grad_pen`、`amp_expert_pred → +1`、`amp_policy_pred → -1`、`mean_amp_reward` 都应在 tensorboard 出现且数值合理
- [ ] **E3** ONNX 导出 smoke:在 E1 的 save 路径下确认 `.onnx` 文件生成且能被 `onnxruntime` 加载
- [ ] **E4** Play 测试:`uv run play Mjlab-AMP-Velocity-Rough-Unitree-G1 --checkpoint-file ...`,确保 policy 能跑(不卡死)

### Phase F:训练验证与调参

- [ ] **F1** 全量训练:`num_envs=4096`,max_iterations=30_000(参考 velocity 默认)
- [ ] **F2** 对比基线:`Mjlab-Velocity-Rough-Unitree-G1`(同种子 / 步数)的 episode reward / 行为质量
- [ ] **F3** 调整 `amp_task_reward_lerp`(0.0 / 0.3 / 0.5)看风格 vs 任务性能的折衷
- [ ] **F4** 调整 `amp_reward_coef`(0.5 / 1.0 / 2.0)
- [ ] **F5** 若 discriminator 过强(`expert_pred` 长期接近 +1 且 `policy_pred` 接近 -1 不收敛),降低 lr 或增加 grad_pen lambda

### Phase G:文档与收尾

- [ ] **G1** 在 `docs/source/training/rsl_rl.rst:33` 附近加 `Mjlab-AMP-Velocity-Rough-Unitree-G1` 任务卡片,说明 motion 数据准备 + lerp 参数语义
- [ ] **G2** `CHANGELOG.md` 加 `[1.4.0] - 2026-05-14` 条目(Features / Design Rationale / Notes)
- [ ] **G3** `make check` 跑通(format / lint / type)
- [ ] **G4** 单文件单元测试:`tests/test_amp_velocity.py`,至少覆盖 `AMPLoader` 加载、`Discriminator.forward / predict_amp_reward` 形状、`AMPPPO.update` 在 toy storage 上的一次迭代不报错

---

## 五、风险与开放问题

1. **5.2 PPO 的 KL 自适应 lr 与 AMP loss 共用同一 optimizer**——本仓库 PPO 一个 optimizer 同时管 actor+critic,AMP_mjlab 是同一 optimizer 还管了 discriminator(`weight_decay` 分组)。需要决定:
   - **A**:沿用单 optimizer 多参数组(weight_decay 分组),保持原行为
   - **B**:discriminator 单独一个 optimizer(类似 RND 的处理),代码更清晰
   推荐 **B**,放进 todo `C1` 实施。

2. **obs_groups 中是否暴露 amp**——若放进 `obs_groups`,会被 actor / critic obs normalization 影响;若不放,runner 直接 `obs["amp"]` 读原始(未归一化)值,然后用 AMP 自己的 `Normalizer` 归一化。**推荐后者**(与 AMP_mjlab 一致)。

3. **`amp` obs group 是否需要 enable_corruption**——AMP_mjlab 设为 False(critic-style),本任务沿用 False。

4. **body 顺序兼容性**——AMP_mjlab 的 npz 中 `body_pos_w[:, idx, :]` 的 idx 顺序由当时仿真器解析模型时的顺序决定;mjlab_prior 加载同一份 g1 XML 后顺序**理应**一致,但仍需 D2 的 sanity check 验证。若不一致,需在 `AMPLoader._load_dir` 里加一个 reorder map。

5. **height_scan 与 AMP 的兼容性**——velocity_env_cfg 默认有 `height_scan` CNN encoder。AMP_mjlab 的 actor obs 不含 height_scan(只在 critic 用);本任务 actor 仍带 height_scan(rough 地形需要)。这不会影响 AMP——discriminator 只看 amp group,与 actor obs 完全解耦。

6. **`num_envs` 与 replay buffer 大小**——AMP 的 amp_storage 是 ReplayBuffer(size=100k 默认),当 num_envs * num_steps 远小于 replay_buffer_size 时,前几个 iter 可能采不到足够样本。需在 `C2` 里检查 `amp_storage.feed_forward_generator` 是否对小 buffer 鲁棒。

7. **AMP_mjlab 的 motion 数据是 23dof G1**——确认本仓库 `get_g1_robot_cfg()` 也是 23dof(已通过 `g1_constants_bp.py` 命名暗示),否则 joint_pos 维度不匹配。

---

## 六、工作量预估

| 阶段 | 时长 | 备注 |
|---|---|---|
| Phase A | 0.5h | 纯文件拷贝 |
| Phase B | 1h | 配置层 |
| Phase C | 4–6h | 核心适配,主要难度集中在 AMPPPO.update 重写 |
| Phase D | 0.5h | 数据软链 + sanity check |
| Phase E | 1h | smoke test + 修小 bug |
| Phase F | 1–3 天 | 训练 + 调参(GPU 时间) |
| Phase G | 1h | 文档收尾 |

总计:**代码 8–10 小时,训练验证 1–3 天**。
