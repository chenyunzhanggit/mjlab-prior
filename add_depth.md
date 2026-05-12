# Add Depth Camera Support to Motion Prior Distillation

> 目标:把 motion prior 蒸馏管线中**学生侧**(student / motion_prior / decoder / downstream policy)的 `scandot`(`height_scan`)输入替换为 **深度相机图像**(depth image,`[B, 1, H, W]`)。Teacher 侧维持现状(teacher_a 不用 terrain,teacher_b 仍吃 scandot,因为 teacher 是已经训好的 frozen ckpt,改 obs 会跑不起来)。
>
> 参考实现:`/home/lenovo/project/mjlab-loco`(已有完整 depth 相机管线 + EncoderMLP 模型),把它**移植**过来,不要改 loco 自身。

---

## 决策点(请先确认,再开工)

| # | 决策 | 备选 | 我的建议 |
|---|---|---|---|
| D1 | **要替换 scandot 的位置范围** | (a) 仅 student/motion_prior 学生侧;(b) 同时改 teacher_b 让 teacher 也输入 depth;(c) 同时让 velocity rough policy(也就是未来的 teacher_b)也改吃 depth | **(a)** 最小动作面,teacher_b ckpt 不重训。如果 (c) 也要,需重训 velocity teacher |
| D2 | **Teacher_b 的 `height_scan` 输入怎么处理** | (i) 保留,继续给 teacher_b 喂 scandot(只是学生不再用);(ii) 用一个 scandot↔depth 映射伪造给 teacher_b;(iii) 在 deploy 时改成完全用 student 路径而绕开 teacher_b | **(i)**:训练时 sim 里 scandot 是免费的,蒸馏 rollout 仍能产生 teacher_b 动作作为监督信号 |
| D3 | **Encoder_b 是否也改成 depth 输入** | 是 / 否 | **是**(顺手):encoder_b 是可训练的,把它从「读 286-dim 一维 obs(含 scandot)」改成「读 prop + depth(2D)」最自然。否则 encoder_b 输入和 student 输入异构,语义不一致 |
| D4 | **Motion prior head 输入** | (a) 只 prop;(b) prop + depth latent | **(b)**:既然 student 拿到了 depth,prior 也应该能见 depth,否则 deploy 时 prior 看到的世界比 encoder 少一截 |
| D5 | **Decoder 输入** | (a) `[prop, z]`(老);(b) `[prop, depth_latent, z]` | **(a)** 保持不变。z 已经吸收了 depth 信息,decoder 不必重复 |
| D6 | **CNN 编码器是否在 encoder_b / motion_prior / decoder 之间共享权重** | 共享 / 独立 | **共享**(像 velocity 里 `share_cnn_encoders=True`):省参数,且保证 deploy 时 student 看到的 depth latent 和训练时 prior 看到的一致 |
| D7 | **是否同步给 student 加 5 步 lower-actor history**(loco 里的做法) | 加 / 不加 | **不在这次做**。先单跑 depth,history 另开一个 PR |
| D8 | **保留旧的 1D `height_scan` term 不删** | 删 / 保留(可由 flag 控制) | **flag 控制**(`use_depth=True` 时禁用 scandot term,`False` 时用老路径);保留向后兼容,方便 A/B |
| D9 | **Depth 相机的安装位置** | torso / pelvis / head | **pelvis**(对齐 loco 里的 `_DEPTH_CAMERA_CFG.frame`,翻俯角 45°),并确认 G1 MJCF 里有 pelvis body |
| D10 | **Depth 分辨率 / FOV** | 沿用 loco:raw 64×64,fovy 57.9°,crop+resize → 60×60 | 沿用 |

---

## TODO List

### Phase 0 — 基础设施移植(从 mjlab-loco → mjlab_prior)

> **不能直接 cp -r**,因为两个 repo 的依赖版本可能漂移。需要逐文件移植并跑 import smoke test。

- [ ] **0.1** 移植 `mjlab.utils.noise` 的图像噪声管线
  - 源文件:`mjlab-loco/src/mjlab/utils/noise/noise_cfg.py`(只新增 `ImageNoiseCfg / DepthNormalizationCfg / CropAndResizeCfg / DepthDistanceGaussianNoiseCfg / DepthDropoutCfg` 这五个 class)
  - 目标:`mjlab_prior/src/mjlab/utils/noise/noise_cfg.py`(追加,不动现有 `NoiseCfg` / `UniformNoiseCfg` / `GaussianNoiseCfg` 等)
  - 同步更新 `mjlab_prior/src/mjlab/utils/noise/__init__.py` 的 re-export
  - 移植 `mjlab-loco/src/mjlab/utils/noise/noise_model.py` 里和图像 pipeline 相关的部分(如有 NoiseModel 实现)
  - **验证**:`uv run python -c "from mjlab.utils.noise import DepthNormalizationCfg, CropAndResizeCfg, DepthDistanceGaussianNoiseCfg, DepthDropoutCfg"`

- [ ] **0.2** 移植 `mjlab.sensor.grouped_ray_caster` 整个子包
  - 源目录:`mjlab-loco/src/mjlab/sensor/grouped_ray_caster/`(4 个 py 文件)
  - 目标:`mjlab_prior/src/mjlab/sensor/grouped_ray_caster/`
  - 留意:`grouped_ray_caster_camera.py` 依赖 `RayCastSensor`,要确认 mjlab_prior 现有的 `raycast_sensor.py` 接口和 loco 一致(尤其是 `_compute_rays_world`、`hit_distance` 字段、`SensorContext`)
  - **验证**:`uv run python -c "from mjlab.sensor.grouped_ray_caster import GroupedRayCasterCamera, GroupedRayCasterCameraCfg"`

- [ ] **0.3** 移植 `mjlab.sensor.noisy_camera` 子包
  - 源目录:`mjlab-loco/src/mjlab/sensor/noisy_camera/`(2 个 py 文件)
  - 目标:`mjlab_prior/src/mjlab/sensor/noisy_camera/`
  - 它会调用 0.1 的 ImageNoiseCfg 和 0.2 的 GroupedRayCasterCamera
  - 同步更新 `mjlab_prior/src/mjlab/sensor/__init__.py` 加 re-export(`NoisyGroupedRayCasterCamera`、`NoisyGroupedRayCasterCameraCfg`、`ExtrinsicPerturbationCfg`、`IntrinsicPerturbationCfg`)
  - **验证**:同上,import smoke test

- [ ] **0.4** 移植 `EncoderMLPModel`(rsl_rl 侧扩展模型)
  - 源文件:`mjlab-loco/src/mjlab/rl/encoder_mlp_model.py`
  - 目标:`mjlab_prior/src/mjlab/rl/encoder_mlp_model.py`
  - 这个文件继承自 rsl_rl 的 `CNNModel`,**前提是 mjlab_prior 用的 rsl_rl 版本里有 `CNNModel` 且 `cnn_cfg` 通道存在**。先 check `uv run python -c "from rsl_rl.models.cnn_model import CNNModel; print(CNNModel.__init__.__doc__)"`。如果版本不够新,需要先升 rsl_rl
  - **验证**:`uv run python -c "from mjlab.rl.encoder_mlp_model import EncoderMLPModel"` + 走一个 dummy obs dict 看 forward 不报错

- [ ] **0.5** 移植 `depth_image` MDP observation function
  - 源文件:`mjlab-loco/src/mjlab/tasks/velocity/mdp/observations.py:52-81`
  - 目标:`mjlab_prior/src/mjlab/envs/mdp/observations.py`(放公共 mdp 下,而非只放在 velocity 任务里,因为 motion_prior 也要用)
  - 同步更新 `mjlab_prior/src/mjlab/envs/mdp/__init__.py` 的 re-export(如已存在 wildcard re-export 可跳过)
  - **验证**:`uv run python -c "from mjlab.envs.mdp import depth_image"`

---

### Phase 1 — 在 motion_prior 环境中注册 depth 相机

- [ ] **1.1** 在 `motion_prior/config/g1/env_cfgs.py` 中加 `_make_g1_depth_camera_cfg()` 工厂函数
  - 1:1 复制 `mjlab-loco/src/mjlab/tasks/velocity/config/g1/dual_env_cfgs.py:66-115` 的 `_DEPTH_CAMERA_CFG`
  - frame 选 `pelvis`,offset 沿用 loco 的设置(pos=(0.15, 0, -0.05),pitch ~45° down)
  - **决策点 D9**:确认 G1 MJCF 里有没有 pelvis body(应该有),否则改 torso_link

- [ ] **1.2** 在 `unitree_g1_flat_motion_prior_env_cfg` 中注入 depth 相机
  - 类似当前注入 `_make_g1_terrain_scan_sensor()` 的位置(line 76-78),增加 depth 相机
  - 标记可选:`use_depth: bool = True` 参数

- [ ] **1.3** 在 `unitree_g1_rough_motion_prior_env_cfg` 中注入 depth 相机
  - rough env 已经有 `terrain_scan`,现在再加 depth 相机即可(两者共存)

- [ ] **1.4** 在 `observations_cfg.py` 新增 `make_student_depth_obs_group()` factory
  - 返回 `ObservationGroupCfg(terms={"depth_image": ObservationTermCfg(func=depth_image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"})}, enable_corruption=False, concatenate_terms=True)`
  - **注意**:depth 是 4D `[B, C, H, W]`,不能和其他 1D term 同 group concat。**必须单独一个 group**

- [ ] **1.5** 在 `observations_cfg.py` 修改 `make_student_obs_group`
  - 加一个 `use_depth: bool` 参数:`True` 时**移除** `height_scan` extra term(由调用方控制是否替换,我们让调用方传 `extra_terms={}` 即可,不强改 builder)
  - 或者干脆让调用方负责:在 env_cfgs.py 里 `extra_terms={}`(不含 scandot),然后顶层加一个 `"depth": make_student_depth_obs_group()`

- [ ] **1.6** 修改两个 motion_prior env builder 的 `cfg.observations` 字典
  - **flat env**:`{"student": ..., "teacher_a": ..., "teacher_a_history": ..., "depth": make_student_depth_obs_group()}`
  - **rough env**:`{"student": ..., "teacher_b": ..., "depth": make_student_depth_obs_group()}`
  - student 的 `height_scan` extra term 在 `use_depth=True` 时 **去掉**
  - teacher_b 的 `height_scan` term 保留(决策 D2)

---

### Phase 2 — Motion prior policy 改造(VAE + VQ 两个版本)

- [ ] **2.1** `MotionPriorPolicy.__init__` 改造(`rl/policies/motion_prior_policy.py`)
  - 新增参数:`depth_shape: tuple[int, int, int]`(C, H, W),`depth_cnn_cfg: dict | None = None`
  - 新增子模块:`self.depth_cnn: CNNWithProjection`(用现有的 `CNNProjModel` 里的 `CNNWithProjection`)将 `[B, 1, 60, 60]` → `depth_latent_dim`-d latent
  - encoder_b 输入维度变化:**老**`teacher_b_cfg.actor_obs_dim`(286,含 scandot)→ **新**`prop_obs_dim_without_scandot + depth_latent_dim`(决策 D3)
  - motion_prior head 输入:**老** `prop_obs_dim` → **新** `prop_obs_dim_without_scandot + depth_latent_dim`(决策 D4)
  - decoder 输入:保持 `[prop_obs_without_scandot, z]`(决策 D5)—— 即 prop_obs 改成不含 scandot 的版本
  - **决策 D6**:depth_cnn 在 encoder_b 和 motion_prior 之间**共享**(`self.depth_cnn` 单例,前向时两边都用同一个 instance)

- [ ] **2.2** `forward_a` / `forward_b` / `policy_inference_*` 签名更新
  - 新增 `depth_image` 参数(`[B, 1, H, W]`)
  - 内部先过 `depth_latent = self.depth_cnn(depth_image)`,然后拼到 prop 和 teacher_b_obs 上(分别 for motion_prior 和 encoder_b)
  - `motion_prior_inference(prop_obs, depth_image)` 同样需要 depth

- [ ] **2.3** Encoder_a 的处理(决策 D1=a)
  - encoder_a 仍读 `teacher_a_obs`(166-dim),不需要 depth(teacher_a 是 flat env 上的 tracking,本来就没看地形)
  - **唯一例外**:如果想让 latent 在 flat / rough 两个 env 之间真正对齐,可以也把 depth 给 encoder_a 看(flat 上 depth 几乎全 max_distance)。我倾向先不加,看消融实验

- [ ] **2.4** `MotionPriorVQPolicy` 同步改造(`motion_prior_vq_policy.py`)
  - 改造点与 2.1-2.3 完全平行;quantizer 不动

- [ ] **2.5** Teacher_b 加载逻辑保持不变
  - `velocity_loader.py` / `teacher.evaluate_b` 仍读 286-dim(含 scandot)teacher obs
  - teacher_b 的 obs group `"teacher_b"` 保留 height_scan,runner 路径独立(决策 D2)

---

### Phase 3 — Motion prior runner 接线

- [ ] **3.1** `motion_prior/rl/runner.py`
  - `MotionPriorOnPolicyRunner.collect_rollouts`:在采样 obs 时多取 `depth_image = _t(obs, "depth")`,推到 buffer 里
  - `forward_a` / `forward_b` 调用处加传 `depth_image`
  - `Rollout` dataclass 新增 `depth_image_a` / `depth_image_b` 字段
  - 在更新 loss 时,`policy.forward_a(rollout["prop_obs_a"], rollout["teacher_a_obs"], rollout["depth_image_a"])`(可选,见 2.3)+ `policy.forward_b(rollout["prop_obs_b"], rollout["teacher_b_obs"], rollout["depth_image_b"])`

- [ ] **3.2** ONNX export(`runner.py` 里的 export 路径)
  - Path 3(deploy):`prop_obs + depth_image → motion_prior → decoder → action`
  - 导出的 ONNX 需要两个输入张量(`obs`、`depth`),签名要更新

- [ ] **3.3** `motion_prior/rl_cfg.py`
  - `RslRlMotionPriorPolicyCfg` 新增字段:`depth_shape: tuple[int, int, int] = (1, 60, 60)`、`depth_cnn_cfg: dict | None`、`depth_latent_dim: int = 128`
  - 同 `RslRlMotionPriorVQPolicyCfg`

- [ ] **3.4** `motion_prior/config/g1/rl_cfg.py`
  - 把 depth_cnn_cfg(同 loco `dual_rl_cfg.py:85-93`)填进去

---

### Phase 4 — Downstream policy 改造

- [ ] **4.1** `motion_prior/config/g1/downstream_env_cfgs.py`
  - `_make_policy_obs_group` 移除 `"height_scan"` term,改为单独的 `"depth"` group
  - `_make_critic_obs_group` 同步(critic 也用 depth;同时**保留** `height_scan` 作为 privileged term 给 critic—— critic 是 sim-only,scandot 比 depth 信息更稠密,这样符合 asymmetric actor-critic 的常规做法)
  - `motion_prior_obs` group 的构造:走 `make_student_obs_group(extra_terms={})` + 新增 `"motion_prior_depth"` group(或直接复用 `"depth"`)
  - **决策点(新增)D11**:downstream 的 `motion_prior_obs` 和 `policy` 是不是同一个 depth 来源?**应该是**——因为相机只有一个,且 frozen motion_prior 期望的 depth 跟 task policy 看的 depth 来自同一帧

- [ ] **4.2** `motion_prior/rl/policies/downstream_policy.py` / `downstream_vq_policy.py`
  - actor 接 `[B, 1, H, W]` depth + 1D obs:用 `EncoderMLPModel` 替换现在的 `MLPModel`(或用 `CNNProjModel`)
  - frozen motion_prior 调用处:`policy.motion_prior_inference(motion_prior_prop, motion_prior_depth)`

- [ ] **4.3** `motion_prior/rl/downstream_runner.py`
  - 类似 3.1,采 depth、塞 rollout、传给 policy

- [ ] **4.4** `motion_prior/rl_cfg.py` 中 downstream runner cfg
  - PPO actor 从 `MLPModel` 改成 `mjlab.rl.encoder_mlp_model:EncoderMLPModel`(0.4 移植过来的那个);`cnn_cfg={"depth": depth_cnn_cfg}`
  - critic 保留 MLPModel(它吃 scandot 而非 depth,见 4.1)

---

### Phase 5 — Teacher 侧的衔接(可选,延后做)

- [ ] **5.1** (可选)重训一个**用 depth 替换 scandot 的 velocity teacher**,以替代当前的 286-dim MLPModel(决策 D1=c)
  - 改 `tasks/velocity/config/g1/env_cfgs.py` 加 depth(`use_depth=True`),改 `rl_cfg.py` 用 `EncoderMLPModel + cnn_cfg`,跑 ~30k iter
  - 然后更新 `motion_prior/teacher/velocity_loader.py` 让它知道新 ckpt 是 CNN+MLP 结构
  - **本次先跳过**,沿用现有 ckpt

---

### Phase 6 — 测试 & 验证

- [ ] **6.1** Smoke test:env reset + step 不崩
  - `uv run python -c "from mjlab.tasks.motion_prior.config.g1.env_cfgs import unitree_g1_rough_motion_prior_env_cfg; cfg = unitree_g1_rough_motion_prior_env_cfg(); print(cfg.observations.keys())"`

- [ ] **6.2** Policy forward smoke test
  - 构造 dummy obs(含 1D student、teacher_b 286-d、`[1, 1, 60, 60]` depth),走 `forward_a` / `forward_b` 不报形状错

- [ ] **6.3** 单 GPU 1k iter 收敛 smoke
  - 跑 `train.py` 1000 步,看 loss 不爆 NaN,KL 不塌

- [ ] **6.4** Onnx export smoke
  - `play.py --export-onnx`,确认导出的输入签名包含 `obs` + `depth`,onnxruntime 推理一次

- [ ] **6.5** 更新 unit tests(`tests/test_motion_prior_policy.py` 等)
  - 修改 fixture 添加 dummy depth tensor
  - 修改 `test_motion_prior_onnx.py` 适配新的输入签名

---

## 需要替换 scandot → depth 的具体位置一览表

> 这是「替换」清单,不是「新增」清单。新增的 depth 相机和 obs group 已经在上面 Phase 1-4 列了。
> ✅ = 改;❌ = 保持 scandot 不动;⚠️ = 看决策

| # | 文件 | 行号 / 函数 | 当前 scandot 使用 | 是否改 depth |
|---|---|---|---|---|
| S1 | `tasks/motion_prior/observations_cfg.py` | `make_student_obs_group` 的 `extra_terms={"height_scan": ...}` 调用点 | student 1D height_scan | ✅ 改:不再传 height_scan,改为单独 depth group |
| S2 | `tasks/motion_prior/observations_cfg.py` | `make_teacher_b_obs_group` 里的 `"height_scan"` term | teacher_b 286-dim 输入的一部分 | ❌ **保持**(决策 D2):teacher 是 frozen ckpt,改 obs 就跑不起来 |
| S3 | `tasks/motion_prior/config/g1/env_cfgs.py:71-78` | flat env 注入 `_make_g1_terrain_scan_sensor()` | 给 flat env 加 scandot 让 schema 对齐 | ✅ 改:flat env 改注入 depth 相机(scandot 在 flat 上不再需要,因为 student 不读了);**保留** scandot 给 teacher_b 用?wait — teacher_b 只在 rough env 跑,flat env 上不用 teacher_b。所以 flat env 的 scandot 可以**整个删掉** |
| S4 | `tasks/motion_prior/config/g1/env_cfgs.py:110` | `extra_terms={"height_scan": ...}` (flat student) | flat student 读 scandot | ✅ 删,改 depth group |
| S5 | `tasks/motion_prior/config/g1/env_cfgs.py:138` | `extra_terms={"height_scan": ...}` (rough student) | rough student 读 scandot | ✅ 删,改 depth group |
| S6 | `tasks/motion_prior/config/g1/env_cfgs.py:140-144` | rough env 用 `make_teacher_b_obs_group(height_scan_sensor_name="terrain_scan")` | teacher_b 仍要 scandot | ❌ **保持**(S2 同理) |
| S7 | `tasks/motion_prior/config/g1/downstream_env_cfgs.py:88, 109, 146` | downstream policy / critic / motion_prior_obs 里的 `height_scan` term | downstream actor 学生侧 + critic 都吃 scandot | ✅ 改:actor 改 depth;critic **保留** scandot(privileged,asymmetric actor-critic,见 4.1) |
| S8 | `tasks/motion_prior/rl/policies/motion_prior_policy.py` 里 encoder_b 输入维度 | `teacher_b_cfg.actor_obs_dim`(286) | encoder_b 是 1D MLP 吃含 scandot 的 286-d | ⚠️ 决策 D3:**建议改**——encoder_b 输入改成 `prop(不含scandot) + depth_latent` |
| S9 | `tasks/motion_prior/rl/policies/motion_prior_policy.py` 里 motion_prior head 输入 | `prop_obs_dim` | motion_prior head 吃 prop 1D | ⚠️ 决策 D4:**建议改**——加 depth_latent |
| S10 | `tasks/velocity/config/g1/env_cfgs.py` 整个 rough env | scandot 喂 `CNNProjModel`(2D) | velocity teacher_b 训练时用的 scandot | ❌ **保持**(决策 D1=a):teacher_b 已经训好,不动 |

---

## 风险 & 注意事项

1. **rsl_rl 版本依赖**:`EncoderMLPModel` 需要 rsl_rl 里的 `CNNModel.cnn_cfg` 接口和 `MLP` / `CNN` 模块,如果 mjlab_prior 用的 rsl_rl 比较老,Phase 0.4 可能要先升级或 vendoring 一份。**开工前先 check 版本**:
   ```
   uv run python -c "import rsl_rl; from rsl_rl.models.cnn_model import CNNModel; print(rsl_rl.__version__)"
   ```

2. **shape mismatch 难调**:VAE 那条线的 prop_obs 维度被读到很多地方(`prop_obs_dim` 在 runner、policy、onnx wrapper 里都有),改一处必须同步全部。**Phase 2.1 之前先在草稿里画一张数据流图**。

3. **现有 ckpt 不兼容**:这是个**破坏性改动**,改完后老的 motion_prior ckpt 都用不了,**train 要从头跑**。在 PR 描述里要写清楚。

4. **Flat env 的 depth**:flat 平面上 depth 几乎全是 max(2.0),student 学的 latent 在 flat / rough 两个 env 上分布会更分裂(老 scandot 也有这个问题,但 depth 更明显)。需要消融:看 latent 在 flat vs rough 上的可视化(类似你上次做的 PCA/TSNE)。

5. **Deploy 端**:`deploy_mujoco/` / `deploy_real/` 不在本次范围,但要登记:他们将来需要从相机驱动里读 depth,而不是从 raycast 算高度场。

6. **训练吞吐**:depth 相机比 scandot 慢得多(raycast 数量从 11×17=187 跳到 60×60=3600,接近 20×)。`update_period=1/10`(loco 用的)可以减压,但 motion prior 默认 50Hz 控制,depth 隔 100ms 才更新 = 5 个 control step 才一帧。**留意训练速度退化**。

---

## 开工顺序建议

1. **先 Phase 0**(0.1 → 0.2 → 0.3 → 0.4 → 0.5),每移植一个文件就跑 import smoke,**确保基础设施可用**。
2. **Phase 1**(env 注册),跑 env reset/step smoke。
3. **Phase 2 + 3**(policy + runner),跑 1 iter 训练,看 forward/backward 不崩。
4. **Phase 4**(downstream),独立验证。
5. **Phase 6** 全面回归。
6. Phase 5 留到后续,看消融结果再决定要不要重训 teacher。

---

## 决策回执(请填写后告诉我开工)

| # | 决策 | 我的建议 | **你的选择** |
|---|---|---|---|
| D1 | 替换范围 | a | |
| D2 | teacher_b 是否保留 scandot | i (保留) | |
| D3 | encoder_b 是否改 depth | 是 | |
| D4 | motion_prior 是否输入 depth | 是 | |
| D5 | decoder 输入是否加 depth | 否(保持 [prop, z]) | |
| D6 | depth CNN 是否共享 | 共享 | |
| D7 | 是否同时加 5 步 history | 不在这次做 | |
| D8 | 是否保留 scandot 老路径 | 用 use_depth flag 控制 | |
| D9 | 相机挂载位置 | pelvis | |
| D10 | 分辨率 / FOV | 沿用 loco(64→60,57.9°) | |
| D11 | downstream 的 depth 是否单帧共用 | 是 | |
