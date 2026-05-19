# Depth Pipeline Migration: 把 mjlab-prior-main 的 NoisyGroupedRayCasterCamera 迁到 mjlab-prior

> **目标**:把 mjlab-prior-main 已经实战验证的深度相机 pipeline(`NoisyGroupedRayCasterCamera` + `OffsetCfg` + 4-step noise pipeline + per-episode extrinsic / intrinsic perturbation + history buffer)整套搬到 mjlab-prior,**替换**当前 perception passing 任务的简化实现(`ForwardPinholeCameraPatternCfg` + obs-side `_apply_random_rect_masks` mask augment)。
>
> 参考实现:`/Users/lhy/isaaclab2mjlab/mjlab-prior-main`。**不能直接 cp -r**,因为两个 repo 的 `RayCastSensor` / `Sensor` / `Entity` API 可能有版本漂移,需要逐文件移植 + import smoke test。

---

## 决策点(请填回执后再开工)

| # | 决策 | 备选 | 我的建议 |  **你的选择** |
|---|---|---|---|---|
| D1 | **是否完全替换现有 sensor** | (a) 完全替换 `ForwardPinholeCameraPatternCfg` → `NoisyGroupedRayCasterCameraCfg`,**删除** sensors.py;(b) 新增 sensor 并保留旧 sensor 作为 fallback | **(a)** 最小代码 footprint,旧 sensor 在新 pipeline 下没有任何独占价值 | |
| D2 | **是否保留 VisualMimic 风格 random rect mask augment** | (a) 完全切到 main 的 4-step pipeline(distance Gaussian + Normalization + Dropout + CropResize),**删除** `_apply_random_rect_masks`;(b) 在 4-step pipeline 末尾**追加** rect mask 作为第 5 步;(c) 完全切到 main 但用 noise pipeline 自定义新增一个 `RectMaskCfg` | **(b)** main 的 dropout 只覆盖 per-pixel 噪声,大块遮挡(VisualMimic 论文核心)还是要 rect mask 补充。两套互补 | |
| D3 | **`depth_image` obs func 输出形状** | (a) `(B, 1, H, W)`(main 风格,CNN-ready);(b) `(B, H*W)` flat(我们现有的) | **(a)**:跟 main 的 `envs/mdp/observations.py:depth_image` 签名一致,policy CNN 不用 inline reshape | |
| D4 | **是否独立 obs group `"depth"`** | (a) 独立 group,与 1-D policy group 平行;(b) 塞 policy group 末尾(现状) | **(a)**:4-D 张量跟 1-D term concat 在 mjlab obs_manager 里需要 `concatenate_terms=False`,而 policy 内其它项还要 concat,**必须分离**。同时 main 也是这么做 | |
| D5 | **History length** | (a) 1(单帧);(b) 4(main 默认);(c) 关闭 history,通过 obs `history_length` 自动叠加 | **(b)** 4 帧:球以 5-9 m/s 飞,单帧很难判方向;main 用 `AsyncCircularBuffer + update_period` 实现非阻塞 history | |
| D6 | **Update period(相机刷新率)** | (a) 每 sim sub-step 都跑 raycast;(b) `1/10`(10 Hz,模拟真机 RealSense);(c) 跟 `step_dt` 同步(50 Hz) | **(b)** 10 Hz:模拟真机帧率,raycast 跑量 ÷5,且符合真机 sim2real 时序;policy 会看到"帧间停留 5 个 control step" 同一张图 | |
| D7 | **相机 offset(pos + rot)** | (a) 沿用 main 的 `(0.05, 0.01, 0.44)` + 绕 Y 轴 ~45° pitch down(头部下倾,胸前位置);(b) `(0, 0, 0)`(原点,我们现在的位置即 pelvis);(c) 自定义朝前不下倾 | **(a)** 沿用 main:已经在 mjlab-loco 训练过的位置,且 45° 下倾让相机看见地面+前方球 | |
| D8 | **Extrinsic perturbation 默认值** | (a) main 默认(pos ±2cm, roll/yaw ±1°, pitch ±5°);(b) 全 0(禁用);(c) 仅 pitch ±5°(对齐 VisualMimic 单一 yaw 噪声) | **(a)** main 默认值;pitch ±5° 自然就对齐了 VisualMimic | |
| D9 | **Intrinsic perturbation 是否启用** | (a) 启用 main 默认(fov ±5°, cx/cy ±1px);(b) 禁用,只做 extrinsic | **(a)** 启用:真机标定漂移是 sim2real 第二大 gap | |
| D10 | **Critic 是否同时吃 depth** | (a) 否(保留特权 ball state,asymmetric AC);(b) 是(critic 也读 depth + ball state 并列);(c) critic 完全切 depth(对齐 actor) | **(a)** 维持现状:critic 用 oracle ball state,加速 value 收敛;policy 才用 depth | |
| D11 | **Vision policy 重写程度** | (a) 重写 `_VisionActor`:从 flat slice 改成从 obs dict 取 depth group;(b) 在 obs_manager 层把 depth group flatten 拼到 policy group 末尾,policy 不动 | **(a)** 跟 main 对齐;长期看,vision policy 应该接受 dict 输入 | |
| D12 | **是否同步升级 deploy_mujoco 的相机** | (a) 不动 deploy(留下一期);(b) 同步升级 | **(a)** 这次只 sim2sim 训练,deploy 改 sensor 涉及到真机驱动协议对齐,工作量大 | |

---

## TODO List

### Phase 0 — 基础设施移植

> **不能直接 cp -r**:每移植一个文件就跑 import smoke,确保基础设施可用,再进下一个。

- [ ] **0.1** 移植 `mjlab.utils.noise` 的图像噪声管线
  - **源**:`mjlab-prior-main/src/mjlab/utils/noise/noise_cfg.py:155-235` 的 5 个 class:`ImageNoiseCfg / DepthNormalizationCfg / CropAndResizeCfg / DepthDistanceGaussianNoiseCfg / DepthDropoutCfg`
  - **目标**:`mjlab-prior/src/mjlab/utils/noise/noise_cfg.py`(追加到末尾,不动现有 `NoiseCfg / UniformNoiseCfg / GaussianNoiseCfg`)
  - 同步移植 `mjlab-prior-main/src/mjlab/utils/noise/noise_model.py:93-186` 的图像 noise 函数(`ImageNoiseModel / depth_normalization / crop_and_resize / depth_distance_gaussian_noise / depth_dropout`)→ 追加到 `mjlab-prior/src/mjlab/utils/noise/noise_model.py`
  - 更新 `mjlab-prior/src/mjlab/utils/noise/__init__.py` 加 re-export
  - **验证**:
    ```bash
    uv run python -c "from mjlab.utils.noise import (
        ImageNoiseCfg, DepthNormalizationCfg, CropAndResizeCfg,
        DepthDistanceGaussianNoiseCfg, DepthDropoutCfg)"
    ```

- [ ] **0.2** 移植 `AsyncCircularBuffer`(history buffer 依赖)
  - **源**:`mjlab-prior-main/src/mjlab/utils/buffers/async_circular_buffer.py`(单文件)
  - **目标**:`mjlab-prior/src/mjlab/utils/buffers/async_circular_buffer.py`
  - 同步更新 `mjlab-prior/src/mjlab/utils/buffers/__init__.py` 加 re-export
  - 检查依赖:它可能依赖 `CircularBuffer`,确认 mjlab-prior 已有
  - **验证**:
    ```bash
    uv run python -c "from mjlab.utils.buffers.async_circular_buffer import AsyncCircularBuffer; \
        b = AsyncCircularBuffer(history_length=4, batch_size=2, device='cpu'); print(b)"
    ```

- [ ] **0.3** 移植 `mjlab.sensor.grouped_ray_caster` 子包(2 个 py 文件)
  - **源**:`mjlab-prior-main/src/mjlab/sensor/grouped_ray_caster/`
    - `__init__.py`
    - `grouped_ray_caster.py`(继承 `RayCastSensor`)
    - `grouped_ray_caster_cfg.py`(继承 `RayCastSensorCfg`,新增 `min_distance / mesh_prim_paths / aux_mesh_and_link_names / mesh_filter_max_hops / mesh_filter_epsilon`)
    - `grouped_ray_caster_camera.py`(继承 `GroupedRayCaster`,新增 `CameraData / set_intrinsic_matrices`)
    - `grouped_ray_caster_camera_cfg.py`(继承 `GroupedRayCasterCfg`,新增 `OffsetCfg / data_types / update_period / depth_clipping_behavior / focal_length / horizontal_aperture / pattern`)
  - **目标**:`mjlab-prior/src/mjlab/sensor/grouped_ray_caster/`(同名 4 文件 + `__init__.py`)
  - **关键检查项**(可能漂移的 API):
    1. `mjlab-prior` 的 `RayCastSensor.initialize(mj_model, model, data, device)` 签名是否一致 → 检查 `mjlab-prior/src/mjlab/sensor/raycast_sensor.py:480-559`
    2. `RayCastData` 字段 `distances / hit_pos_w / normals_w / pos_w / quat_w / frame_pos_w / frame_quat_w` 是否齐全
    3. `mujoco_warp.rays` API 是否相同(`wp.array` 类型、`wp_cuda_graph` 接口)
    4. `SensorContext` 是否一致
  - 同步更新 `mjlab-prior/src/mjlab/sensor/__init__.py` 加 re-export:`GroupedRayCaster / GroupedRayCasterCfg / GroupedRayCasterCamera / GroupedRayCasterCameraCfg`
  - **验证**:
    ```bash
    uv run python -c "from mjlab.sensor.grouped_ray_caster import (
        GroupedRayCaster, GroupedRayCasterCamera, GroupedRayCasterCameraCfg)"
    ```

- [ ] **0.4** 移植 `mjlab.sensor.noisy_camera` 子包(4 个 py 文件)
  - **源**:`mjlab-prior-main/src/mjlab/sensor/noisy_camera/`
    - `noisy_camera.py`(`NoisyCameraMixin`,提供 noise pipeline + history 的复用代码)
    - `noisy_camera_cfg.py`(`ExtrinsicPerturbationCfg / IntrinsicPerturbationCfg / NoisyCameraCfgMixin`)
    - `noisy_grouped_raycaster_camera.py`(`NoisyGroupedRayCasterCamera = NoisyCameraMixin + GroupedRayCasterCamera`)
    - `noisy_grouped_raycaster_camera_cfg.py`(`NoisyGroupedRayCasterCameraCfg`)
  - **目标**:`mjlab-prior/src/mjlab/sensor/noisy_camera/`
  - 依赖 0.1(ImageNoiseCfg)+ 0.2(AsyncCircularBuffer)+ 0.3(GroupedRayCasterCamera)
  - 注意 `string_to_callable` 来源(`mjlab.utils.lab_api.string`),mjlab-prior 应该已有,但要确认 import path
  - 同步更新 `mjlab-prior/src/mjlab/sensor/__init__.py` 加 re-export:`NoisyGroupedRayCasterCamera / NoisyGroupedRayCasterCameraCfg / ExtrinsicPerturbationCfg / IntrinsicPerturbationCfg`
  - **验证**:
    ```bash
    uv run python -c "from mjlab.sensor import (
        NoisyGroupedRayCasterCamera, NoisyGroupedRayCasterCameraCfg,
        ExtrinsicPerturbationCfg, IntrinsicPerturbationCfg)"
    ```

- [ ] **0.5** 移植 `depth_image` MDP observation function(标准签名)
  - **源**:`mjlab-prior-main/src/mjlab/envs/mdp/observations.py:174-187`
    ```python
    def depth_image(env, sensor_cfg, data_type="distance_to_image_plane_noised"):
      images = sensor.data.output[data_type].clone()   # (N, H, W, C)
      return images.permute(0, 3, 1, 2)                 # (N, C, H, W)
    ```
  - **目标**:`mjlab-prior/src/mjlab/envs/mdp/observations.py`(追加,**不动**当前 football mdp 里的 `depth_image` 直到 D2 决策)
  - 注意:main 的版本签名是 `sensor_cfg: SceneEntityCfg`(用 SceneEntityCfg 包装 sensor 名),不是 raw `sensor_name: str`;迁移时遵照 main 风格
  - 同步更新 `mjlab-prior/src/mjlab/envs/mdp/__init__.py` 加 re-export(如已 wildcard 可跳过)
  - **验证**:
    ```bash
    uv run python -c "from mjlab.envs.mdp import depth_image"
    ```

- [ ] **0.6** 决定 `football.mdp.depth_image` 何去何从(取决于 D2)
  - **D2=(a) 完全切**:**删除** `mjlab-prior/src/mjlab/tasks/football/mdp/observations.py` 中的 `depth_image` 和 `_apply_random_rect_masks`,只保留 ball 状态 obs。delete `football.mdp.depth_image` 的 export
  - **D2=(b) 保留 rect mask**:把 `_apply_random_rect_masks` 改写成 `ImageNoiseCfg` 子类 `RectMaskCfg` + `rect_mask` noise func,放在 0.1 的 noise_model.py 里。删除 `football.mdp.depth_image`
  - **D2=(c) 同上 (b),但放在 football 包内**
  - **验证**:`grep -r "football_mdp.depth_image" mjlab-prior/src/`,确认没有遗留引用

---

### Phase 1 — Perception passing env_cfg 切换到新 sensor

- [ ] **1.1** 新写 `_make_g1_depth_camera_cfg()` 工厂函数
  - **源参考**:`mjlab-prior-main/src/mjlab/tasks/motion_prior/config/g1/env_cfgs.py:76-130`
  - **目标**:`mjlab-prior/src/mjlab/tasks/football/config/g1/env_cfgs.py`(**替换** `_make_g1_forward_lidar_sensor`)
  - 返回 `NoisyGroupedRayCasterCameraCfg`,关键字段(沿用 main):
    ```python
    name="camera"
    frame=ObjRef(type="body", name="pelvis", entity="robot")
    pattern=PinholeCameraPatternCfg(height=64, width=64, fovy=57.9)
    focal_length=1.0
    data_types=["distance_to_image_plane"]
    ray_alignment="base"
    include_geom_groups=(0, 2)  # 地面 + 球
    min_distance=0.05
    depth_clipping_behavior="max"
    update_period=1/10  # 决策 D6
    offset=OffsetCfg(
      pos=(0.0488, 0.01, 0.438),       # 决策 D7
      rot=(0.9135, 0.0044, 0.4067, 0), # 绕 Y 轴 ~45° pitch down
      convention="world",
    )
    noise_pipeline={
      "distance_gaussian": DepthDistanceGaussianNoiseCfg(depth_std=0.005, depth_std_multiplier=0.01),
      "normalize":         DepthNormalizationCfg(depth_range=(0.0, 2.0), normalize=True),
      "dropout":           DepthDropoutCfg(drop_prob=0.01, fill_value=-1.0),
      "crop_resize":       CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(60, 60)),
      # 如果 D2=(b),在这里追加 "rect_mask": RectMaskCfg(...)
    }
    extrinsic_perturbation=ExtrinsicPerturbationCfg(
      pos_range=(0.02, 0.02, 0.02), roll_range=0.01745, pitch_range=0.08727, yaw_range=0.01745,
    )  # 决策 D8
    intrinsic_perturbation=IntrinsicPerturbationCfg(fov_range=5.0, cx_range=1.0, cy_range=1.0)  # 决策 D9
    data_histories={"distance_to_image_plane_noised": 4}  # 决策 D5
    ```
  - **决策 D7 验证**:确认 G1 MJCF 里有 `pelvis` body(我们之前 `_make_g1_forward_lidar_sensor` 用过,应该有)

- [ ] **1.2** 删除 `_PERCEPTION_CAMERA_*` / `_PERCEPTION_NUM_RAYS` / `_PERCEPTION_SENSOR` 等旧常量
  - 改成 `PERCEPTION_IMAGE_HEIGHT = 60` / `WIDTH = 60`(crop+resize 后的目标尺寸)
  - 暴露给 rl_cfg 用

- [ ] **1.3** 重构 `unitree_g1_passing_perception_env_cfg`
  - sensor 注入改成 `cfg.scene.sensors = (..., _make_g1_depth_camera_cfg())`
  - **obs cfg 改成 4 个 group**(决策 D4 = a):
    ```python
    cfg.observations = {
      "motion_prior_obs": _make_motion_prior_obs_group(...),  # 1-D proprio
      "policy": ObservationGroupCfg(terms={
        "passing_source_position": ...,
        **_make_proprio_terms(),
      }, concatenate_terms=True),                              # 1-D, 不含 depth
      "depth": ObservationGroupCfg(terms={
        "depth_image": ObservationTermCfg(
          func=envs_mdp.depth_image,  # 0.5 移植的标准版
          params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "distance_to_image_plane_noised",
          },
        ),
      }, concatenate_terms=True, enable_corruption=False),     # 4-D, 独立 group
      "critic": ObservationGroupCfg(terms={
        # 保持现状:特权 ball state + proprio + base_lin_vel
      }, concatenate_terms=True),
    }
    ```

- [ ] **1.4** 删除 `ForwardPinholeCameraPatternCfg` import 与 `mdp/sensors.py` 文件本身(D1=a)
  - `git rm mjlab-prior/src/mjlab/tasks/football/mdp/sensors.py`
  - 从 `mjlab-prior/src/mjlab/tasks/football/config/g1/env_cfgs.py` 移除 `from mjlab.tasks.football.mdp.sensors import ForwardPinholeCameraPatternCfg`

- [ ] **1.5** 删除 `football.mdp.depth_image` / `_apply_random_rect_masks`(取决于 D2)
  - D2=(a):删干净
  - D2=(b):删 `depth_image`(因 0.5 已有标准版),保留 `_apply_random_rect_masks` 并 wrap 成 `RectMaskCfg + rect_mask` noise func 加入 0.1 的 noise_model.py
  - 同步更新 `mdp/__init__.py` 的 export

---

### Phase 2 — Vision policy 适配 4-D 输入(决策 D11)

- [ ] **2.1** 重写 `DownStreamVQVisionPolicy._VisionActor`
  - **源参考**:无直接 main 实现(main 用 `EncoderMLPModel` 走 rsl_rl,我们走自己写的 actor)
  - **目标**:`mjlab-prior/src/mjlab/tasks/motion_prior/rl/policies/downstream_vq_vision_policy.py`
  - 改 forward 签名:从 `policy_obs: (B, 3975)` 改成接收 **3 个独立 tensor**:
    - `policy_obs: (B, 375)`(passing_src + proprio)
    - `depth: (B, 1, 60, 60)`(独立 group 输出)
    - 可选:`depth_history: (B, 4, 1, 60, 60)` 如果 D5=b
  - actor.forward 改成 `cat([policy_obs, depth_cnn(depth)], -1) → MLP → code_dim`
  - **决策**:depth_history 是否用?(D5=b 时):用 3D-CNN 还是 reshape 成 `(B, 4, 60, 60)` 再过 2D-CNN(把 history 当 channel)?
    - **建议**:2D-CNN + history-as-channel(简单且 main 没用 3D-CNN)

- [ ] **2.2** 改 `policy_inference` 跟 `act`(同样多 3 个参数)

- [ ] **2.3** 改 `DepthCNNEncoder.forward` — 已经接受 `(B, 1, H, W)`,这块**不用动**(我们当前 vision policy 本来就是 reshape 后过 CNN,只是 reshape 在 actor 内部做)

---

### Phase 3 — Runner & cfg 接线

- [ ] **3.1** `DownStreamVQOnPolicyRunner._collect_rollout`(`downstream_runner.py`)
  - 取 obs 时改成 `policy_obs = _t(obs, "policy")` + `depth = _t(obs, "depth")` + `motion_prior_obs = _t(obs, "motion_prior_obs")` + `critic_obs = _t(obs, "critic")`
  - alg.act 签名加 depth 参数

- [ ] **3.2** `DownStreamPPO.act / process_env_step / update`
  - 改 alg storage init,加 `depth_obs_shape` 字段
  - storage 多存一份 depth tensor:rollout `[T, B, 1, H, W]`
  - update 时 batch 切片同步切 depth

- [ ] **3.3** `RslRlDownstreamVQPolicyCfg`(`motion_prior/downstream_rl_cfg.py`)
  - 删 `image_height / image_width`(由 sensor cfg 自己决定)
  - 加 `depth_obs_group_name: str = "depth"` 字段,让 runner 知道从哪个 group 取 depth

- [ ] **3.4** `unitree_g1_passing_perception_vq_runner_cfg`(`football/config/g1/rl_cfg.py`)
  - 不需要从 env_cfgs 读分辨率了,sensor 自己处理 crop+resize

- [ ] **3.5** ONNX export(`downstream_runner.export_policy_to_onnx` → `mjlab.tasks.motion_prior.onnx.export_downstream_to_onnx`)
  - 改成接受多输入:`policy_obs / prop_obs / depth`
  - **可选**:这步暂时跳过,等训完再补

---

### Phase 4 — 测试 & 验证

- [ ] **4.1** Import smoke
  ```bash
  uv run python -c "import mjlab.tasks; from mjlab.tasks.registry import load_env_cfg; \
      cfg = load_env_cfg('Mjlab-Football-Passing-Perception-Unitree-G1'); \
      print('sensors:', [s.name for s in cfg.scene.sensors])"
  ```
  期望:`['..., camera']`

- [ ] **4.2** Env reset + step smoke
  ```bash
  uv run python -c "
  import torch, mjlab.tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  cfg = load_env_cfg('Mjlab-Football-Passing-Perception-Unitree-G1')
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
  obs = env.get_observations()
  for k, v in obs.items(): print(f'{k}: {tuple(v.shape)}')
  act = torch.zeros(4, env.action_manager.total_action_dim, device='cuda:0')
  for _ in range(20): env.step(act)
  cam = env.scene.sensors['camera']
  print('depth output shape:', cam.data.output['distance_to_image_plane_noised'].shape)
  "
  ```
  期望:
  - `depth: (4, 1, 60, 60)`(crop+resize 后)
  - depth 值在 [0, 1] 范围内
  - 1% 像素 = -1(dropout fill)

- [ ] **4.3** Policy forward smoke
  ```bash
  uv run python -c "
  from mjlab.tasks.motion_prior.rl.policies.downstream_vq_vision_policy import DownStreamVQVisionPolicy
  # 构造 dummy 多输入 forward
  ..."
  ```

- [ ] **4.4** 100 iter 训练 smoke
  ```bash
  uv run python -m mjlab.scripts.train \
    Mjlab-Football-Passing-Perception-Unitree-G1 \
    --agent.motion-prior-ckpt-path /path/to/mp.pt \
    --agent.max-iterations 100 --num-envs 1024
  ```
  期望:loss 不 NaN,SPS > 5000(降 sensor refresh 到 10 Hz 后 throughput 应该上去)

- [ ] **4.5** Depth viz 脚本兼容
  - 改 `scripts/play_perception_depth_viz.py` 默认 sensor 名为 `"camera"`,默认从 `sensor.data.output["distance_to_image_plane_noised"]` 取数据(而不是 raycast 距离)
  - 验证 play 时窗口标题 `shape 60×60`,turbo cmap 下能看到完整透视深度图

---

### Phase 5 — Deploy 升级(暂缓,留下一期)

- [ ] **5.1** `deploy/deploy_mujoco/` 加 `mujoco_camera_helper.py` 真渲染深度
  - mjlab-prior-main 也没 deploy 这条路径,要重新写
  - 真机:从 RealSense 驱动读 depth → spatial/temporal filter → resize → 喂 policy

> Phase 5 不在本次范围,**等真机部署阶段再启动**。

---

## 需要替换的具体位置一览表

| # | 文件 | 当前内容 | 替换为 |
|---|---|---|---|
| S1 | `football/mdp/sensors.py` | `ForwardPinholeCameraPatternCfg` | **删除整个文件**(D1=a) |
| S2 | `football/mdp/observations.py:depth_image` | flat 输出 + obs-side `_apply_random_rect_masks` | **删除**,改用 `envs/mdp/observations.py:depth_image`(0.5)(D2 决定 rect mask 是否保留为 noise pipeline 一步) |
| S3 | `football/mdp/__init__.py` | export `depth_image` | 移除 |
| S4 | `football/config/g1/env_cfgs.py:_make_g1_forward_lidar_sensor` | `RayCastSensorCfg + ForwardPinholeCameraPatternCfg` | **替换**为 `_make_g1_depth_camera_cfg() → NoisyGroupedRayCasterCameraCfg`(1.1) |
| S5 | `football/config/g1/env_cfgs.py:unitree_g1_passing_perception_env_cfg` 的 obs cfg | `policy` group 末尾塞 `ball_depth_image` | **拆**:独立 `"depth"` group(1.3) |
| S6 | `motion_prior/rl/policies/downstream_vq_vision_policy.py:_VisionActor` | 从 flat slice 切 depth | **改**:接受独立的 `depth: (B, 1, H, W)` 参数(2.1) |
| S7 | `motion_prior/rl/downstream_runner.py:_collect_rollout` | 单 `policy_obs` | **改**:取 `depth` group(3.1) |
| S8 | `motion_prior/downstream_rl_cfg.py:RslRlDownstreamVQPolicyCfg` | `image_height / image_width / depth_channels` | **改**:删 image_*, 加 `depth_obs_group_name`(3.3) |
| S9 | `scripts/play_perception_depth_viz.py` | sensor 名 `pelvis_forward_camera`,从 `data.distances` 读 | **改**:sensor 名 `"camera"`,从 `data.output["distance_to_image_plane_noised"]` 读(4.5) |

---

## 风险 & 注意事项

1. **`mujoco_warp.rays` API 漂移**:`mjlab-prior` 和 `mjlab-prior-main` 用的 `mujoco_warp` 版本可能不同。`GroupedRayCaster` 里调用 `mujoco_warp.rays.*` 的接口要逐一对照检查。
   - 开工前 check:
     ```bash
     uv run python -c "import mujoco_warp; print(mujoco_warp.__version__)"
     ```
   - 对比 `mjlab-prior-main/.venv/lib/python3.X/site-packages/mujoco_warp` 的版本

2. **`SensorContext` 接口一致性**:`GroupedRayCaster.initialize` 需要拿 `SensorContext`(BVH 加速)。mjlab-prior 的 `SensorContext` 接口若漂移,raycast 路径会 silently failed(返回全 miss)。

3. **`RayCastSensorCfg` 字段差异**:`GroupedRayCasterCfg` 继承 `RayCastSensorCfg`,新增 `min_distance / mesh_filter_*`。要确认 mjlab-prior 的 `RayCastSensorCfg` 没有冲突字段。

4. **ckpt 不兼容**:这是**破坏性改动**,改完后老的 perception passing ckpt 全部不能用,**train 要从头跑**。文档要写清楚。

5. **obs shape 跨多个 group**:rsl_rl PPO 默认期望 `obs` 是单 tensor,我们改成 dict 后,storage / batching / clipping 都要相应改。Phase 3.2 是最容易出 bug 的环节。

6. **`AsyncCircularBuffer` 异步性**:`update_period=1/10` 配合 history=4 意味着 history 跨越 ~0.4s,policy 看到的相邻"帧"是 5 个 control step 之前的。如果 reset 时 history 没清 0,会出现"上一 episode 末帧 + 当前 episode 头帧"混在一起。**reset hook 必须正确 clear history buffer**。

7. **Performance**:相比当前 80×45 raycast(每 sim step 跑),新方案是 64×64 raycast + 10 Hz 节流。raycast 总量 ≈ 4096 cam_rays / 10 ≈ 同等 GPU 压力,但加上 noise pipeline 4 步 + history buffer,**单次 sense 时间会增加**。整体 throughput 应该和现状持平或略好。

8. **`OffsetCfg.convention`**:`"world"` convention 意味着 forward=+X / up=+Z(机器人坐标系约定)。这跟 `"opengl"` / `"ros"` 都不同,**务必传 `convention="world"` 否则相机朝向会差 90°**。

9. **`pelvis` body 与 `OffsetCfg.pos` 的复合**:有了 `OffsetCfg(pos=(0.05, 0.01, 0.44))`,相机起点是 `world(pelvis.xpos + R_pelvis @ offset)`。世界 z ≈ 0.78 + 0.44 = 1.22 m,接近真机 RealSense 安装高度。**之前我们撤回过 `origin_offset` 字段,这次通过标准 `OffsetCfg` 实现相同效果且更通用**。

10. **MJCF asset group filter**:`include_geom_groups=(0, 2)`(地面 group 0 + 球 group 2),要确认 mjlab-prior 的足球 spec 把球 geom 放在 group 2;否则改成 `None`(全 group)。

---

## 开工顺序建议

1. **Phase 0 全部完成**(0.1 → 0.2 → 0.3 → 0.4 → 0.5 → 0.6),**每移植一个文件就跑 import smoke**,确保基础设施可用
2. **Phase 1**(env 注册),跑 4.1 + 4.2 smoke,确认 sensor 在场景里能产生合理深度图
3. **Phase 2 + 3**(policy + runner),跑 4.3 + 4.4,确认 forward / backward 不崩
4. **Phase 4** 全面回归
5. Phase 5 留到后续

---

## 决策回执(请填写后告诉我开工)

| # | 决策 | 我的建议 | **你的选择** |
|---|---|---|---|
| D1 | 是否完全替换现有 sensor | a(完全替换) | |
| D2 | 是否保留 rect mask augment | b(noise pipeline 末尾追加 rect mask) | |
| D3 | depth_image 输出形状 | a((B, 1, H, W)) | |
| D4 | 是否独立 obs group | a(独立 `"depth"`) | |
| D5 | history length | b(4 帧) | |
| D6 | update period | b(1/10s = 10 Hz) | |
| D7 | 相机 offset | a(沿用 main,胸前 45° 下倾) | |
| D8 | extrinsic perturbation | a(main 默认值) | |
| D9 | intrinsic perturbation | a(启用 main 默认) | |
| D10 | critic 是否吃 depth | a(否,保留特权 ball state) | |
| D11 | vision policy 重写程度 | a(改 actor 接受多 tensor 输入) | |
| D12 | deploy 是否同步升级 | a(留下一期) | |

---

## 预计工作量(给定全部按建议选)

| Phase | 子任务数 | 预计代码行数(净增) | 预计实施时间 |
|---|---|---|---|
| 0 | 6 项 | +1500(基础设施 port) | 半天 |
| 1 | 5 项 | +50,-200(净减,因为删了 sensors.py + obs func) | 30 分钟 |
| 2 | 3 项 | +80(actor 多输入) | 1 小时 |
| 3 | 5 项 | +50(rollout / cfg) | 1-2 小时 |
| 4 | 5 项 | smoke test 不写代码 | 30 分钟跑通 |
| **合计** | **24 项** | **+1480 行** | **~1 天** |

不写代码,跑通也得反复 import 调依赖问题 ≈ 半天。**估算总工时:1-1.5 天纯实施**。
