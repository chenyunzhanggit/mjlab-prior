# Downstream Task Policy Migration: ckpt shape audit

不 commit。给迁移实现 (#24-#27) 用做参照。

## 目标

把 ~/zcy/motionprior 的 `DownStreamPolicy` 移植到 mjlab_prior，**严格用我们自己训好的 motion_prior ckpt 做 frozen 底座**。重点是 frozen 三件套（`decoder` / `motion_prior` MLP / `mp_mu` Linear）的 shape 必须对得上。

## 1. 我们 ckpt 的实际结构

来源：`MotionPriorOnPolicyRunner.save()` 写入文件，顶层是一个 dict。

```
ckpt = {
    "encoder_a": <state_dict>,
    "encoder_b": <state_dict>,
    "es_a_mu":   <state_dict>,
    "es_a_var":  <state_dict>,
    "es_b_mu":   <state_dict>,
    "es_b_var":  <state_dict>,
    "decoder":      <state_dict>,    # ← downstream 需要
    "motion_prior": <state_dict>,    # ← downstream 需要
    "mp_mu":        <state_dict>,    # ← downstream 需要 (VAE)
    "mp_var":       <state_dict>,    # downstream 不需要
    "optimizer":    <state_dict>,
    "iter":         int,
    "infos":        {...},
}
```

下游训练只需要切出 `decoder` / `motion_prior` / `mp_mu` 三份。**无需 prefix remap**，每份本身就是一个干净的 nested state_dict。

## 2. 各 frozen 部件实际形状（实测自训出来的真 ckpt）

> ⚠️ **更新**：实测一份真 ckpt
> (`logs/rsl_rl/g1_motion_prior/2026-04-29_18-28-32/model_5.pt`) 的形状显示：
>
> ```
> motion_prior.0.weight: (512, 559)   ← prop_obs_dim = 559（不是 372!)
> decoder.0.weight: (512, 591)        ← input = 559 + 32 = 591
> mp_mu.weight: (32, 32)
> ```
>
> 根本原因：当前 `unitree_g1_flat_motion_prior_env_cfg` 在 student obs 里
> 注入了 `height_scan` 一项（187 维），所以 student = 5 proprio×4 + 187 = **559**。
>
> 含义：DownStreamPolicy 必须**从 env 推断 prop_obs_dim**（不能 hardcode 372），
> downstream 的 `motion_prior_obs` 必须输出同样 559 维（schema 与 motion_prior
> 训练时 student 完全一致，包括 height_scan）。

`latent_z_dims = 32`，`prop_obs_dim` 由 motion_prior 训练时的 student obs 决定（当前为 559，未来如果 student schema 改了对应改）。

### `motion_prior` (`rsl_rl.modules.MLP`)

```
Linear(372, 512) → ELU
Linear(512, 256) → ELU
Linear(256, 128) → ELU
Linear(128, 32)              ← 输出 latent_z_dims=32
```

state_dict keys: `0.weight (512,372)`, `0.bias (512)`, `2.weight (256,512)`, `2.bias (256)`, `4.weight (128,256)`, `4.bias (128)`, `6.weight (32,128)`, `6.bias (32)`.

### `mp_mu` (`nn.Linear`)

```
Linear(32, 32)              ← 输入 latent_z_dims=32
```

state_dict keys: `weight (32,32)`, `bias (32)`.

### `decoder` (`rsl_rl.modules.MLP`)

```
Linear(372+32=404, 512) → ELU
Linear(512, 256) → ELU
Linear(256, 128) → ELU
Linear(128, 29)              ← 输出 num_actions
```

state_dict keys: `0.weight (512,404)`, `0.bias (512)`, ..., `6.weight (29,128)`, `6.bias (29)`.

## 3. Reference DownStreamPolicy 期望（≠ 我们）

reference (`~/zcy/motionprior/.../downstream_task_policy.py`) **以 `latent_z_dims=32` 为最终 latent**，但中间过了一层 64：

| Module | reference 期望 | 我们 ckpt 实际 | 一致 |
|---|---|---|---|
| `motion_prior` 输出 dim | **64** (硬编码 `nn.Linear(motion_prior_hidden_dims[-1], 64)`) | **32** (=latent_z_dims) | ❌ |
| `mp_mu` 输入 dim | **64** (硬编码 `nn.Linear(64, latent_z_dims)`) | **32** (=latent_z_dims) | ❌ |
| `decoder` 输入 dim | latent_z_dims+372 = **404** | 372+32 = **404** | ✅ |
| `decoder` 输出 dim | num_actions = **29** | **29** | ✅ |
| hidden_dims (motion_prior / mp_mu / decoder) | `[512, 256, 128]` | `[512, 256, 128]` | ✅ |

**只有 motion_prior 输出 / mp_mu 输入这个"中间 dim"不一致**：reference 用 64 当中转，我们直接用 32。

### 含义

如果 verbatim 复制 reference 代码 + `strict=True` load，会在以下两个 key 上 shape mismatch：
- `motion_prior.6.weight`: 期望 `(64, 128)`，实际 `(32, 128)`
- `mp_mu.weight`: 期望 `(32, 64)`，实际 `(32, 32)`

### 解决

我们的 `DownStreamPolicy` **不复制 reference 的硬编码 64**，改成：

```python
# motion_prior MLP
self.motion_prior = MLP(prop_obs_dim, latent_z_dims, hidden_dims=motion_prior_hidden_dims, ...)
# mp_mu Linear
self.mp_mu = nn.Linear(latent_z_dims, latent_z_dims)
```

跟我们 `MotionPriorPolicy` 写法 1:1，shape 自动匹配。

**功能上等价**：reference 多走一层 64-d 映射，我们少走一层。actor 学的 latent residual 加到 `mp_mu(motion_prior(prop))` 的 32 维输出上，下游 PPO 训练逻辑完全相同。

## 4. trainable 部件期望（不依赖 ckpt）

reference 默认值：
- `actor`: MLP(num_obs → 512 → 256 → 128 → latent_z_dims=32)
- `critic`: MLP(num_privileged_obs → 512 → 256 → 128 → 1)
- `std`: `nn.Parameter(init_noise_std * torch.ones(latent_z_dims=32))`

直接复用，不需要修改。

## 5. obs schema 对照

reference (`g1_downstream_vq_cfg.py`) 三组 obs：

| 组 | 内容 | 维度 | 用途 |
|---|---|---|---|
| `policy` | velocity_commands(3) + 5 项 proprio × hist=4 | 3 + 372 = **375** | actor 输入 |
| `motion_prior_obs` | 5 项 proprio × hist=4（无 vel cmd） | **372** | frozen `motion_prior` 头输入 |
| `critic` | policy + base_lin_vel(3) | 375 + 3 = **378** | critic 输入（含特权信息）|

我们要构造的 mjlab env_cfg 严格按这个 schema 复刻，**`motion_prior_obs` 必须等于 motion_prior 训练时的 `student` 372 维**（noise / order / history_length 全部一致），否则 frozen `motion_prior(prop_obs)` 在分布外。

## 6. 风险清单（给后续 task 留意）

1. **新版 motion_prior ckpt 训练时如果 hidden_dims 不是 (512,256,128)**，downstream 需要从 cfg 显式接收 hidden_dims，不能硬编码。**对策**：DownStreamPolicy cfg 暴露 `frozen_hidden_dims` 字段，默认 (512,256,128)，调 build 时跟 ckpt 对齐。
2. **prop_obs_dim 372 是写死的**（5 项 proprio × hist=4 = 372 for G1）。换 robot 必须重新算。**对策**：DownStreamPolicy 接收 `prop_obs_dim` 参数，从 env obs 自动推断而非硬编码。
3. **`MotionPriorPolicy.motion_prior` 是单 MLP 直接输出 32**，跟 reference 的"MLP→64→Linear→32"不同。这意味着 reference 那个 64-d 中转层在我们这是不存在的，**`mp_mu` 在我们这其实只是个 32×32 的恒等近似**（reparameterize 头）。功能上没问题，但要知道这点。
4. **`mp_var` 在 downstream 不被消费**。ckpt 里有这个 key 但 DownStreamPolicy 不 load 它（reference 也不 load），不影响。
5. **decoder 在 motion_prior 训练时被 KL loss 学过**：见过 `[prop_obs, mp_mu_output]` 这个 latent 分布。downstream actor 输出 `latent = mp_mu_output + raw_action`，这超出了 decoder 训练时见过的分布。**这是 reference 设计本身的 distribution shift 风险，不是迁移的额外引入**。

## 7. audit 结论

可以推进 #24（ckpt loader）和 #25（DownStreamPolicy）。**唯一与 reference 的实现差异是中间 dim 64 → 32**，文档记录在此，避免后续 review 时被以为是 bug。
