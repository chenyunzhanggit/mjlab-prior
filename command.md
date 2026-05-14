

```bash
# motion tracking
# train
uv run python -m mjlab.scripts.train Mjlab-Trackingbfm-Flat-Unitree-G1 --motion-path ./motions/football-qingtong-trimmed_mujoco --num-envs 4096 --motion-type mujoco

# play
uv run python -m mjlab.scripts.play Mjlab-Trackingbfm-Flat-Unitree-G1 --motion-path ./motions/football-qingtong-trimmed_mujoco --num-envs 1 --motion-type mujoco --checkpoint-file /home/dx/mjlab-prior/logs/rsl_rl/g1_tracking/2026-05-08_20-18-02/model_33000.pt


# motion prior
# train
uv run python -m mjlab.scripts.train Mjlab-MotionPrior-Single-VQ-Trackingbfm-Unitree-G1 --motion-path ./motions/football-qingtong-trimmed_mujoco --num-envs 4096 --teacher-policy-path /home/dx/mjlab-prior/logs/rsl_rl/g1_tracking/2026-05-08_20-18-02/model_33000.pt --motion-type mujoco

# play
uv run python -m mjlab.scripts.play Mjlab-MotionPrior-Single-VQ-Trackingbfm-Unitree-G1 --motion-path ./motions/football-qingtong-trimmed_mujoco --num-envs 1 --teacher-policy-path /home/dx/mjlab-prior/logs/rsl_rl/g1_tracking/2026-05-08_20-18-02/model_33000.pt --motion-type mujoco --checkpoint-file /home/dx/mjlab-prior/logs/rsl_rl/g1_motion_prior_single_vq/2026-05-09_14-55-43/model_3000.pt --inference-path encoder


# downstream
# train
# vel
uv run python -m mjlab.scripts.train Mjlab-Downstream-VQ-Velocity-Unitree-G1 --num-envs 4096 --motion-prior-ckpt-path /home/dx/mjlab-prior/logs/rsl_rl/g1_motion_prior_single_vq/2026-05-09_17-36-38/model_31500.pt --max-iterations 100000
# football
uv run python -m mjlab.scripts.train Mjlab-Football-Passing-VQ-Unitree-G1 --motion-prior-ckpt-path "/home/dx/mjlab-prior/logs/rsl_rl/g1_motion_prior_single_vq/2026-05-09_17-36-38/model_31500.pt" --num-envs 4096

kicking dribbling


# play

uv run python -m mjlab.scripts.play Mjlab-Downstream-VQ-Velocity-Unitree-G1 --checkpoint-file /home/dx/mjlab-prior/logs/rsl_rl/g1_downstream_vq_velocity/2026-05-13_18-43-17/model_75500.pt --num-envs 1 --motion-prior-ckpt-path "/home/dx/mjlab-prior/logs/rsl_rl/g1_motion_prior_single_vq/2026-05-13_18-18-49/model_1500.pt" --viewer viser 



```

