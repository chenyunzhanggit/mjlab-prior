Teacher Policy:
Nothing need to mention.

Motion Prior:
VAE:
Train CMD：
uv run python -m mjlab.scripts.train  Mjlab-MotionPrior-Flat-Unitree-G1 --env.commands.motion.motion-path /home/lenovo/g1_retargeted_data/npz/Data10k/  --env.scene.num-envs 1024 --agent.secondary-num-envs 1024 --agent.max-iterations 20000 --agent.teacher-a-policy-path /path/to/teacher_tracking_ckp.pt --agent.teacher-b-policy-path /path/to/teacher_locomotion_ckp.pt

flat tracking latent + mp + prop_obs: play_motion_prior_flat.sh
rough locomotion latent + mp + prop_obs: play_motion_prior_rough.sh
only mp + prop_obs: play_motion_prior_deploy.sh (change the task in this bash to change the FLAT or ROUGH)

VQ-VAE
uv run python -m mjlab.scripts.train  Mjlab-MotionPrior-VQ-Flat-Unitree-G1 --env.commands.motion.motion-path /home/lenovo/g1_retargeted_data/npz/Data10k --env.scene.num-envs 2048 --agent.secondary-num-envs 2048 --agent.max-iterations 100000 --agent.teacher-a-policy-path /home/lenovo/project/Teleopit/track.pt --agent.teacher-b-policy-path /home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt

flat tracking latent + mp + prop_obs: play_motion_prior_vq_flat.sh
rough locomotion latent + mp + prop_obs: play_motion_prior_vq_rough.sh
only mp + prop_obs: play_motion_prior_vq_deploy.sh (change the task in this bash to change the FLAT or ROUGH)

DownStream Task:
VAE: run_downstream_velocity.sh
VQ-VAE: run_downstream_velocity_vq.sh