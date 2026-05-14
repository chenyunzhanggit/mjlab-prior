import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import onnxruntime as ort
def load_onnx_policy(onnx_path, use_gpu=False):
    providers = ["CPUExecutionProvider"]
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"Loaded ONNX policy: {onnx_path}")
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    return session, input_names, output_names


def onnx_inference(session, input_names, observations, history):
    """
    observations: np.ndarray shape [B, obs_dim]
    history: np.ndarray shape [B, history_dim]
    """
    ort_inputs = {
        input_names[0]: observations.astype(np.float32),
        input_names[1]: history.astype(np.float32)
    }
    ort_outs = session.run(None, ort_inputs)
    actions = ort_outs[0]  # shape [B, action_dim]
    return actions

def key_call_back( keycode):
    global paused
    global pref_weight
    delta = 0.2
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "1": 
        pref_weight[0] = max(0.0, pref_weight[0] - delta)
        pref_weight[1] = min(1.0, pref_weight[1] + delta)
        print("Updated pref_weight:", pref_weight)

    elif chr(keycode) == "2": 
        pref_weight[0] = min(1.0, pref_weight[0] + delta)
        pref_weight[1] = max(0.0, pref_weight[1] - delta)
        print("Updated pref_weight:", pref_weight)
    else:
        print("not mapped", chr(keycode))

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse


    with open(f"/home/teleai/lhy/Humanoid-AMP-ahc/deploy_pref/deploy_mujoco/configs/g1.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history_len = config["obs_history_len"]
    trajectory_history = torch.zeros(size=(1, obs_history_len, num_obs))

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    d.qpos[0:3] = config["init_pos"]  # x, y, z
    d.qpos[3:7] = config["init_rot"]
    d.qpos[7:] = default_angles.copy()

    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # load policy
    # policy = torch.jit.load(policy_path)

    session, input_names, output_names = load_onnx_policy(policy_path, use_gpu=False)

    # import ipdb;ipdb.set_trace()
    global paused
    paused = True
    global pref_weight
    pref_weight = np.array(config["pref_weight"], dtype=np.float32)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_call_back) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            if not paused:
                tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                current_qj = qj.copy()

                # qj = (qj - default_angles) * dof_pos_scale
                qj = qj * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = cmd * cmd_scale
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 1] = action_scale

                # pref
                obs[-2:] = pref_weight
                
                # # onnx
                action = onnx_inference(session, input_names, obs.reshape(1, -1), trajectory_history.numpy().astype(np.float32)).squeeze(0)

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                 # policy inference
                # action = policy(obs_tensor, trajectory_history).detach().numpy().squeeze()

                trajectory_history = torch.cat([trajectory_history[:, 1:], obs_tensor.unsqueeze(1)], dim=1)

                # transform action to target_dof_pos
                # target_dof_pos = action * action_scale + default_angles
                target_dof_pos = action * action_scale + current_qj

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
