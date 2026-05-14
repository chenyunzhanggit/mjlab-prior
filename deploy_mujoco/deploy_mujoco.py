import re
import time
import mujoco, mujoco_viewer, mujoco.viewer
import numpy as np
import torch
import yaml
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
import onnxruntime as ort
import argparse
    

isaaclab_joint_names = [
    'left_hip_pitch_joint', 
    'right_hip_pitch_joint',
    'waist_yaw_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'waist_roll_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'waist_pitch_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint'
]

mujoco_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

stiffness_dict = {
    '.*_hip_pitch_joint': 40.17923847137318,
    '.*_hip_roll_joint': 99.09842777666113,
    '.*_hip_yaw_joint': 40.17923847137318,
    '.*_knee_joint': 99.09842777666113,
    '.*_ankle_pitch_joint': 28.50124619574858,
    '.*_ankle_roll_joint': 28.50124619574858,
    'waist_roll_joint': 28.50124619574858,
    'waist_pitch_joint': 28.50124619574858,
    'waist_yaw_joint': 40.17923847137318,
    '.*_shoulder_pitch_joint': 14.25062309787429,
    '.*_shoulder_roll_joint': 14.25062309787429,
    '.*_shoulder_yaw_joint': 14.25062309787429,
    '.*_elbow_joint': 14.25062309787429,
    '.*_wrist_roll_joint': 14.25062309787429,
    '.*_wrist_pitch_joint': 16.77832748089279,
    '.*_wrist_yaw_joint': 16.77832748089279,
}

damping_dict = {
    '.*_hip_pitch_joint': 2.5578897650279457,
    '.*_hip_roll_joint': 6.3088018534966395,
    '.*_hip_yaw_joint': 2.5578897650279457,
    '.*_knee_joint': 6.3088018534966395,
    '.*_ankle_pitch_joint': 1.814445686584846,
    '.*_ankle_roll_joint': 1.814445686584846,
    'waist_roll_joint': 1.814445686584846,
    'waist_pitch_joint': 1.814445686584846,
    'waist_yaw_joint': 2.5578897650279457,
    '.*_shoulder_pitch_joint': 0.907222843292423,
    '.*_shoulder_roll_joint': 0.907222843292423,
    '.*_shoulder_yaw_joint': 0.907222843292423,
    '.*_elbow_joint': 0.907222843292423,
    '.*_wrist_roll_joint': 0.907222843292423,
    '.*_wrist_pitch_joint': 1.06814150219,
    '.*_wrist_yaw_joint': 1.06814150219,
}
scale_dict = {
    '.*_hip_yaw_joint': 0.5475464652142303,
    '.*_hip_roll_joint': 0.3506614663788243,
    '.*_hip_pitch_joint': 0.5475464652142303,
    '.*_knee_joint': 0.3506614663788243,
    '.*_ankle_pitch_joint': 0.43857731392336724,
    '.*_ankle_roll_joint': 0.43857731392336724,
    'waist_roll_joint': 0.43857731392336724,
    'waist_pitch_joint': 0.43857731392336724,
    'waist_yaw_joint': 0.5475464652142303,
    '.*_shoulder_pitch_joint': 0.43857731392336724,
    '.*_shoulder_roll_joint': 0.43857731392336724,
    '.*_shoulder_yaw_joint': 0.43857731392336724,
    '.*_elbow_joint': 0.43857731392336724,
    '.*_wrist_roll_joint': 0.43857731392336724,
    '.*_wrist_pitch_joint': 0.07450087032950714,
    '.*_wrist_yaw_joint': 0.07450087032950714,
}

def get_param(joint_name, param_dict):
    for pattern, value in param_dict.items():
        if pattern.startswith('.*'):
            if joint_name.startswith('left') or joint_name.startswith('right'):
                if joint_name.endswith(pattern[3:]):
                    return value
        else:
            if joint_name == pattern:
                return value
    raise ValueError(f"No value found for joint: {joint_name}")

def get_action_scale(joint_name):
    for pattern, scale in scale_dict.items():
        # 用 left/right 替换 .*
        if pattern.startswith('.*'):
            if joint_name.startswith('left') or joint_name.startswith('right'):
                if joint_name.endswith(pattern[3:]):
                    return scale
        else:
            if joint_name == pattern:
                return scale
    raise ValueError(f"No scale found for joint: {joint_name}")

isaaclab_to_mujoco_reindex = [isaaclab_joint_names.index(name) for name in mujoco_joint_names]
mujoco_to_isaaclab_reindex = [mujoco_joint_names.index(name) for name in isaaclab_joint_names]
joint_names = mujoco_joint_names
kps = [get_param(name, stiffness_dict) for name in joint_names]
kds = [get_param(name, damping_dict) for name in joint_names]
action_scale = np.array([get_action_scale(name) for name in joint_names], dtype=np.float32)


print('isaaclab_to_mujoco_reindex')
print(isaaclab_to_mujoco_reindex)
print('mujoco_to_isaaclab_reindex')
print(mujoco_to_isaaclab_reindex)

print('kp:')
print(kps)
print('kd:')
print(kds)

print('action_scale')
print(action_scale)

# ====== MotionLoader 类（参考 isaaclab） ======
class MotionLoader:
    def __init__(self, motion_file):
        data = np.load(motion_file)
        self.joint_pos = data['joint_pos']  # [T, 29]
        self.joint_vel = data['joint_vel']  # [T, 29]
        self.body_pos = data['body_pos_w']  # [T, N, 3]
        self.body_ori = data['body_quat_w'] # [T, N, 4]
        self.body_ang_vel_w = data['body_ang_vel_w'] 
        self.fps = data['fps']
        self.T = self.joint_pos.shape[0]

# ====== obs getter 函数（可根据需要扩展） ======
def get_command(motion_loader, t):
    return np.concatenate([motion_loader.joint_pos[t], motion_loader.joint_vel[t]], axis=0)

def motion_ref_pos_b(sim_data, motion_loader, t):
    # DEBUG
    robot_pos = sim_data.body('torso_link').xpos.copy().reshape(1,3)
    robot_quat = sim_data.body('torso_link').xquat.copy().reshape(1,4)
    ref_pos = motion_loader.body_pos[t][9].reshape(1,3)
    ref_quat = motion_loader.body_ori[t][9].reshape(1,4)
    print('robot pos:', robot_pos)
    print('ref pos:', ref_pos)

    t12, _ = subtract_frame_transforms(
        torch.from_numpy(robot_pos).float(),
        torch.from_numpy(robot_quat).float(),
        torch.from_numpy(ref_pos).float(),
        torch.from_numpy(ref_quat).float(),
    )
    return np.array(t12, dtype=np.float32) 

def motion_ref_ori_b(sim_data, motion_loader, t):
    # DEBUG
    robot_pos = sim_data.body('torso_link').xpos.copy().reshape(1,3)
    robot_quat = sim_data.body('torso_link').xquat.copy().reshape(1,4)

    ref_pos = motion_loader.body_pos[t][9].reshape(1,3)
    ref_quat = motion_loader.body_ori[t][9].reshape(1,4)
    _, q12 = subtract_frame_transforms(
        torch.from_numpy(robot_pos).float(),
        torch.from_numpy(robot_quat).float(),
        torch.from_numpy(ref_pos).float(),
        torch.from_numpy(ref_quat).float(),
    )
    mat = matrix_from_quat(q12)
    mat = mat[..., :2].reshape(mat.shape[0], -1)
    return np.array(mat, dtype=np.float32)

def get_base_lin_vel(sim_data):
    return sim_data.qvel[0:3]  # shape [3]

def get_base_ang_vel(sim_data):
    return sim_data.qvel[3:6]  # shape [3]

def get_joint_pos(sim_data):
    return sim_data.qpos[7:36] - default_angles  # shape [29]

def get_joint_vel(sim_data):
    return sim_data.qvel[6:35]  # shape [29]

def get_ref_ang_vel(motion_loader, t):
    return motion_loader.body_ang_vel_w[t][9].reshape(1, 3)

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

# ====== observation 计算函数 ======
def compute_observation_0(sim_data, motion_loader, t, last_actions):
    # breakpoint()
    t = t-1
    if t < 0:
        return np.zeros(160, dtype=np.float32)
    
    obs = np.zeros(160, dtype=np.float32)
    obs[:58] = get_command(motion_loader, t)
    obs[58:61] = motion_ref_pos_b(sim_data, motion_loader, t)
    obs[61:67] = motion_ref_ori_b(sim_data, motion_loader, t)
    obs[67:70] = get_base_lin_vel(sim_data)
    obs[70:73] = get_base_ang_vel(sim_data)
    obs[73:102] = get_joint_pos(sim_data)[mujoco_to_isaaclab_reindex]
    obs[102:131] = get_joint_vel(sim_data)[mujoco_to_isaaclab_reindex]
    obs[131:160] = last_actions
    return obs

# new_obs_add_prog
def compute_observation_1(sim_data, motion_loader, t, last_actions):
    # breakpoint()
    # t = t-1
    # if t < 0:
    #     return np.zeros(154, dtype=np.float32)
    obs = np.zeros(154, dtype=np.float32)
    obs[:58] = np.zeros(58, dtype=np.float32)
    obs[58:61] = get_gravity_orientation(sim_data.qpos[3:7])
    obs[61:64] = np.zeros(3, dtype=np.float32)
    obs[64:67] = get_base_ang_vel(sim_data)
    obs[67:96] = get_joint_pos(sim_data)[mujoco_to_isaaclab_reindex]
    obs[96:125] = get_joint_vel(sim_data)[mujoco_to_isaaclab_reindex]
    obs[125:154] = last_actions
    return obs

def compute_observation_2(sim_data, motion_loader, t, last_actions):
    t = t-1
    if t < 0:
        return np.zeros(158, dtype=np.float32)
    obs = np.zeros(158, dtype=np.float32)
    obs[:58] = get_command(motion_loader, t)
    obs[58:61] = get_gravity_orientation(sim_data.qpos[3:7])
    obs[61:64] = get_ref_ang_vel(motion_loader, t)
    obs[64:67] = get_base_lin_vel(sim_data)
    obs[67:70] = get_base_ang_vel(sim_data)
    obs[70:71] = sim_data.qpos[2]  # base pos z
    obs[71:100] = get_joint_pos(sim_data)[mujoco_to_isaaclab_reindex]
    obs[100:129] = get_joint_vel(sim_data)[mujoco_to_isaaclab_reindex]
    obs[129:158] = last_actions
    return obs


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    
    VIEW_MOTION = False

    # policy_path = '/home/bcj/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-14_21-31-24_dance1_subject2/exported/policy.onnx'
    # policy_path = '/home/bcj/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-18_17-40-34_dance1_subject2-new_obs/exported/policy.onnx'
        
    # motion global root pos 2 - second train
    # policy_path = '/home/bcj/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-20_10-17-53_dance1_subject2-new_obs_add_prog-motion_global_root_pos_2/exported/policy.onnx'
    
    # add prog, base lin vel, base pos z
    # policy_path = '/home/bcj/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-20_21-45-11_dance1_subject2-new_obs_add_prog-add_baselinvel-add_baseposz/exported/policy.onnx'
    parser = argparse.ArgumentParser(description='Deploy MuJoCo simulation with motion tracking')
    parser.add_argument('--motion_path', type=str, 
                       default='/home/bcj/whole_body_tracking/motions/lafan_dance1_subject2.npz',
                       help='Path to motion file (.npz)')
    parser.add_argument('--policy_path', type=str,
                       default='/home/bcj/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-20_21-45-11_dance1_subject2-new_obs_add_prog-add_baselinvel-add_baseposz/exported/policy.onnx',
                       help='Path to policy file (.onnx)')
    parser.add_argument('--obs_type', type=int, choices=[0, 1, 2], default=2,
                       help='Observation type (0, 1, or 2)')
    
    args = parser.parse_args()
    
    policy_path = args.policy_path
    motion_path = args.motion_path
    obs_type = args.obs_type
    # Select observation function based on obs_type
    # Select observation function based on obs_type
    observation_functions = {
        0: compute_observation_0,
        1: compute_observation_1,
        2: compute_observation_2
    }
    
    compute_observation = observation_functions.get(obs_type)
    if compute_observation is None:
        raise ValueError(f"Invalid obs_type: {obs_type}. Must be 0, 1, or 2")
    
    xml_path = '/home/bcj/BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml'
    
    simulation_duration = 1000
    simulation_dt = 0.002
    control_decimation = 10

    kps = np.array(kps, dtype=np.float32)
    kds = np.array(kds, dtype=np.float32)

    # 根据 isaaclab joint_pos 配置设置默认关节位置
    joint_pos_config = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }

    def get_joint_default_pos(joint_name):
        for pattern, pos in joint_pos_config.items():
            if pattern.startswith('.*'):
                if joint_name.startswith('left') or joint_name.startswith('right'):
                    if joint_name.endswith(pattern[3:]):
                        return pos
            else:
                if joint_name == pattern:
                    return pos
        return 0.0  # 默认值为0

    default_angles = np.array([get_joint_default_pos(name) for name in mujoco_joint_names], dtype=np.float32)
    print('default joint pos:')
    print(default_angles)

    num_actions = 29

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    last_action = action.copy()

    counter = 0
    inner_counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    viewer = mujoco_viewer.MujocoViewer(m, d)
    # breakpoint()
    m.opt.timestep = simulation_dt

    # load policy
    
    session = ort.InferenceSession(policy_path)
    obs_name = session.get_inputs()[0].name
    time_step_name = session.get_inputs()[1].name

    time_step = torch.zeros(1,1)
    motion_loader = MotionLoader(motion_path)
    
    d.qpos[7:] = default_angles
    mujoco.mj_step(m, d)
    
    joint_pos = np.zeros(29, dtype=np.float32)
    joint_vel = np.zeros(29, dtype=np.float32)
    ref_ang_vel = np.zeros(3, dtype=np.float32)
    
    # with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_alive and time.time() - start < simulation_duration:
        step_start = time.time()
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        if VIEW_MOTION:
            d.qpos[:3] = motion_loader.body_pos[inner_counter][0]
            d.qpos[3:7] = motion_loader.body_ori[inner_counter][0]
            d.qpos[7:] = motion_loader.joint_pos[inner_counter][isaaclab_to_mujoco_reindex]
            d.qvel[6:] = motion_loader.joint_vel[inner_counter][isaaclab_to_mujoco_reindex]
        else:
            d.ctrl[:] = tau
        
        # breakpoint()
        mujoco.mj_step(m, d)
        counter += 1
        if counter % control_decimation == 0:
            
            # create observation
            # obs
            obs = compute_observation(d, motion_loader, inner_counter, action)
            obs[:29] = joint_pos
            obs[29:58] = joint_vel
            obs[61:64] = ref_ang_vel
            # obs[61:64] = get_ref_ang_vel(motion_loader, inner_counter)
            
            obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)  # obs 为一维或二维
            time_step = np.array([[inner_counter]], dtype=np.float32)  # 或 float，根据模型要求

            output = session.run(None, {obs_name: obs_tensor, time_step_name: time_step})
            action = output[0].squeeze()
            joint_pos = output[1].squeeze()
            joint_vel = output[2].squeeze()
            ref_ang_vel = output[6].squeeze()[7]
            
            print('onnx output:')
            print(ref_ang_vel)
            print('motion loader:')
            print(get_ref_ang_vel(motion_loader, inner_counter))
            
            # transform action to target_dof_pos
            target_dof_pos = action[isaaclab_to_mujoco_reindex] * action_scale + default_angles
            
            inner_counter += 1

        viewer.render()

        # Rudimentary time keeping, will drift relative to wall clock.
        # time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)
