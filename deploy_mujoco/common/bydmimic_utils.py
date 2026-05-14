import numpy as np
# Joint names and mappings
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

# Parameter dictionaries
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

# Joint position config for default angles
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

def get_param(joint_name, param_dict):
    """Get parameter value for a joint from parameter dictionary"""
    for pattern, value in param_dict.items():
        if pattern.startswith('.*'):
            if joint_name.startswith('left') or joint_name.startswith('right'):
                if joint_name.endswith(pattern[3:]):
                    return value
        else:
            if joint_name == pattern:
                return value
    raise ValueError(f"No value found for joint: {joint_name}")

def get_joint_default_pos(joint_name):
    """Get default position for a joint"""
    for pattern, pos in joint_pos_config.items():
        if pattern.startswith('.*'):
            if joint_name.startswith('left') or joint_name.startswith('right'):
                if joint_name.endswith(pattern[3:]):
                    return pos
        else:
            if joint_name == pattern:
                return pos
    return 0.0  # Default value

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position commands using PD control"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# Create reindex arrays
isaaclab_to_mujoco_reindex = [isaaclab_joint_names.index(name) for name in mujoco_joint_names]
mujoco_to_isaaclab_reindex = [mujoco_joint_names.index(name) for name in isaaclab_joint_names]

# Create parameter arrays
kps = np.array([get_param(name, stiffness_dict) for name in mujoco_joint_names], dtype=np.float32)
kds = np.array([get_param(name, damping_dict) for name in mujoco_joint_names], dtype=np.float32)
action_scale = np.array([get_param(name, scale_dict) for name in mujoco_joint_names], dtype=np.float32)
default_angles = np.array([get_joint_default_pos(name) for name in mujoco_joint_names], dtype=np.float32)

def get_gravity_orientation(quaternion):
    """Get gravity orientation from quaternion"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation