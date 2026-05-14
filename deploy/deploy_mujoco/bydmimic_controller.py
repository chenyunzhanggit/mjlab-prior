import numpy as np
import onnxruntime as ort
from common.bydmimic_utils import *
from mujoco_sim import MujocoSimulator
from base_controller import BaseController

class BydMimicController(BaseController):
    def __init__(self, policy_path: str):
        # Load policy
        self.session = ort.InferenceSession(policy_path)
        self.obs_name = self.session.get_inputs()[0].name
        self.time_step_name = self.session.get_inputs()[1].name
        self.default_dof_pos = default_angles.copy()
        
        # get max time step
        output = self.session.run(None, {self.obs_name: np.zeros(154, dtype=np.float32).reshape(1, -1), 
                                       self.time_step_name: np.array([[0]], dtype=np.float32)})
        # print(output[-1])
        self.max_time_step = 420 #int(output[-1][0][0])
        
        # Control variables
        self.num_actions = 29
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.joint_pos = np.zeros(29, dtype=np.float32)
        self.joint_vel = np.zeros(29, dtype=np.float32)
        self.ref_ang_vel = np.zeros(3, dtype=np.float32)
        
        # Time step counter for policy
        self.time_step = 0
        
        # set kp, kd
        self.kps = kps
        self.kds = kds
        self.action_scale = action_scale
        
    def reset(self):
        self.action.fill(0)
        self.joint_pos.fill(0)
        self.joint_vel.fill(0)
        self.ref_ang_vel.fill(0)
        self.time_step = 0
        
    def step(self, mujoco_data):
        """
        Compute target DOF positions from mujoco data and time step
        
        Args:
            mujoco_data: MuJoCo data object containing current simulation state
            time_step: Current time step
            
        Returns:
            target_dof_pos: Target joint positions (29,) array
        """
        # Create observation
        obs = np.zeros(154, dtype=np.float32)
        
        # Update observation with current state
        obs[:29] = self.joint_pos
        obs[29:58] = self.joint_vel
        obs[58:61] = get_gravity_orientation(mujoco_data.qpos[3:7])
        obs[61:64] = self.ref_ang_vel
        obs[64:67] = mujoco_data.qvel[3:6]  # base angular velocity
        obs[67:96] = (mujoco_data.qpos[7:36] - default_angles)[mujoco_to_isaaclab_reindex]
        obs[96:125] = mujoco_data.qvel[6:35][mujoco_to_isaaclab_reindex]
        obs[125:154] = self.action
        
        # Prepare inputs for ONNX model
        obs = np.array(obs, dtype=np.float32).reshape(1, -1)
        time_step = np.array([[self.time_step]], dtype=np.float32)
        
        # Run inference
        output = self.session.run(None, {self.obs_name: obs, 
                                       self.time_step_name: time_step})
        
        # Update state variables
        self.action = output[0].squeeze()
        self.joint_pos = output[1].squeeze()
        self.joint_vel = output[2].squeeze()
        self.ref_ang_vel = output[6].squeeze()[7]

        # Transform action to target joint positions
        target_q = (self.action[isaaclab_to_mujoco_reindex] * 
                         action_scale + default_angles)
        
        self.time_step += 1
        return target_q, self.kps, self.kds


if __name__ == "__main__":

    policy_path = '/home/bcj/BeyondMimic/logs/rsl_rl/g1_flat/2025-08-20_10-17-53_dance1_subject2-new_obs_add_prog-motion_global_root_pos_2/exported/policy_7000.onnx'
    
    controller = BydMimicController(policy_path)
    simulator = MujocoSimulator()
    
    target_q = controller.default_dof_pos
    kps = controller.kps
    kds = controller.kds
    
    simulator.reset()
    controller.reset()

    try:
        while simulator.is_alive():
            if simulator.should_run_control():
                target_q, kps, kds = controller.step(simulator.data)
                
            simulator.step(target_q, kps, kds)                
            simulator.render()

    except KeyboardInterrupt:
        print("Simulation interrupted")
    finally:
        print("Simulation ended")