import onnxruntime as ort
import numpy as np
from common.bydmimic_utils import *
from base_controller import BaseController

class AMPController(BaseController):
    def __init__(self, policy_path: str, image_path: str='/home/bcj/BeyondMimic/deploy_mujoco/camera/depth_imgs_clean.npy'):
        # Load policy
        self.session = ort.InferenceSession(policy_path)
        self.obs_name = self.session.get_inputs()[0].name
        self.his_name = self.session.get_inputs()[1].name
        self.depth_name = self.session.get_inputs()[2].name
        
        self.num_obs = 57        
        self.num_actions = 29
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        # fix image for amp policy
        self.depth_image = np.zeros((1,2,64,64)).astype(np.float32)
        self.warp_depth = np.load(image_path)[0]
        self.depth_image[:, 0] = self.warp_depth
        self.depth_image[:, 1] = self.warp_depth
        self.history = np.zeros((1, 10, self.num_obs)).astype(np.float32)
        
        self.kps = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
                            100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 
                            400.0, 400.0, 400.0, 
                            100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0, 
                            100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0],dtype=np.float32)
        self.kds = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 
                            2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 
                            5.0, 5.0, 5.0, 
                            2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 
                            2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,],dtype=np.float32)
        self.action_scale = 0.25
        self.default_dof_pos = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                0.0, 0.0, 0.0,
                                0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0,
                                0.2, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0], dtype=np.float32)
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.cmd_scale = [2.0, 2.0, 0.25]
        self.max_cmd = [1.2, 0.8, 1.5]
        self.used_joint_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 18, 19, 22]

        self.cmd = np.array([0., 0., 0.], dtype=np.float32)

    def reset(self):
        self.action.fill(0)
        self.history = np.zeros((1, 10, self.num_obs)).astype(np.float32)
        self.cmd = np.array([0., 0., 0.], dtype=np.float32)
    
    def set_cmd(self, cmd):
        self.cmd = cmd

    def step(self, mujoco_data):
        
        q = mujoco_data.qpos[7:]
        dq = mujoco_data.qvel[6:]

        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[0:3] = self.cmd[:3] * self.cmd_scale * self.max_cmd
        obs[3:6] = mujoco_data.qvel[3:6] * self.ang_vel_scale
        obs[6:9] = get_gravity_orientation(mujoco_data.qpos[3:7])
        obs[9:25] = (q - self.default_dof_pos)[self.used_joint_idx] * self.dof_pos_scale
        obs[25:41] = dq[self.used_joint_idx] * self.dof_vel_scale
        obs[41:57] = self.action[self.used_joint_idx]
        
        obs = obs[np.newaxis, :].astype(np.float32)
        self.history = np.concatenate([self.history[:, 1:], obs[np.newaxis, :]], axis=1)
        output = self.session.run(
                None, {self.obs_name: obs, self.his_name: self.history.reshape(1, 10 * self.num_obs), self.depth_name: self.depth_image}
            )
        self.action[self.used_joint_idx] = output[0].squeeze()
        
        target_q = self.action * self.action_scale + self.default_dof_pos
        
        return target_q, self.kps, self.kds

if __name__ == "__main__":
    
    amp_policy_path = '/home/bcj/BeyondMimic/deploy_mujoco/ckpts/amp_slope_walk_stand.onnx'
    image_path = '/home/bcj/BeyondMimic/deploy_mujoco/camera/depth_imgs_clean.npy'

    controller = AMPController(amp_policy_path, image_path)
    
    from mujoco_sim import MujocoSimulator
    simulator = MujocoSimulator()
    
    target_q = controller.default_dof_pos
    kps = controller.kps
    kds = controller.kds
    
    # Initialize simulation
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