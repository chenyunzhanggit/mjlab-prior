import mujoco
import mujoco_viewer
from common.bydmimic_utils import *
import numpy as np
import joblib
import os
from datetime import datetime

class MujocoSimulator:
    """MuJoCo simulator - handles simulation, PD control, and rendering"""
    
    def __init__(self, xml_path: str='/home/bcj/BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml', dt: float = 0.002, control_decimation: int = 10):
        # Initialize MuJoCo
        self.xml_path = xml_path    
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.default_dof_pos = default_angles.copy()
        
        # Control parameters
        self.dt = dt
        self.control_decimation = control_decimation
        
        # Counters
        self.sim_step_counter = 0
        self.control_step_counter = 0
        
        # Control state
        self.is_running = False
        self.view_motion = False
        
        self.log = {
            'time_step':[],
            'target_q':[],
            'q':[],
            'dq':[],
            'tau':[]
        }
        self.use_log = True
        
    def reset(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.default_dof_pos
        mujoco.mj_step(self.model, self.data)
        
        # Reset counters
        self.sim_step_counter = 0
        self.control_step_counter = 0
        
    def step(self, target_q, kps, kds):
        """Advance simulation by one step"""
        tau = pd_control(target_q, self.data.qpos[7:], kps, 
                        np.zeros_like(kds), self.data.qvel[6:], kds)
        # Apply control torques
        self.data.ctrl[:] = tau
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        if self.use_log:
            self.log['time_step'].append(self.sim_step_counter)
            self.log['target_q'].append(target_q.copy())
            self.log['q'].append(self.data.qpos[7:].copy())
            self.log['dq'].append(self.data.qvel[6:].copy())
            self.log['tau'].append(tau.copy())

        self.sim_step_counter += 1
    
    def plot(self):
        import matplotlib.pyplot as plt

        if not self.log['time_step']:
            print("No data to plot")
            return

        time_steps = np.array(self.log['time_step'])
        target_q = np.array(self.log['target_q'])
        q = np.array(self.log['q'])
        dq = np.array(self.log['dq'])
        tau = np.array(self.log['tau'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Joint Data Over Time')

        # Plot target_q
        axes[0, 0].set_title('Target Joint Positions')
        for joint in range(29):
            axes[0, 0].plot(time_steps, target_q[:, joint], label=f'Joint {joint}')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Position (rad)')
        axes[0, 0].grid(True)

        # Plot q
        axes[0, 1].set_title('Actual Joint Positions')
        for joint in range(29):
            axes[0, 1].plot(time_steps, q[:, joint], label=f'Joint {joint}')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Position (rad)')
        axes[0, 1].grid(True)

        # Plot dq
        axes[1, 0].set_title('Joint Velocities')
        for joint in range(29):
            axes[1, 0].plot(time_steps, dq[:, joint], label=f'Joint {joint}')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Velocity (rad/s)')
        axes[1, 0].grid(True)

        # Plot tau
        axes[1, 1].set_title('Joint Torques')
        for joint in range(29):
            axes[1, 1].plot(time_steps, tau[:, joint], label=f'Joint {joint}')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Torque (Nm)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
    
    def save_log(self, dir):
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_log_{timestamp}.pkl"
        filepath = os.path.join(dir, filename)

        # Save the log using joblib
        joblib.dump(self.log, filepath)
        print(f"Log saved to: {filepath}")
        
    def should_run_control(self):
        """Check if controller should run (based on control decimation)"""
        return self.sim_step_counter % self.control_decimation == 0
        
    def render(self):
        """Render the simulation"""
        self.viewer.render()
        
    def is_alive(self):
        """Check if viewer is still alive"""
        return self.viewer.is_alive