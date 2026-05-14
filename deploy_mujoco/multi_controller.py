from typing import List

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from pynput import keyboard
from bydmimic_controller import BydMimicController
from amp_controller import AMPController
from mujoco_sim import MujocoSimulator
from base_controller import BaseController

PLOT = True
SAVE_LOG = True

if __name__ == "__main__":
    
    amp_policy_path = '/home/lenovo/project/BeyondMimic/deploy_mujoco/ckpts/amp_slope_walk_stand.onnx'
    image_path = '/home/lenovo/project/BeyondMimic/deploy_mujoco/camera/depth_imgs_clean.npy'
    amp_controller = AMPController(amp_policy_path, image_path)
    simulator = MujocoSimulator(xml_path='/home/lenovo/project/BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1_climb_box.xml')
    
    # Control state using a simple class
    class MainController:
        def __init__(self):
            self.cur_controller: BaseController = None
            self.cur_idx = 0
            self.controller_list: List[BaseController] = []
        
        def add_controller(self, controller):
            self.controller_list.append(controller)
        
        def set_controller(self, idx):
            self.cur_controller = self.controller_list[idx]
        
        def reset_all_controllers(self):
            for controller in self.controller_list:
                controller.reset()
            self.set_controller(self.cur_idx)
        
        def check_motion_end(self):
            if self.cur_controller is not None and isinstance(self.cur_controller, BydMimicController):
                if self.cur_controller.time_step >= self.cur_controller.max_time_step - 1:
                    print('Motion ended, switched to default policy')
                    self.cur_idx = 0
                    self.set_controller(self.cur_idx)

        @property
        def num_controllers(self):
            return len(self.controller_list)

    # byd_policy_path_1 = '/home/bcj/BeyondMimic/logs/rsl_rl/g1_flat/2025-08-20_10-17-53_dance1_subject2-new_obs_add_prog-motion_global_root_pos_2/exported/policy_7000.onnx'
    # byd_controller_1 = BydMimicController(byd_policy_path_1)
    
    # byd_policy_path_2 = '/home/bcj/BeyondMimic/models/2025-09-03_17-36-43_obs_v1_lafan_fall_and_getup1_subject1_models_policy_15000.onnx'
    # byd_controller_2 = BydMimicController(byd_policy_path_2)
    
    # byd_policy_path_3 = '/home/bcj/BeyondMimic/logs/rsl_rl/old_motion_obs_v1/2025-09-01_21-58-35_obs_v1_fly_punch/exported/policy_15000.onnx'
    # byd_controller_3 = BydMimicController(byd_policy_path_3)

    
    main_controller = MainController()
    main_controller.add_controller(amp_controller)
    
    # byd_path = '/home/bcj/BeyondMimic/models/2025-09-03_17-36-34_obs_v1_lafan_dance2_subject4_models_policy_15000.onnx'
    # byd_path = '/home/bcj/BeyondMimic/logs/rsl_rl/hjr/walk2/exported/policy_1000.onnx'
    
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--byd_policy_path', required=True)
    args = args.parse_args()
    byd_policy_path = args.byd_policy_path
    
    main_controller.add_controller(BydMimicController(byd_policy_path))

    def on_key_press(key):
        try:
            # Handle special keys like arrow keys
            if key.char == '[':
                main_controller.cur_idx = (main_controller.cur_idx - 1) % main_controller.num_controllers
                print(f'Current controller candidate: {main_controller.cur_idx}')
            elif key.char == ']':
                main_controller.cur_idx = (main_controller.cur_idx + 1) % main_controller.num_controllers
                print(f'Current controller candidate: {main_controller.cur_idx}')
            elif key.char == '/':
                main_controller.set_controller(main_controller.cur_idx)
                main_controller.cur_controller.reset()
                print(f'Switched to controller {main_controller.cur_idx}')
            elif key.char == 'r':
                simulator.reset()
                main_controller.reset_all_controllers()
        except AttributeError:
            pass

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    
    # Initialize simulation
    simulator.reset()
    main_controller.reset_all_controllers()

    target_q = main_controller.cur_controller.default_dof_pos
    kps = main_controller.cur_controller.kps
    kds = main_controller.cur_controller.kds

    try:
        while simulator.is_alive():
            main_controller.check_motion_end()
            if simulator.should_run_control():
                target_q, kps, kds = main_controller.cur_controller.step(simulator.data)
                
            simulator.step(target_q, kps, kds)
            simulator.render()

    except KeyboardInterrupt:
        print("Simulation interrupted")
    finally:
        listener.stop()
        print("Simulation ended")
    
    if PLOT:
        simulator.plot()
    
    if SAVE_LOG:
        simulator.save_log('/home/bcj/BeyondMimic/deploy_mujoco/logs')