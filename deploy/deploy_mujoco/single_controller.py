from bydmimic_controller import BydMimicController
from amp_controller import AMPController
from mujoco_sim import MujocoSimulator


if __name__ == '__main__':
    
    # byd_policy_path = '/home/bcj/BeyondMimic/logs/rsl_rl/old_motion_obs_v1/2025-09-01_21-58-35_obs_v1_fly_punch/exported/policy_15000.onnx'
    # byd_policy_path = '/home/bcj/BeyondMimic/models/2025-08-29_20-34-12_obs_v1_shaolinquanM3-GMR_models_policy_30000.onnx'

    byd_policy_path = '/home/bcj/BeyondMimic/models/2025-09-03_17-36-34_obs_v1_lafan_dance2_subject4_models_policy_15000.onnx'

    # byd_policy_path = '/home/bcj/BeyondMimic/models/2025-08-29_20-34-16_obs_v1_taiji01M3-GMR_models_policy_30000.onnx'
    
    # byd_policy_path = '/home/bcj/BeyondMimic/logs/rsl_rl/jump/2025-09-08_12-15-25_jump_1m/exported/policy_9000.onnx'
    
    # byd_policy_path = '/home/bcj/BeyondMimic/logs/rsl_rl/hjr/walk2/exported/policy_1000.onnx'
    
    # byd_policy_path = '/home/bcj/BeyondMimic/logs/rsl_rl/jump/2025-09-08_12-15-25_jump_1m/exported/policy_40000.onnx'
    
    # byd_policy_path = '/home/bcj/BeyondMimic/models/2025-09-07_23-10-14_obs_v1_lafan_dance1_subject2_models_policy_14000.onnx'
    
    # byd_policy_path = '/home/bcj/BeyondMimic/models/2025-09-08_00-29-51_obs_v1_lafan_fight1_subject3_models_policy_13000.onnx'
    
    # byd_policy_path = '/home/bcj/BeyondMimic/deploy_onboard/ckpts/2025-09-03_18-02-52_obs_v1_lafan_dance1_subject1_models_policy_15000.onnx'
    # byd_policy_path = '/home/bcj/BeyondMimic/logs/rsl_rl/jump/2025-09-09_10-02-30_jump_1m_small_mu/exported/policy_15000.onnx'
    byd_policy_path = '/home/lenovo/project/BeyondMimic/logs/rsl_rl/g1_flat/2025-09-10_18-50-10_climb_box/exported/policy_4000.onnx'
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--policy_path', required=True)
    args = args.parse_args()
    byd_policy_path = args.policy_path

    byd_controller = BydMimicController(byd_policy_path)
    
    amp_policy_path = '/home/lenovo/project/BeyondMimic/deploy_mujoco/ckpts/amp_slope_walk_stand.onnx'
    image_path = '/home/lenovo/project/BeyondMimic/deploy_mujoco/camera/depth_imgs_clean.npy'
    amp_controller = AMPController(amp_policy_path, image_path)
    simulator = MujocoSimulator(xml_path='/home/lenovo/project/BeyondMimic/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1_climb_box.xml')
    
    
    target_q = byd_controller.default_dof_pos
    kps = byd_controller.kps
    kds = byd_controller.kds
    byd_controller.reset()
    amp_controller.reset()
    simulator.reset()
    
    # motion controller
    controller = byd_controller
    
    # amp controller
    # controller = amp_controller
    
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