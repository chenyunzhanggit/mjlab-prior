import os
import sys
import argparse
import glob
import numpy as np
import mujoco
import mujoco.viewer
import threading
import time

try:
  from pynput import keyboard

  KEYBOARD_AVAILABLE = True
except ImportError:
  KEYBOARD_AVAILABLE = False
  print("警告: pynput 库未安装，暂停功能将不可用。请运行: pip install pynput")


def load_npz_data(npz_path):
  """加载 NPZ 动作数据，返回字典。"""
  data = np.load(npz_path, allow_pickle=True)
  result = {
    "fps": float(data["fps"].item())
    if data["fps"].size == 1
    else float(data["fps"][0]),
    "joint_pos": data["joint_pos"],
    "joint_vel": data.get("joint_vel", None),
    "body_pos_w": data.get("body_pos_w", None),
    "body_quat_w": data.get("body_quat_w", None),
  }
  data.close()
  return result


def visualize_npz_sequence(
  model_path,
  npz_path,
  save_gif=False,
  gif_path=None,
  loop=False,
):
  """
  可视化单个 NPZ 动作序列。

  NPZ 文件需包含:
      - joint_pos: (num_frames, num_joints)
      - body_pos_w: (num_frames, num_bodies, 3)  [可选但建议提供，用于根位置]
      - body_quat_w: (num_frames, num_bodies, 4) [可选但建议提供，用于根朝向]
      - fps: scalar

  控制:
      按空格键: 暂停/继续播放
      按 'n' 键: 跳到下一个文件 (仅文件夹模式)
      按 'q' 键: 退出
  """
  # 1. 加载 MuJoCo 模型
  model = mujoco.MjModel.from_xml_path(model_path)
  data = mujoco.MjData(model)

  # 2. 加载 NPZ 数据
  npz_data = load_npz_data(npz_path)
  joint_pos = npz_data["joint_pos"]
  body_pos_w = npz_data["body_pos_w"]
  body_quat_w = npz_data["body_quat_w"]
  fps = npz_data["fps"]

  num_frames = joint_pos.shape[0]
  num_joints = joint_pos.shape[1]
  print(f"加载 NPZ: {os.path.basename(npz_path)}")
  print(f"  帧数: {num_frames}, 关节数: {num_joints}, FPS: {fps}")

  # 3. 检查维度匹配
  has_freejoint = model.nq > num_joints
  if has_freejoint:
    root_dof = model.nq - num_joints
    if root_dof != 7:
      print(
        f"警告: 模型 nq={model.nq}, 关节数={num_joints}, "
        f"根自由度={root_dof} 不是 7，尝试直接映射..."
      )
    if body_pos_w is None or body_quat_w is None:
      print(
        "警告: NPZ 中缺少 body_pos_w / body_quat_w，根位姿将保持默认 (可能导致错位)。"
      )
  else:
    if num_joints != model.nq:
      raise ValueError(f"动作数据关节数({num_joints})与模型关节数({model.nq})不匹配！")

  # 4. 计算每帧间隔
  frame_dt = 1.0 / fps if fps > 0 else 0.02

  # 5. 暂停/控制变量
  paused = False
  stop_requested = False
  next_requested = False
  pause_lock = threading.Lock()

  # 6. 键盘监听函数
  def on_press(key):
    nonlocal paused, stop_requested, next_requested
    try:
      if key == keyboard.Key.space:
        with pause_lock:
          paused = not paused
          print("\n[暂停]" if paused else "\n[继续播放]")
      elif hasattr(key, "char") and key.char == "n":
        with pause_lock:
          next_requested = True
          print("\n[跳到下一个]")
      elif hasattr(key, "char") and key.char == "q":
        with pause_lock:
          stop_requested = True
          print("\n[退出]")
    except AttributeError:
      pass

  # 7. 启动键盘监听器
  listener = None
  if KEYBOARD_AVAILABLE:
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("提示: 按空格键暂停/继续, 'n' 下一个文件, 'q' 退出")
  else:
    print("提示: 键盘监听不可用")

  # 8. 创建可视化窗口
  try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
      # 设置视图参数
      viewer.cam.azimuth = 135
      viewer.cam.elevation = -20
      viewer.cam.distance = 3.0

      frames = []
      file_finished = False

      while not file_finished:
        for i in range(num_frames):
          # 检查控制状态
          with pause_lock:
            is_paused = paused
            should_stop = stop_requested
            should_next = next_requested

          if should_stop:
            file_finished = True
            break

          if should_next:
            next_requested = False
            file_finished = True
            break

          if is_paused:
            viewer.sync()
            time.sleep(0.05)
            continue

          # 设置根位姿 (前 7 个 DOF: pos + quat)
          if has_freejoint and body_pos_w is not None and body_quat_w is not None:
            # body_pos_w[:, 0] 对应 pelvis (第一个非 world body)
            data.qpos[:3] = body_pos_w[i, 0]
            data.qpos[3:7] = body_quat_w[i, 0]
            data.qpos[7:] = joint_pos[i]
          else:
            data.qpos[:] = joint_pos[i]

          data.qvel[:] = 0.0
          mujoco.mj_step(model, data)
          viewer.sync()

          if save_gif:
            frames.append(viewer.render())

          print(f"播放进度: {i + 1}/{num_frames}", end="\r")
          time.sleep(frame_dt)

        if not file_finished:
          if loop:
            print("\n[循环播放]")
          else:
            file_finished = True

      print(f"\n播放完成: {os.path.basename(npz_path)}")

      # 保存 GIF
      if save_gif and gif_path and frames:
        print(f"正在保存 GIF 到: {gif_path}")
        try:
          import imageio

          imageio.mimsave(gif_path, frames, fps=int(fps))
          print(f"GIF 保存成功! {gif_path}")
        except Exception as e:
          print(f"保存 GIF 失败: {e}")

      print("可视化完成!")
  finally:
    if listener is not None:
      listener.stop()


def main():
  parser = argparse.ArgumentParser(
    description="可视化 NPZ 机器人动作数据 (支持单个文件或文件夹)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
使用示例:
  python visualize_npz.py data.npz
  python visualize_npz.py /path/to/folder/
  python visualize_npz.py data.npz --model g1.xml --save-gif
  python visualize_npz.py folder/ --loop
        """,
  )

  parser.add_argument(
    "input_path",
    type=str,
    help="NPZ 文件路径或包含 NPZ 文件的文件夹路径",
  )

  parser.add_argument(
    "--model",
    type=str,
    default="/home/lenovo/project/AMP_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml",
    help="MuJoCo 模型 XML 文件路径 (默认: %(default)s)",
  )

  parser.add_argument(
    "--save-gif",
    action="store_true",
    help="是否保存为 GIF 动画",
  )

  parser.add_argument(
    "--gif-path",
    type=str,
    default=None,
    help="GIF 保存路径 (默认使用输入文件名.gif)",
  )

  parser.add_argument(
    "--loop",
    action="store_true",
    help="是否循环播放单个文件",
  )

  args = parser.parse_args()

  # 收集 NPZ 文件列表
  if os.path.isfile(args.input_path):
    npz_files = [args.input_path]
  elif os.path.isdir(args.input_path):
    npz_files = sorted(glob.glob(os.path.join(args.input_path, "*.npz")))
    if not npz_files:
      print(f"错误: 文件夹中未找到 .npz 文件: {args.input_path}")
      sys.exit(1)
    print(f"找到 {len(npz_files)} 个 NPZ 文件")
  else:
    print(f"错误: 路径不存在: {args.input_path}")
    sys.exit(1)

  # 逐个播放
  for idx, npz_path in enumerate(npz_files):
    gif_path = args.gif_path
    if args.save_gif and gif_path is None:
      base_name = os.path.splitext(npz_path)[0]
      gif_path = f"{base_name}.gif"

    try:
      visualize_npz_sequence(
        model_path=args.model,
        npz_path=npz_path,
        save_gif=args.save_gif,
        gif_path=gif_path,
        loop=args.loop and len(npz_files) == 1,
      )
    except KeyboardInterrupt:
      print("\n用户中断")
      break
    except Exception as e:
      print(f"\n播放 {npz_path} 时出错: {e}")
      import traceback

      traceback.print_exc()
      continue

    # 文件夹模式下，播完一个暂停一下提示用户
    if len(npz_files) > 1 and idx < len(npz_files) - 1:
      print(f"\n[{idx + 1}/{len(npz_files)}] 已播放完成，准备播放下一个...")


if __name__ == "__main__":
  main()
