#!/usr/bin/env python3
from gym_pusht.envs.pusht_image_env import PushTImageEnv
import numpy as np
import time
import cv2
import os
import threading
from datetime import datetime
import tqdm

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class PushTImageRunner(Node):
    def __init__(
        self,
        render_size: int = 96,
        fps: int = 10,
        action_topic: str = "pusht/action",
        state_topic: str = "pusht/state",
        image_topic: str = "pusht/image",
        init_topic: str = "pusht/init",
        enable_human_render: bool = True,
        perturb_level: float = 0.0,
        max_steps: int = 300,
        max_episodes: int = 10,
    ):
        super().__init__("pusht_image_runner")

        self.render_size = render_size
        self.fps = float(fps)

        self.action_topic = action_topic
        self.state_topic = state_topic
        self.image_topic = image_topic
        self.init_topic = init_topic

        self.enable_human_render = enable_human_render
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.perturb_level = perturb_level
        self.episode_count = 0

        # Env
        self.seed = 100000
        PushTImageEnv.metadata["video.frames_per_second"] = int(fps)
        self.env = PushTImageEnv(render_size=render_size, perturb_level=self.perturb_level)

        self.bridge = CvBridge()

        self._act_lock = threading.Lock()
        self._act_event = threading.Event()
        self.latest_ros_action = None

        # Recording 
        self.record_dir = "recordings"
        os.makedirs(self.record_dir, exist_ok=True)
        self.recording = False
        self.video_writer = None

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.sub_action = self.create_subscription(Float32MultiArray, self.action_topic, self._command_callback, qos_profile=qos)
        self.state_pub = self.create_publisher(Float32MultiArray, self.state_topic, qos_profile=qos)
        self.image_pub = self.create_publisher(Image, self.image_topic, qos_profile=qos)
        self.init_pub = self.create_publisher(Bool, self.init_topic, qos_profile=qos)

        self.get_logger().info(
            f"Started PushTImageRunner (action-driven step). "
            f"fps={fps}, action={action_topic}, state={state_topic}, image={image_topic}"
        )

    # -------- action callback (executor thread) --------
    def _command_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        try:
            a = np.array(msg.data[:2], dtype=np.float32)
            if np.any(~np.isfinite(a)):
                return
            with self._act_lock:
                self.latest_ros_action = np.clip(a, 0, self.env.window_size)
                self._act_event.set()
        except Exception:
            pass

    def _select_action(self, obs):
        with self._act_lock:
            a = self.latest_ros_action.copy() if self.latest_ros_action is not None else None
            self.latest_ros_action = None
            self._act_event.clear()
        return a if a is not None else obs["agent_pos"]

    def _publish_state(self, obs):
        msg = Float32MultiArray()
        msg.data = [float(obs["agent_pos"][0]), float(obs["agent_pos"][1])]
        self.state_pub.publish(msg)

    def _publish_image(self, obs):
        img_bgr = np.transpose((obs["image"] * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))[..., ::-1]
        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(msg)

    def _publish_init(self):
        self.init_pub.publish(Bool(data=True))

    def _step_env(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"Unexpected step() return length: {len(out)}")
        return obs, reward, done, info

    # -------- recording (streaming) --------
    def start_recording(self, filename=None):
        if self.recording:
            return
        filename = filename or f"recording_{self.seed}_{self.perturb_level}.mp4"
        h, w = self.env._render_frame("rgb_array").shape[:2]
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.record_dir, filename), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        self.recording = True

    def _record_frame(self):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(self.env._render_frame("rgb_array")[..., ::-1])

    def stop_recording(self):
        if not self.recording:
            return
        if self.video_writer is not None:
            self.video_writer.release()
        self.video_writer = None
        self.recording = False

    def close(self):
        self.stop_recording()
        try:
            self.env.close()
        except Exception:
            pass

    # -------- main episode runner (main thread) --------
    def run(self, action_timeout=5.0):
        all_max_rewards = []
        pbar = tqdm.tqdm(range(self.max_episodes), desc="Episodes")

        for episode in pbar:
            self._publish_init()
            self.env.seed(self.seed)
            print(f"Resetting environment with seed {self.seed}")
            obs = self.env.reset()

            if self.enable_human_render:
                self.env._render_frame("human")
            self._publish_state(obs)
            self._publish_image(obs)

            self._act_event.clear()
            with self._act_lock:
                self.latest_ros_action = None

            self.start_recording()
            max_reward = 0.0
            done = False
            step = 0

            while not done and step < self.max_steps:
                self._act_event.wait(timeout=action_timeout)
                action = self._select_action(obs)
                obs, reward, done, info = self._step_env(action)
                step += 1
                max_reward = max(max_reward, float(reward))

                self._publish_state(obs)
                self._publish_image(obs)
                if self.enable_human_render:
                    self.env._render_frame("human")
                self._record_frame()
                pbar.set_postfix({"reward": f"{reward:.4f}", "max": f"{max_reward:.4f}", "step": f"{step}/{self.max_steps}"})

            all_max_rewards.append(max_reward)
            self.stop_recording()
            self.seed += 1

        if all_max_rewards:
            mean_reward = np.mean(all_max_rewards)
            std_reward = np.std(all_max_rewards)
            stats = "\n".join([
                "="*50,
                f"All {self.max_episodes} episodes completed!",
                f"Mean max_reward: {mean_reward:.4f}",
                f"Std max_reward: {std_reward:.4f}",
                f"Individual max_rewards: {[f'{r:.4f}' for r in all_max_rewards]}",
                f"Policy: flow matching",
                f"perturb: {self.perturb_level}",
                f"mode: non_realtime_openloop",
                "="*50
            ])
            print(f"\n{stats}")
            with open(os.path.join(self.record_dir, "result.txt"), 'w') as f:
                f.write(stats + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = PushTImageRunner()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run(action_timeout=5.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
