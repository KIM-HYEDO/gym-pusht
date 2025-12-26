#!/usr/bin/env python3
from gym_pusht.envs.pusht_image_env import PushTImageEnv
import pygame
import numpy as np
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class PushTImageRealtimeRunner(Node):
    def __init__(
        self,
        render_size: int = 96,
        fps: int = 30,
        action_topic: str = "pusht/action",
        state_topic: str = "pusht/state",
        image_topic: str = "pusht/image",
    ):
        super().__init__("pusht_image_realtime_runner")

        self.render_size = render_size
        self.fps = fps
        self.action_topic = action_topic
        self.state_topic = state_topic
        self.image_topic = image_topic

        # Env
        PushTImageEnv.metadata["video.frames_per_second"] = fps
        self.env = PushTImageEnv(render_size=render_size, perturb_level=1.0)
        self.obs = self.env.reset()
        self.env._render_frame("human")

        self.latest_ros_action = None
        self.agent = self.env.teleop_agent()
        self.running = True

        self.bridge = CvBridge()

        self.sub_action = self.create_subscription(
            Float32MultiArray,
            self.action_topic,
            self._command_callback,
            1,
        )

        self.state_pub = self.create_publisher(Float32MultiArray, self.state_topic, 1)

        self.image_pub = self.create_publisher(Image, self.image_topic, 1)

        self.get_logger().info(
            f"Started PushTImageRealtimeRunner (fps={fps}) "
            f"action={action_topic}, state={state_topic}, image={image_topic}"
        )

    def _command_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        action_array = np.array(msg.data[:2], dtype=np.float32)
        action_array = np.clip(action_array, 0, self.env.window_size)
        self.latest_ros_action = action_array

    def _select_action(self):
        if self.latest_ros_action is not None:
            action = self.latest_ros_action
            self.latest_ros_action = None
            return action

        action = self.agent.act(self.obs)
        if action is None:
            action = self.obs["agent_pos"]
        return action

    def _publish_state(self, obs):
        msg = Float32MultiArray()
        agent_pos = obs["agent_pos"]
        msg.data = [float(agent_pos[0]), float(agent_pos[1])]
        self.state_pub.publish(msg)

    def _publish_image(self, obs):
        img = obs["image"]  # (3, H, W) float, 0~1
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        img_bgr = np.transpose(img_uint8, (1, 2, 0))[:, :, ::-1]  # HWC, BGR

        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(msg)

    def step_once(self):
        # pygame 창 이벤트 처리 (닫기 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

        action = self._select_action()
        self.obs, reward, done, info = self.env.step(action)
        self.env._render_frame("human")

        self._publish_state(self.obs)
        self._publish_image(self.obs)

        if done:
            self.obs = self.env.reset()

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PushTImageRealtimeRunner()
    period = 1.0 / float(node.fps)

    try:
        while rclpy.ok() and node.running:
            t0 = time.perf_counter()

            rclpy.spin_once(node, timeout_sec=0.0)
            node.step_once()

            elapsed = time.perf_counter() - t0
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
