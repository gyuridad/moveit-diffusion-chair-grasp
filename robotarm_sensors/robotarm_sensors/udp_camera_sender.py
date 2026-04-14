#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import socket
import struct
from collections import deque
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from robotarm_common.chair_grasp_common import make_frame_payload, send_udp_chunks 


"""
Isaac Sim ROS2 RGB / Depth / CameraInfo 토픽을 받아
UDP로 frame payload를 chunk 분할 전송하는 송신 노드

입력:
- /rgb            : sensor_msgs/Image
- /depth          : sensor_msgs/Image
- /camera_info    : sensor_msgs/CameraInfo

출력:
- UDP frame payload (RGB + Depth + CameraInfo)

주의:
- 수신측은 chair_grasp_common.make_frame_payload() 형식과 호환되어야 함
- 즉, image_socket_server 쪽도 같은 payload 규약을 해석해야 함
"""

# 이미지를 축소/확대했을 때, 카메라 내부 파라미터도 그 크기에 맞게 같이 바꿔주는 함수
def resize_camera_info(msg: CameraInfo, scale: float) -> CameraInfo:
    resized = CameraInfo()    # 원본 msg를 직접 바꾸지 않고 resized라는 새 객체를 만듭니다.
    resized.header = msg.header
    resized.height = max(1, int(round(msg.height * scale)))
    resized.width = max(1, int(round(msg.width * scale)))
    resized.distortion_model = msg.distortion_model
    resized.d = list(msg.d)
    resized.k = list(msg.k)
    resized.r = list(msg.r)
    resized.p = list(msg.p)

    resized.k[0] *= scale
    resized.k[2] *= scale
    resized.k[4] *= scale
    resized.k[5] *= scale
    resized.p[0] *= scale
    resized.p[2] *= scale
    resized.p[5] *= scale
    resized.p[6] *= scale
    return resized
        # 카메라 행렬의 핵심 값만 scale에 맞게 조정
        #     k[0], k[4] : 초점거리 fx, fy
        #     k[2], k[5] : 주점 cx, cy
        #     p[0], p[5], p[2], p[6] 도 같은 이유로 조정


class IsaacRgbdUdpSender(Node):
    def __init__(self):
        super().__init__("isaac_rgbd_udp_sender")

        # =========================
        # Parameters
        # =========================
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")

        self.declare_parameter("dest_ip", "127.0.0.1")
        self.declare_parameter("dest_port", 9002)
        self.declare_parameter("send_hz", 1.0)   # 보내는 횟수
        self.declare_parameter("max_chunk_payload", 12000)  # 한 번에 자르는 조각 크기
            # 예전 값 10, 1200에서
            # 수정 값 1.0, 12000으로 바뀌었다는 건,
            #     “자주 잘게 보내던 방식”에서 → “덜 자주, 크게 보내는 방식”으로 바뀐 거예요.
            
        self.declare_parameter("resize_scale", 0.5)
        self.declare_parameter("camera_frame_override", "")

        self.declare_parameter("qos_reliable", True)
        self.declare_parameter("qos_depth", 20)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.dest_ip = self.get_parameter("dest_ip").value
        self.dest_port = int(self.get_parameter("dest_port").value)
        self.send_hz = float(self.get_parameter("send_hz").value)
        self.max_chunk_payload = int(self.get_parameter("max_chunk_payload").value)
        self.resize_scale = float(self.get_parameter("resize_scale").value)
        self.camera_frame_override = str(self.get_parameter("camera_frame_override").value)
        self.qos_reliable = bool(self.get_parameter("qos_reliable").value)
        self.qos_depth = int(self.get_parameter("qos_depth").value)

        # =========================
        # QoS
        # =========================
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE
            if self.qos_reliable
            else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.qos_depth,
        )

        qos_caminfo = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE
            if self.qos_reliable
            else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # =========================
        # UDP
        # =========================
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))

        # =========================
        # Buffers
        # =========================
        self.bridge = CvBridge()

        self.rgb_queue = deque(maxlen=1)
        self.depth_queue = deque(maxlen=1)

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_encoding: Optional[str] = None
        self.latest_camera_info: Optional[CameraInfo] = None

        self.frame_id = 0

        # =========================
        # ROS interfaces
        # =========================
        self.sub_rgb = self.create_subscription(
            Image,
            self.rgb_topic,
            self.cb_rgb,
            qos_img
        )

        self.sub_depth = self.create_subscription(
            Image,
            self.depth_topic,
            self.cb_depth,
            qos_img
        )

        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.cb_camera_info,
            qos_caminfo
        )

        timer_period = 1.0 / max(1e-6, self.send_hz)
        self.timer = self.create_timer(timer_period, self.on_timer_send)

        self.get_logger().info(
            f"[IsaacRgbdUdpSender] "
            f"rgb={self.rgb_topic}, depth={self.depth_topic}, camera_info={self.camera_info_topic}, "
            f"udp={self.dest_ip}:{self.dest_port}, send_hz={self.send_hz:.2f}, "
            f"chunk={self.max_chunk_payload}, resize_scale={self.resize_scale:.2f}, "
            f"qos={'RELIABLE' if self.qos_reliable else 'BEST_EFFORT'}"
        )

    # =========================================================
    # RGB callback
    # =========================================================
    def cb_rgb(self, msg: Image):
        try:
            enc = msg.encoding.lower()

            if enc == "rgb8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                rgb = img.copy()

            elif enc == "bgr8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            elif enc == "rgba8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 4
                )
                rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            elif enc == "bgra8":
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 4
                )
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self.rgb_queue.append((rgb, stamp_sec))

        except Exception as e:
            self.get_logger().warn(f"cb_rgb error: {e}")

    # =========================================================
    # Depth callback
    # =========================================================
    def cb_depth(self, msg: Image):
        try:
            self.latest_depth_encoding = msg.encoding  # “이번에 받은 depth 이미지의 인코딩 이름을 그대로 보관한다”

            if msg.encoding.upper() == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
                    msg.height, msg.width
                ).copy()

            elif msg.encoding.upper() == "16UC1":
                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width
                ).copy()
                depth = raw.astype(np.float32) / 1000.0
            else:
                converted = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth = converted.astype(np.float32)

            stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self.depth_queue.append((depth, stamp_sec, msg.encoding))

        except Exception as e:
            self.get_logger().warn(f"cb_depth error: {e}")

    # =========================================================
    # CameraInfo callback
    # =========================================================
    def cb_camera_info(self, msg: CameraInfo):
        self.latest_camera_info = msg

    # =========================================================
    # Timer send
    # =========================================================
    def on_timer_send(self):
        if not self.rgb_queue or not self.depth_queue or self.latest_camera_info is None:
            return
        
        try:
            rgb, rgb_stamp = self.rgb_queue[-1]
            depth, depth_stamp, depth_encoding = self.depth_queue[-1]

            self.latest_rgb = rgb
            self.latest_depth = depth
            self.latest_depth_encoding = depth_encoding

            camera_info_msg = self.latest_camera_info
            if self.resize_scale != 1.0:
                target_w = max(1, int(round(rgb.shape[1] * self.resize_scale)))
                target_h = max(1, int(round(rgb.shape[0] * self.resize_scale)))
                self.latest_rgb = cv2.resize(self.latest_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
                self.latest_depth = cv2.resize(self.latest_depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                camera_info_msg = resize_camera_info(self.latest_camera_info, self.resize_scale)

            payload = make_frame_payload(
                rgb=self.latest_rgb,
                depth=self.latest_depth,
                camera_info={
                    "k": list(camera_info_msg.k),
                    "p": list(camera_info_msg.p),  # Projection Matrix (투영 행렬)
                    "d": list(camera_info_msg.d),
                    "width": int(camera_info_msg.width),
                    "height": int(camera_info_msg.height),
                    "frame_id": self.camera_frame_override or camera_info_msg.header.frame_id,
                    "depth_encoding": self.latest_depth_encoding,
                    "rgb_stamp": rgb_stamp,
                    "depth_stamp": depth_stamp,
                },
                t_world_camera=None,
                stamp=self.get_clock().now().nanoseconds * 1e-9,
            )

            send_udp_chunks(
                sock=self.sock,
                target=(self.dest_ip, self.dest_port),
                payload=payload,
                frame_id=self.frame_id,
                max_payload=self.max_chunk_payload,
            )

            self.get_logger().debug(
                f"sent frame_id={self.frame_id}, "
                f"rgb_shape={self.latest_rgb.shape}, "
                f"depth_shape={self.latest_depth.shape}, "
                f"depth_encoding={self.latest_depth_encoding}"
            )

            self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF

        except Exception as e:
            self.get_logger().warn(f"on_timer_send error: {e}")


    # =========================================================
    # Cleanup
    # =========================================================
    def destroy_node(self):
        try:
            self.sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IsaacRgbdUdpSender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



### 실행예시
# ros2 run robotarm_sensors udp_camera_sender --ros-args \
#   -p rgb_topic:=/rgb \
#   -p depth_topic:=/depth \
#   -p camera_info_topic:=/camera_info \
#   -p dest_ip:=127.0.0.1 \
#   -p dest_port:=9002 \
#   -p send_hz:=1.0 \
#   -p max_chunk_payload:=12000 \
#   -p resize_scale:=0.5




