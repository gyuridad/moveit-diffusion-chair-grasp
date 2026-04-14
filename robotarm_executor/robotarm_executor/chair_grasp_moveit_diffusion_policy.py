#!/usr/bin/env python3

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from moveit_msgs.msg import Constraints, OrientationConstraint, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from robotarm_common.chair_grasp_common import camera_to_world


THIS_DIR = Path(__file__).resolve().parent


def resolve_robotarm_project_root() -> Path:
    env_root = os.environ.get("ROBOTARM_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "diffusion_vla" / "diffusion_vla_pretrained.py").exists():
            return candidate

    search_roots = [THIS_DIR, Path.cwd().resolve()]
    for root in search_roots:
        for candidate in [root, *root.parents]:
            direct = candidate / "diffusion_vla" / "diffusion_vla_pretrained.py"
            src_layout = (
                candidate
                / "src"
                / "robotarm_project"
                / "diffusion_vla"
                / "diffusion_vla_pretrained.py"
            )
            if direct.exists():
                return candidate
            if src_layout.exists():
                return candidate / "src" / "robotarm_project"

    raise RuntimeError(
        "Could not locate robotarm_project root. "
        "Set ROBOTARM_PROJECT_ROOT to the directory containing diffusion_vla."
    )


PROJECT_ROOT = resolve_robotarm_project_root()
DIFFUSION_DIR = PROJECT_ROOT / "diffusion_vla"
DEFAULT_DIFFUSION_CHECKPOINT = str(DIFFUSION_DIR / "diffusion_policy_output" / "policy_final.pt")
DEFAULT_DATASET_ROOT = str(PROJECT_ROOT / "debug_runs" / "chair_grasp_openvla")

if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

from diffusion_vla_pretrained import predict_single  # noqa: E402
    # diffusion_vla_pretrained.py 안의 추론 기능을 불러오는 import 문
    #         result = predict_single(
    #             checkpoint_path="policy_final.pt",
    #             image_path="sample.jpg",
    #             instruction="의자를 집어",
    #             state=[...],
    #             steps=50,
    #             device="cuda"
    #         )
    # 그러면 result 안에 이런 게 나옴:
    #         {
    #         "action_format": "joint_delta",
    #         "predicted_action_sequence": [...],
    #         "first_action": {
    #             "joint_delta": [...],
    #             "gripper_delta": [...],
    #             "terminate_episode": ...
    #         }
    #         }


BASE_DEFAULT_PARAMS = {
    "detection_topic": "/chair_detection_json",
    "joint_state_topic": "/joint_states",
    "joint_command_topic": "/joint_command",
    "rgb_topic": "/hand_Camera_rgb",
    "ik_service": "/compute_ik",
    "fk_service": "/compute_fk",
    "camera_frame": "",
    "tf_base_frame": "World",
    "moveit_base_frame": "world",
    "tip_link": "panda_hand",
    "tf_timeout_sec": 1.0,
    "approach_offset": 0.20,
    "grasp_offset": 0.03,
    "min_goal_z": 0.17,
    "lift_offset": 0.08,
    "retreat_offset": 0.05,
    "open_finger": 0.04,
    "close_finger": 0.0,
    "duration": 1.5,
        # duration은 move_smooth(..., duration=self.args.duration, ...)로 들어가고, 
        # 이 함수는 시작 자세에서 목표 자세까지 steps = duration * rate_hz 만큼 잘게 나눠서 보간 이동해.
        # 즉, 1.5초 동안 부드럽게 이동한다는 뜻
    "gripper_motion_duration": 0.8,
    "gripper_settle_sec": 0.5,
    "pre_grasp_pause_sec": 0.8,      # pre-grasp 단계에서 기다리는 시간
    "frame_sample_period_sec": 0.10,
    "fk_timeout_sec": 2.0,
    "preferred_camera_frame": "",    # 카메라 좌표계를 선택할 때 우선적으로 사용할 frame 이름
    "lock_detection_once": True,
    "ignore_stale_when_locked": True,
    "detection_stale_sec": 2.0,
    "orientation_mode": "top_down",
    "fixed_grasp_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "top_down_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "orientation_tolerance_rad": 0.12,
    "dataset_root": DEFAULT_DATASET_ROOT,
    "instruction_text": "의자를 집어",
    "image_format": "jpg",
    "jpeg_quality": 95,
    "wait_for_rgb_before_start": True,
    "rgb_wait_timeout_sec": 1.0,
    "max_rgb_staleness_sec": 6.0,
    "enable_retreat": False,
}

# Diffusion 정책 모델을 실제 실행할 때 쓰는 기본 설정값 묶음
DIFFUSION_EXTRA_DEFAULTS = {
    "diffusion_checkpoint_path": DEFAULT_DIFFUSION_CHECKPOINT,
    "diffusion_lora_path": "",                # 지금은 빈 문자열이니까 기본적으로 안 쓰겠다는 뜻
    "diffusion_device": "cuda",
    "diffusion_steps": 20,                    # 추론 지연을 줄이기 위해 기본 복원 step 수를 낮춘다.
    "diffusion_image_size": 224,              # 모델에 넣기 전에 224 x 224 로 resize
    "policy_max_steps": 20,                   # 한 번의 집기 시도에서 최대 행동 횟수
    "policy_timeout_sec": 120.0,              # 정책 실행 최대 시간을 늘려 접근 동작을 더 관찰한다.
    "policy_action_duration": 0.5,            # 각 action이 실제 움직임으로 더 충분히 반영되도록 적용 시간을 늘린다.
    "policy_action_rate_hz": 30.0,            # 제어 command 전송 주파수
    "max_joint_delta_rad": 0.08,              # 한 step에서 허용할 최대 관절 변화량 / 로봇이 갑자기 확 움직이지 않게 하는 안전 장치
    "gripper_delta_close_threshold": -0.01,   # 약한 음수 흔들림으로 바로 close되지 않게 더 보수적으로 둔다.
    "gripper_delta_open_threshold": 0.002,    # 열기/닫기 명령으로 해석할 임계값
    "gripper_close_min_ee_z": 0.30,           # EE z가 이 높이보다 충분히 낮아졌을 때만 close 허용
    "gripper_close_confirmation_steps": 2,    # close 신호가 연속으로 나온 횟수
    "use_terminate_signal": False,            # 모델 출력 안의 terminate_episode 값을 실제로 사용할지 여부
    "terminate_threshold": 0.5,
}

GROUP_NAME = "panda_arm"
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]


def quat_normalize(quat):
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = quat_normalize(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def copy_joint_state(msg: JointState) -> JointState:
    out = JointState()
    out.header = msg.header
    out.name = list(msg.name)
    out.position = list(msg.position)
    out.velocity = list(msg.velocity)
    out.effort = list(msg.effort)
    return out


def quat_from_param(value, default):
    if value is None or len(value) != 4:
        return quat_normalize(default)
    return quat_normalize([float(component) for component in value])


def apply_gripper(js: JointState, open_width: float):
    for name in FINGER_JOINT_NAMES:
        if name in js.name:
            js.position[js.name.index(name)] = open_width


def gripper_width_from_joint_state(js: JointState):
    if js is None:
        return None
    finger_positions = []
    for name in FINGER_JOINT_NAMES:
        if name in js.name:
            finger_positions.append(float(js.position[js.name.index(name)]))
    if not finger_positions:
        return None
    return float(sum(finger_positions))


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def ros_image_to_rgb(msg: Image):
    enc = (msg.encoding or "").lower()
    if enc == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if enc == "bgr8":
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if enc == "rgba8":
        rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    if enc == "bgra8":
        bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    if enc == "mono8":
        mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)
    raise RuntimeError(f"Unsupported RGB encoding: {msg.encoding}")


class OpenVLADatasetCollector(Node):
    def __init__(self, node_name="chair_grasp_moveit_openvla_dataset", param_overrides=None):
        super().__init__(node_name)

        defaults = dict(BASE_DEFAULT_PARAMS)
        if param_overrides:
            defaults.update(param_overrides)

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.args = SimpleNamespace(
            detection_topic=str(self.get_parameter("detection_topic").value),
            joint_state_topic=str(self.get_parameter("joint_state_topic").value),
            joint_command_topic=str(self.get_parameter("joint_command_topic").value),
            rgb_topic=str(self.get_parameter("rgb_topic").value),
            ik_service=str(self.get_parameter("ik_service").value),
            fk_service=str(self.get_parameter("fk_service").value),
            camera_frame=str(self.get_parameter("camera_frame").value),
            tf_base_frame=str(self.get_parameter("tf_base_frame").value),
            moveit_base_frame=str(self.get_parameter("moveit_base_frame").value),
            tip_link=str(self.get_parameter("tip_link").value),
            tf_timeout_sec=float(self.get_parameter("tf_timeout_sec").value),
            approach_offset=float(self.get_parameter("approach_offset").value),
            grasp_offset=float(self.get_parameter("grasp_offset").value),
            min_goal_z=float(self.get_parameter("min_goal_z").value),
            lift_offset=float(self.get_parameter("lift_offset").value),
            retreat_offset=float(self.get_parameter("retreat_offset").value),
            open_finger=float(self.get_parameter("open_finger").value),
            close_finger=float(self.get_parameter("close_finger").value),
            duration=float(self.get_parameter("duration").value),
            gripper_motion_duration=float(self.get_parameter("gripper_motion_duration").value),
            gripper_settle_sec=float(self.get_parameter("gripper_settle_sec").value),
            pre_grasp_pause_sec=max(0.0, float(self.get_parameter("pre_grasp_pause_sec").value)),
            frame_sample_period_sec=max(0.01, float(self.get_parameter("frame_sample_period_sec").value)),
            fk_timeout_sec=float(self.get_parameter("fk_timeout_sec").value),
            preferred_camera_frame=str(self.get_parameter("preferred_camera_frame").value),
            lock_detection_once=bool(self.get_parameter("lock_detection_once").value),
            ignore_stale_when_locked=bool(self.get_parameter("ignore_stale_when_locked").value),
            detection_stale_sec=float(self.get_parameter("detection_stale_sec").value),
            orientation_mode=str(self.get_parameter("orientation_mode").value),
            fixed_grasp_quat_xyzw=[float(v) for v in self.get_parameter("fixed_grasp_quat_xyzw").value],
            top_down_quat_xyzw=[float(v) for v in self.get_parameter("top_down_quat_xyzw").value],
            orientation_tolerance_rad=float(self.get_parameter("orientation_tolerance_rad").value),
            dataset_root=str(self.get_parameter("dataset_root").value),
            instruction_text=str(self.get_parameter("instruction_text").value),
            image_format=str(self.get_parameter("image_format").value).lower(),
            jpeg_quality=int(self.get_parameter("jpeg_quality").value),
            wait_for_rgb_before_start=bool(self.get_parameter("wait_for_rgb_before_start").value),
            rgb_wait_timeout_sec=max(0.0, float(self.get_parameter("rgb_wait_timeout_sec").value)),
            max_rgb_staleness_sec=max(0.0, float(self.get_parameter("max_rgb_staleness_sec").value)),
            enable_retreat=bool(self.get_parameter("enable_retreat").value),
        )

        self.service_cb_group = ReentrantCallbackGroup()
        self.io_cb_group = ReentrantCallbackGroup()
        self.ik_cli = self.create_client(GetPositionIK, self.args.ik_service, callback_group=self.service_cb_group)
        self.fk_cli = self.create_client(GetPositionFK, self.args.fk_service, callback_group=self.service_cb_group)
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"IK service not available: {self.args.ik_service}")
        if not self.fk_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"FK service not available: {self.args.fk_service}")

        # 이 변수들은 전부 노드가 실행되면서 현재 상태를 기억해두는 내부 저장값이야.
        #     쉽게 말하면:
        #             지금 최신 joint state가 뭔지
        #             최신 RGB가 뭔지
        #             detection이 들어왔는지
        #             지금 동작 중인지
        #             몇 번째 샘플까지 저장했는지
        #     이런 걸 기억하는 상태 변수들이야
        self._latest_js = None
        self._latest_rgb = None
        self._latest_rgb_time = None
        self._latest_rgb_receive_time = None
        self._last_saved_rgb_receive_time = None
        self._latest_detection = None
        self._locked_detection = None
        self._executed = False
        self._execution_in_progress = False
        self._phase = "idle"
        self._sample_index = 0
        self._recording_active = False
        self._next_record_time = 0.0

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.args.dataset_root, run_id)
        self.images_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.samples_path = os.path.join(self.run_dir, "samples_openvla.jsonl")
        self.meta_path = os.path.join(self.run_dir, "meta.json")

        # 메타정보 파일(meta.json)을 새로 만들어서 기록하는 부분
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_type": "openvla_stream",
                    "instruction_text": self.args.instruction_text,
                    "rgb_topic": self.args.rgb_topic,
                    "joint_state_topic": self.args.joint_state_topic,
                    "joint_command_topic": self.args.joint_command_topic,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        # 최근 10초 동안의 좌표계(transform) 기록을 보관하는 저장소
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
            # 왜 과거 기록이 필요하냐
            #     로봇에서는 센서와 TF 시간이 정확히 딱 맞지 않는 경우가 많아.
            #     예를 들어:
            #         카메라 이미지 timestamp = 12:00:01.200
            #         TF는 12:00:01.180, 12:00:01.250 이런 식으로 들어옴
            #     이럴 때 버퍼가 최근 TF 기록을 들고 있어야
            #     이미지 시각과 가장 가까운 transform을 찾아 쓸 수 있어.
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            JointState, self.args.joint_state_topic, self.on_joint_state, 10, callback_group=self.io_cb_group
        )
        self.create_subscription(
            String, self.args.detection_topic, self.on_detection, 10, callback_group=self.io_cb_group
        )
        self.create_subscription(Image, self.args.rgb_topic, self.on_rgb, 10, callback_group=self.io_cb_group)
        self.pub = self.create_publisher(JointState, self.args.joint_command_topic, 10)
        self.create_timer(0.1, self.try_execute, callback_group=self.io_cb_group)

    def on_joint_state(self, msg: JointState):
        self._latest_js = copy_joint_state(msg)

    def on_detection(self, msg: String):
        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Failed to parse detection JSON: {exc}")
            return
        self._latest_detection = result
        if self.args.lock_detection_once and self._locked_detection is None:
            self._locked_detection = result

    def on_rgb(self, msg: Image):
        try:
            self._latest_rgb = ros_image_to_rgb(msg)
            self._latest_rgb_time = stamp_to_sec(msg.header.stamp)
            self._latest_rgb_receive_time = time.time()
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode RGB image: {exc}")

    def _extract_detection_age_sec(self, detection):
        stamp = detection.get("stamp") if isinstance(detection, dict) else None
        if not isinstance(stamp, dict):
            return None
        sec = stamp.get("sec")
        nanosec = stamp.get("nanosec")
        if sec is None or nanosec is None:
            return None
        return time.time() - (float(sec) + float(nanosec) * 1e-9)

    def _resolve_goal_orientation(self, current_orientation):
        mode = (self.args.orientation_mode or "keep_current").strip().lower()
        if mode == "fixed":
            return quat_from_param(self.args.fixed_grasp_quat_xyzw, current_orientation)
        if mode == "top_down":
            return quat_from_param(self.args.top_down_quat_xyzw, current_orientation)
        return quat_normalize(current_orientation)

    def _joint_indices(self, js):
        indices = []
        for name in ARM_JOINT_NAMES:
            if name not in js.name:
                raise RuntimeError(f"Missing arm joint in JointState: {name}")
            indices.append(js.name.index(name))
        return indices
            # 반환값 한 줄 예시
            #     입력:
            #             ARM_JOINT_NAMES = ["joint1", "joint2", "joint3"]
            #             js.name = ["gripper", "joint2", "joint1", "joint3"]
            #     반환값:
            #             [2, 1, 3]

    def _arm_joint_vector(self, js):
        indices = self._joint_indices(js)
        return [float(js.position[index]) for index in indices]
            # 위 예시를 이어서 적용하면:
            #     js.position = [0.04, 0.1, 0.2, 0.3] 일때,
            # 반환값은:
            #     [0.2, 0.1, 0.3]

    def camera_point_to_world(self, detection):
        xyz_camera = detection["detection"].get("xyz_camera")
        if xyz_camera is None:
            raise RuntimeError("Detection does not contain 3D camera coordinates")

        point_camera = np.asarray(xyz_camera, dtype=np.float32)
        t_world_camera = detection.get("t_world_camera")
        if t_world_camera is not None:
            return camera_to_world(point_camera, np.asarray(t_world_camera, dtype=np.float32))

        camera_info = detection.get("camera_info") or {}
        raw_frame = camera_info.get("frame_id") or ""
        candidates = []
        for candidate in [raw_frame, raw_frame.lstrip("/"), self.args.preferred_camera_frame, self.args.camera_frame]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError("Detection does not contain T_world_camera or camera frame_id")

        deadline = time.time() + max(0.0, self.args.tf_timeout_sec)
        last_exc = None
        while time.time() < deadline:
            for camera_frame in candidates:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.args.tf_base_frame,
                        camera_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=self.args.tf_timeout_sec),
                    )
                    rotation = transform.transform.rotation
                    translation = transform.transform.translation
                    rotmat = quat_xyzw_to_rotmat((rotation.x, rotation.y, rotation.z, rotation.w))
                    trans = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
                    return (rotmat @ point_camera) + trans
                except TransformException as exc:
                    last_exc = exc
            rclpy.spin_once(self, timeout_sec=0.02)
        raise RuntimeError(f"TF lookup failed for camera frame candidates {candidates}: {last_exc}")

    def ee_pose_from_fk(self, base_frame=None, tip_link=None, timeout=None, joint_state=None):
        joint_state = self._latest_js if joint_state is None else joint_state
        if joint_state is None:
            raise RuntimeError("No joint state available for FK")

        req = GetPositionFK.Request()
        req.header.frame_id = self.args.moveit_base_frame if base_frame is None else base_frame
        req.fk_link_names = [self.args.tip_link if tip_link is None else tip_link]
            # fk_link_names 는 문자열 배열
            # Forward Kinematics는 원래
            # 하나의 조인트 상태로 여러 링크의 pose를 한 번에 계산할 수 있기 때문
        req.robot_state.joint_state = joint_state

        future = self.fk_cli.call_async(req)
        timeout = self.args.fk_timeout_sec if timeout is None else timeout
        end_time = time.time() + timeout
        while rclpy.ok() and not future.done():
            if time.time() >= end_time:
                raise RuntimeError("FK request timed out")
            rclpy.spin_once(self, timeout_sec=0.02)

        resp = future.result()
        if resp is None or len(resp.pose_stamped) == 0:
            raise RuntimeError("FK returned empty pose")

        pose = resp.pose_stamped[0].pose
        return SimpleNamespace(
            position=(pose.position.x, pose.position.y, pose.position.z),
            orientation=quat_normalize((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)),
        )

    def compute_ik(self, pos, quat_xyzw, timeout=2.0, avoid_collisions=False, ik_link_name=None):
        px, py, pz = [float(v) for v in np.asarray(pos).ravel()[:3]]
        qx, qy, qz, qw = quat_normalize([float(v) for v in np.asarray(quat_xyzw).ravel()[:4]])

        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout = Duration(seconds=float(timeout)).to_msg()
        req.ik_request.ik_link_name = self.args.tip_link if ik_link_name is None else ik_link_name
        req.ik_request.pose_stamped.header.frame_id = self.args.moveit_base_frame
        req.ik_request.pose_stamped.pose.position.x = px
        req.ik_request.pose_stamped.pose.position.y = py
        req.ik_request.pose_stamped.pose.position.z = pz
        req.ik_request.pose_stamped.pose.orientation.x = qx
        req.ik_request.pose_stamped.pose.orientation.y = qy
        req.ik_request.pose_stamped.pose.orientation.z = qz
        req.ik_request.pose_stamped.pose.orientation.w = qw

        constraint = OrientationConstraint()
        constraint.header.frame_id = self.args.moveit_base_frame
        constraint.link_name = req.ik_request.ik_link_name
        constraint.orientation = req.ik_request.pose_stamped.pose.orientation
        constraint.absolute_x_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.absolute_y_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.absolute_z_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.weight = 1.0
        constraints = Constraints()
        constraints.orientation_constraints = [constraint]
        req.ik_request.constraints = constraints

        seed = RobotState()
        seed.joint_state = self._latest_js
        req.ik_request.robot_state = seed

        future = self.ik_cli.call_async(req)
        end_time = time.time() + timeout
        while rclpy.ok() and not future.done():
            if time.time() >= end_time:
                raise RuntimeError("IK request timed out")
            rclpy.spin_once(self, timeout_sec=0.02)

        resp = future.result()
        if resp is None:
            raise RuntimeError("IK returned no response")
        if getattr(resp.error_code, "val", 0) != 1:
            raise RuntimeError(f"IK failed with error code {resp.error_code.val}")
            # 1. resp 구조
            #     resp = {
            #         "solution": ...,
            #         "error_code": ...
            #     }
            #     예시 느낌:
            #         resp.error_code.val = 1   # SUCCESS 같은 값
            #         resp.solution = <RobotState>
            # 2. resp.solution 구조
            #     resp.solution = {
            #         "joint_state": ...,
            #         "multi_dof_joint_state": ...,
            #         "attached_collision_objects": ...
            #     }
            # 3. resp.solution.joint_state 구조
            #     resp.solution.joint_state = {
            #         "header": ...,
            #         "name": ["panda_joint1", "panda_joint2", ...],
            #         "position": [0.12, -0.43, ...],
            #         "velocity": [...],
            #         "effort": [...]
            #     }

        solution = resp.solution.joint_state
        solution_map = dict(zip(solution.name, solution.position))
        isaac_order = list(self._latest_js.name)
        base_map = dict(zip(self._latest_js.name, self._latest_js.position))
        base_map.update(solution_map)

        out = JointState()
        out.name = isaac_order
        out.position = [float(base_map.get(name, 0.0)) for name in isaac_order]
        return out

    def _save_image(self):
        if self._latest_rgb is None:
            raise RuntimeError("No RGB image available")

        ext = "png" if self.args.image_format == "png" else "jpg"
        image_name = f"sample_{self._sample_index:06d}.{ext}"
        image_path = os.path.join(self.images_dir, image_name)
        bgr = cv2.cvtColor(self._latest_rgb, cv2.COLOR_RGB2BGR)
        if ext == "png":
            ok = cv2.imwrite(image_path, bgr)
        else:
            ok = cv2.imwrite(image_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to save image: {image_path}")
        self._last_saved_rgb_receive_time = self._latest_rgb_receive_time
        return image_path

    # 지금 저장하거나 사용할 RGB 이미지가 있는지,
    # 있다면 충분히 “새로운 프레임”인지 잠깐 기다려 확인하는 함수
    def _wait_for_fresh_rgb(self, require_new_frame=False):
        baseline = self._last_saved_rgb_receive_time if require_new_frame else None
        if self._latest_rgb is not None:
            if baseline is None:
                return True
            if self._latest_rgb_receive_time is not None and self._latest_rgb_receive_time > baseline:
                return True
        start = time.time()
        while time.time() - start < self.args.rgb_wait_timeout_sec:
            if self._latest_rgb is not None:
                if baseline is None:
                    return True
                if self._latest_rgb_receive_time is not None and self._latest_rgb_receive_time > baseline:
                    return True
            rclpy.spin_once(self, timeout_sec=0.02)
        if self._latest_rgb is None:
            return False
        if baseline is None:
            return True
        return self._latest_rgb_receive_time is not None and self._latest_rgb_receive_time > baseline

    def _rgb_is_stale(self, sample_time):
        rgb_reference_time = self._latest_rgb_receive_time
        if rgb_reference_time is None:
            return True
        if self.args.max_rgb_staleness_sec <= 0.0:
            return False
        return abs(sample_time - rgb_reference_time) > self.args.max_rgb_staleness_sec

    def _save_openvla_sample(self, phase, joint_state, note):
        """
        이 함수는 대충 이런 순서로 동작해:

            최신 RGB가 준비됐는지 확인
            RGB가 너무 오래됐는지 확인
            이미지 파일 저장
            현재 joint state로 FK 해서 ee pose 계산
            JSON 한 줄(sample) 만들기
            samples_path 파일에 한 줄 append
            샘플 인덱스 증가
        """
        now = time.time()
        if self.args.wait_for_rgb_before_start:
            self._wait_for_fresh_rgb(require_new_frame=self._last_saved_rgb_receive_time is not None)
        if self._latest_rgb is None:
            raise RuntimeError("No RGB image available for sample")
        if self._rgb_is_stale(now):
            raise RuntimeError("RGB image is stale for sample")

        image_path = self._save_image()
        ee_pose = self.ee_pose_from_fk(joint_state=joint_state)
        sample = {
            "image": os.path.relpath(image_path, self.run_dir),
            "instruction": self.args.instruction_text,
            "phase": phase,
            "note": note,
            "action": {
                "arm_joint_position": self._arm_joint_vector(joint_state),
                "gripper_width": gripper_width_from_joint_state(joint_state),
                "ee_pose": [
                    float(ee_pose.position[0]),
                    float(ee_pose.position[1]),
                    float(ee_pose.position[2]),
                    float(ee_pose.orientation[0]),
                    float(ee_pose.orientation[1]),
                    float(ee_pose.orientation[2]),
                    float(ee_pose.orientation[3]),
                ],
            },
            "timestamp": now,
            "rgb_timestamp": self._latest_rgb_time,
        }
        with open(self.samples_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._sample_index += 1

    def _start_continuous_recording(self):
        self._recording_active = True
        self._next_record_time = time.time()

    def _stop_continuous_recording(self):
        self._recording_active = False

    def _maybe_record_frame(self, phase, joint_state, note="", force=False):
        if not self._recording_active and not force:
            return
        now = time.time()
        if not force and now < self._next_record_time:
            return
        self._save_openvla_sample(phase, joint_state, note)
        self._next_record_time = now + self.args.frame_sample_period_sec

    def _settle_and_capture_joint_state(self, settle_sec=None):
        settle_sec = self.args.gripper_settle_sec if settle_sec is None else settle_sec
        deadline = time.time() + max(0.0, settle_sec)
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.time())))
        if self._latest_js is None:
            raise RuntimeError("No joint state available after settle")
        return copy_joint_state(self._latest_js)

    def _pause_for_observation(self, pause_sec, hold_joint_state=None, rate_hz=20.0):
        deadline = time.time() + max(0.0, pause_sec)
        while time.time() < deadline:
            if hold_joint_state is not None:
                hold_msg = JointState()
                hold_msg.name = list(hold_joint_state.name)
                hold_msg.position = list(hold_joint_state.position)
                self.pub.publish(hold_msg)
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.time())))
            if rate_hz > 0.0:
                time.sleep(1.0 / rate_hz)

    def move_smooth(self, from_js, to_js, duration=1.5, rate_hz=100, phase=None):
        names = list(from_js.name)
        if names != list(to_js.name):
            raise RuntimeError("Joint name order mismatch")
        start = np.asarray(from_js.position, dtype=float)
        goal = np.asarray(to_js.position, dtype=float)
        steps = max(1, int(duration * rate_hz))
        active_phase = self._phase if phase is None else phase
            # idle → approach → pre_grasp → grasp → lift

        for index in range(steps + 1):
            alpha = index / steps
            point = (1.0 - alpha) * start + alpha * goal
            msg = JointState()
            msg.name = names
            msg.position = point.tolist()
            self.pub.publish(msg)
            self._latest_js = copy_joint_state(msg)
            self._maybe_record_frame(active_phase, self._latest_js, note="stream")
            time.sleep(1.0 / rate_hz)

    def try_execute(self):
        return


class DiffusionPolicyExecutor(OpenVLADatasetCollector):
    def __init__(self):
        super().__init__(
            node_name="chair_grasp_moveit_diffusion_policy",
            param_overrides=DIFFUSION_EXTRA_DEFAULTS,
        )
        # 파라미터 선언이 없어도 되는 이유:
        #     부모 클래스에 아래와 같은 로직이 있기 때문에
        #         def __init__(self, node_name="chair_grasp_moveit_openvla_dataset", param_overrides=None):
        #             if param_overrides:
        #                 defaults.update(param_overrides)
        #             for name, value in defaults.items():
        #                 self.declare_parameter(name, value)

        self.args.diffusion_checkpoint_path = str(self.get_parameter("diffusion_checkpoint_path").value)
        self.args.diffusion_lora_path = str(self.get_parameter("diffusion_lora_path").value)
        self.args.diffusion_device = str(self.get_parameter("diffusion_device").value)
        self.args.diffusion_steps = int(self.get_parameter("diffusion_steps").value)
        self.args.diffusion_image_size = int(self.get_parameter("diffusion_image_size").value)
        self.args.policy_max_steps = int(self.get_parameter("policy_max_steps").value)
            # 조건 없으면 → 최대 20번 예측 -> 행동
            # 중간에 끝나면 → 그 전에 종료
            # 언제 20번보다 적게 끝나냐?
            #     1️⃣ terminate 신호 사용하는 경우
            #     2️⃣ timeout 걸리는 경우
            #     3️⃣ 목표 달성 조건
            #         예를 들어:
            #             물체 잡았다
            #             그리퍼 닫힘 완료

        self.args.policy_timeout_sec = float(self.get_parameter("policy_timeout_sec").value)

        # 아래 두 개의 파라미터는 move_smooth의 개념으로 부드럽게 예측된 액션 적용
        self.args.policy_action_duration = float(self.get_parameter("policy_action_duration").value)
        self.args.policy_action_rate_hz = float(self.get_parameter("policy_action_rate_hz").value)

        # 관절 1개가 한 번에 움직일 수 있는 최대 각도 변화량(Diffusion policy가 예측한 값이 실수로 클 수 있기 때문)
        self.args.max_joint_delta_rad = float(self.get_parameter("max_joint_delta_rad").value)
        self.args.gripper_delta_close_threshold = float(self.get_parameter("gripper_delta_close_threshold").value)
        self.args.gripper_delta_open_threshold = float(self.get_parameter("gripper_delta_open_threshold").value)
        self.args.gripper_close_min_ee_z = float(self.get_parameter("gripper_close_min_ee_z").value)
        self.args.gripper_close_confirmation_steps = max(
            1, int(self.get_parameter("gripper_close_confirmation_steps").value)
        )
        self.args.use_terminate_signal = bool(self.get_parameter("use_terminate_signal").value)
        self.args.terminate_threshold = float(self.get_parameter("terminate_threshold").value)

        # 정책(policy)이 그리퍼를 닫으라고 요청했는지 기록해 두는 내부 상태 플래그
        self._policy_requested_close = False
        self._close_signal_streak = 0
        self._policy_gripper_deltas = []
        self._policy_close_step = None
            # 처음에는 아직 “닫아라” 요청이 없다고 초기화

        self.policy_log_path = os.path.join(self.run_dir, "policy_log.jsonl")

        # “이번 diffusion policy 실행이 어떤 조건으로 돌았는지 기록해 두는 메타데이터 저장 코드”
        # 👉 실험 설명서
        # 👉 실행 로그의 헤더
        # 👉 재현성(reproducibility) 기록
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": "pre_grasp_then_diffusion_policy",
                    "instruction_text": self.args.instruction_text,
                    "rgb_topic": self.args.rgb_topic,
                    "joint_state_topic": self.args.joint_state_topic,
                    "joint_command_topic": self.args.joint_command_topic,
                    "diffusion_checkpoint_path": self.args.diffusion_checkpoint_path,
                    "diffusion_lora_path": self.args.diffusion_lora_path,
                    "diffusion_device": self.args.diffusion_device,
                    "diffusion_steps": self.args.diffusion_steps,
                    "diffusion_image_size": self.args.diffusion_image_size,
                    "gripper_delta_close_threshold": self.args.gripper_delta_close_threshold,
                    "gripper_close_min_ee_z": self.args.gripper_close_min_ee_z,
                    "gripper_close_confirmation_steps": self.args.gripper_close_confirmation_steps,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self.get_logger().info(
            f"Diffusion policy ready. run_dir={self.run_dir}, checkpoint={self.args.diffusion_checkpoint_path}"
        )

    def _append_policy_log(self, payload):
        with open(self.policy_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                # self.policy_log_path 파일을 엶
                # "a" 는 append 모드
                # 즉:
                # 👉 기존 내용 지우지 않고 뒤에 계속 붙여서 저장
                # payload:
                #     저장할 내용
                #     예:
                #         payload = {
                #             "step": 3,
                #             "joint_delta": [0.01, -0.02, 0.0],
                #             "gripper_delta": -0.01,
                #             "timestamp": 1710000000.12
                #         }

    # 현재 로봇의 joint state를 diffusion policy가 먹을 수 있는 1차원 state 벡터로 바꾸는 함수
    def _build_policy_state(self, joint_state):
        ee_pose = self.ee_pose_from_fk(joint_state=joint_state)
        gripper_width = gripper_width_from_joint_state(joint_state)
        if gripper_width is None:
            raise RuntimeError("No gripper width available from joint state")
        return self._arm_joint_vector(joint_state) + [
            float(gripper_width),
            float(ee_pose.position[0]),
            float(ee_pose.position[1]),
            float(ee_pose.position[2]),
            float(ee_pose.orientation[0]),
            float(ee_pose.orientation[1]),
            float(ee_pose.orientation[2]),
            float(ee_pose.orientation[3]),
        ]
            # [j1, j2, j3, j4, j5, j6, j7, gripper, x, y, z, qx, qy, qz, qw] / len(state) == 15

    # “지금 policy가 볼 이미지 한 장을 확보해서 저장하고, 그 이미지 정보까지 같이 넘겨주는 함수”
    # 반환값은 policy 추론 단계(image_path 필요)와 로그 기록 단계에서 활용
    def _capture_policy_frame(self):
        now = time.time()
        if self.args.wait_for_rgb_before_start:
            self._wait_for_fresh_rgb(require_new_frame=self._last_saved_rgb_receive_time is not None)
        if self._latest_rgb is None:
            raise RuntimeError("No RGB image available for policy inference")
        if self._rgb_is_stale(now):
            raise RuntimeError("RGB image is stale for policy inference")
        image_path = self._save_image()
        return {
            "image_path": image_path,
            "rgb_timestamp": self._latest_rgb_time,
            "receive_time": self._latest_rgb_receive_time,
            "capture_time": now,
            "sample_index": self._sample_index - 1,
        }

    def _run_diffusion_inference(self, frame_item, joint_state):
        checkpoint_path = self.args.diffusion_checkpoint_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Diffusion checkpoint not found: {checkpoint_path}")

        lora_path = (self.args.diffusion_lora_path or "").strip()
        if lora_path and not os.path.exists(lora_path):
            raise RuntimeError(f"Diffusion LoRA checkpoint not found: {lora_path}")

        image_size = self.args.diffusion_image_size
        image_size = None if image_size <= 0 else image_size

        state_vec = self._build_policy_state(joint_state)   # 15차원 벡터
        start = time.time()
        pred = predict_single(
            checkpoint_path=checkpoint_path,
            image_path=frame_item["image_path"],
            instruction=self.args.instruction_text,
            state=state_vec,
            image_size=image_size,
            steps=self.args.diffusion_steps,
            device=self.args.diffusion_device,
            lora_path=lora_path or None,
        )
            # pred 구조:
            #         {
            #         "action_format": "joint_delta",
            #         "predicted_action_sequence": [...],
            #         "first_action": {
            #             "joint_delta": [...],
            #             "gripper_delta": [...],
            #             "terminate_episode": ...
            #         }
            #         }

        latency_sec = time.time() - start
        first = pred["first_action"]
        action_format = pred.get("action_format", "")
        if action_format != "joint_delta":
            raise RuntimeError(
                f"Diffusion executor expects joint_delta model output, but got action_format={action_format}"
            )

        action = np.asarray(
            list(first["joint_delta"]) + list(first["gripper_delta"]) + [float(first["terminate_episode"])],
            dtype=np.float32,
        )
        return action, latency_sec, pred, state_vec

    # diffusion policy가 예측한 결과를 실제 로봇 명령으로 바꾸기 직전 단계
    def _apply_joint_delta(self, joint_state, action):
        """
        이 함수는 diffusion policy가 낸 “행동(action)”을
        로봇이 바로 이해할 수 있는 “목표 joint state”로 바꿔주는 역할이야.

        policy output
        → joint delta / gripper delta 해석
        → 현재 joint state에 반영
        → 새 목표 joint state 생성
        → 이후 smooth move나 publish에 사용

        전체 예시

            입력:

                joint_state.position = [0.10, -0.20, 0.30, -1.10, 0.50, 1.20, -0.70, 0.04, 0.04]
                action = [0.02, -0.01, 0.00, 0.03, 0.00, -0.02, 0.01, -0.005]

            가정:

                max_joint_delta_rad = 0.08
                gripper_delta_close_threshold = -0.002

            출력:

                out.position = [
                    0.12, -0.21, 0.30, -1.07, 0.50, 1.18, -0.69,
                    ... # gripper는 close_finger 값으로 변경
                ]

                joint_delta = [0.02, -0.01, 0.00, 0.03, 0.00, -0.02, 0.01]
                gripper_delta = -0.005
                close_requested = True
        """
        joint_delta = np.asarray(action[:7], dtype=np.float32)
        gripper_delta = float(action[7])

        if self.args.max_joint_delta_rad > 0.0:
            joint_delta = np.clip(joint_delta, -self.args.max_joint_delta_rad, self.args.max_joint_delta_rad)

        out = copy_joint_state(joint_state)
        joint_name_to_index = {name: idx for idx, name in enumerate(out.name)}
        for joint_index, joint_name in enumerate(ARM_JOINT_NAMES):
            if joint_name not in joint_name_to_index:
                raise RuntimeError(f"Missing arm joint in JointState: {joint_name}")
            state_index = joint_name_to_index[joint_name]
            out.position[state_index] = float(out.position[state_index] + joint_delta[joint_index])

        close_requested = False
        if gripper_delta <= self.args.gripper_delta_close_threshold:
            apply_gripper(out, self.args.close_finger)
            close_requested = True
        elif gripper_delta >= self.args.gripper_delta_open_threshold:
            apply_gripper(out, self.args.open_finger)

        return out, joint_delta, gripper_delta, close_requested

    def run_diffusion_policy_loop(self):
        """
        이 함수는 pre-grasp 자세에 도달한 뒤 실행되는 단계로서,
        diffusion policy를 반복 실행해서 “보고 → 예측하고 → 움직이고 → 기록하는” 제어 루프야.

        한 줄로 말하면:

            최신 이미지와 현재 관절상태를 바탕으로 action을 계속 예측하고,
            그 action을 실제 joint motion으로 적용하면서, 종료 조건이 올 때까지 반복하는 함수

        전체 역할

            이 함수가 하는 일은 크게 5단계야.
                1. 시작 시간과 step 초기화
                2. 반복문 안에서 최신 이미지 캡처
                3. diffusion policy로 다음 action 예측
                4. 예측한 action을 joint state에 적용하고 부드럽게 실행
                5. 로그 저장 후 종료 조건 검사
        """
        start_time = time.time()
        step_index = 0
        self._policy_requested_close = False
        self._close_signal_streak = 0
        self._policy_gripper_deltas = []
        self._policy_close_step = None
        self.get_logger().info("Diffusion policy loop started from pre-grasp pose")

        while rclpy.ok():     # ROS2 노드가 정상 동작 중인 동안 계속 반복
            if step_index >= self.args.policy_max_steps:
                self.get_logger().info(f"Policy loop stopped at max steps: {step_index}")
                break
            if (time.time() - start_time) >= self.args.policy_timeout_sec:
                self.get_logger().warn("Policy loop timed out")
                break

            frame_item = self._capture_policy_frame()
            current_js = copy_joint_state(self._latest_js)
            current_pose = self.ee_pose_from_fk(joint_state=current_js)

            self._phase = "policy_infer"
                # 예외가 나면 의미가 생김
                # 추론 중 예외가 터지면, 그 시점의 현재 phase 는 "policy_infer" 야.
            action, latency_sec, pred, state_vec = self._run_diffusion_inference(frame_item, current_js)
            q_next, clipped_joint_delta, gripper_delta, close_signal_model = self._apply_joint_delta(
                current_js, action
            )

            next_pose = self.ee_pose_from_fk(joint_state=q_next)
            close_allowed_by_height = min(
                float(current_pose.position[2]), float(next_pose.position[2])
            ) <= self.args.gripper_close_min_ee_z
            if close_signal_model:
                self._close_signal_streak += 1
            else:
                self._close_signal_streak = 0
            close_requested = (
                close_signal_model
                and close_allowed_by_height
                and self._close_signal_streak >= self.args.gripper_close_confirmation_steps
            )
            if close_signal_model and not close_requested:
                q_next = copy_joint_state(q_next)
                apply_gripper(q_next, self.args.open_finger)
                next_pose = self.ee_pose_from_fk(joint_state=q_next)

            self._phase = "policy_apply"
            self.move_smooth(
                current_js,
                q_next,
                duration=self.args.policy_action_duration,
                rate_hz=self.args.policy_action_rate_hz,
                phase="policy_apply",
            )
            self._latest_js = copy_joint_state(q_next)
            self._policy_requested_close = self._policy_requested_close or close_requested
                # _apply_joint_delta() 안에서 close_requested 가 결정됨 

            terminate_value = float(action[8])
            terminate_requested = self.args.use_terminate_signal and (
                terminate_value >= self.args.terminate_threshold
            )
            margin_to_threshold = float(gripper_delta - self.args.gripper_delta_close_threshold)
            self._policy_gripper_deltas.append(float(gripper_delta))
            if close_requested and self._policy_close_step is None:
                self._policy_close_step = step_index

            self._append_policy_log(
                {
                    "step_index": step_index,
                    "phase": self._phase,
                    "image_path": frame_item["image_path"],
                    "sample_index": frame_item["sample_index"],
                    "rgb_timestamp": frame_item["rgb_timestamp"],
                    "receive_time": frame_item["receive_time"],
                    "capture_time": frame_item["capture_time"],
                    "policy_state": state_vec,
                    "raw_action": action.tolist(),
                    "raw_prediction": pred,
                    "inference_latency_sec": latency_sec,
                    "applied_joint_delta": clipped_joint_delta.tolist(),
                    "gripper_delta": gripper_delta,
                    "gripper_action": float(action[7]),
                    "gripper_delta_close_threshold": float(self.args.gripper_delta_close_threshold),
                    "gripper_delta_margin_to_close_threshold": margin_to_threshold,
                    "close_signal_model": bool(close_signal_model),
                    "close_allowed_by_height": bool(close_allowed_by_height),
                    "gripper_close_min_ee_z": float(self.args.gripper_close_min_ee_z),
                    "close_signal_streak": self._close_signal_streak,
                    "gripper_close_confirmation_steps": self.args.gripper_close_confirmation_steps,
                    "close_requested": bool(close_requested),
                    "terminate_value": terminate_value,
                    "terminate_requested": bool(terminate_requested),
                    "current_ee_pose": list(current_pose.position) + list(current_pose.orientation),
                    "target_ee_pose": list(next_pose.position) + list(next_pose.orientation),
                    "joint_target": list(q_next.position),
                    "timestamp": time.time(),
                }
            )

            if close_requested:
                self.get_logger().info("Diffusion policy requested gripper close; ending policy loop")
                break
            if terminate_requested:
                self.get_logger().info(
                    f"Diffusion policy requested termination with value={terminate_value:.4f}"
                )
                break

            step_index += 1

    def try_execute(self):
        """
        탐지된 물체가 있으면, pre-grasp 위치로 먼저 이동하고,
        그 다음 diffusion policy로 마지막 접근/집기를 수행하고,
        닫기 요청이 있었으면 마지막으로 들어 올리는 함수

        [실행 가능 여부 확인]
            ↓
        [탐지 결과 가져오기]
            ↓
        [RGB / stale detection 확인]
            ↓
        [카메라 좌표 → world 좌표]
            ↓
        [goal / pre-grasp / lift 위치 계산]
            ↓
        [pre-grasp IK 계산]
            ↓
        [pre-grasp로 이동]
            ↓
        [잠깐 멈춰 관측]
            ↓
        [diffusion policy loop 실행]
            ├─ 이미지 캡처
            ├─ action 예측
            ├─ joint delta 적용
            ├─ move_smooth
            └─ close/terminate 시 종료
            ↓
        [close 요청 있었으면 lift]
            ↓
        [완료]

        policy는 close 시점 판단까지만 담당
        lift는 별도 IK/move_smooth로 수행

            이렇게 역할을 나누는 건 이상한 게 아니라, 오히려 실무적으로 자주 쓰는 방식이야.

        왜 문제 없냐?

            policy 데이터에 lift가 들어 있었더라도, 실행 시점에 그 뒤 동작을 안 쓰는 건 가능해.
            그건 학습된 행동 공간 일부를 실행 단계에서 잘라서 사용하는 것일 뿐이야.
        """
        if self._executed or self._execution_in_progress or self._latest_js is None:
            return
                # 이미 한 번 실행 끝났으면 실행 안 함
                # 지금 다른 실행이 진행 중이면 실행 안 함
                # 현재 joint state가 없으면 실행 안 함

        detection = self._locked_detection or self._latest_detection
        if detection is None:
            return

        if self.args.wait_for_rgb_before_start and self._latest_rgb is None:
            self.get_logger().warn(f"Waiting for RGB image on topic {self.args.rgb_topic} before starting")
            return
                # 뜻:
                #     설정상 RGB가 있어야 시작하는데
                #     아직 이미지가 없으면 시작 안 함

        detection_age_sec = self._extract_detection_age_sec(detection)
        if (
            (self._locked_detection is None or not self.args.ignore_stale_when_locked)
            and detection_age_sec is not None
            and detection_age_sec > self.args.detection_stale_sec
        ):
            self.get_logger().warn(
                f"Detection is stale: age={detection_age_sec:.3f}s, limit={self.args.detection_stale_sec:.3f}s"
            )
            return
                # 뜻:
                #     detection 시간이 너무 오래된 경우
                #     예전 탐지 결과로 로봇을 움직이지 않도록 중단

        self._execution_in_progress = True
        try:
            self._stop_continuous_recording()   # 이번 grasp 실행에 집중하기 위한 정리 단계
            point_world = self.camera_point_to_world(detection)

            raw_goal_z = float(point_world[2] + self.args.grasp_offset)
            goal_pos = (
                float(point_world[0]),
                float(point_world[1]),
                max(raw_goal_z, self.args.min_goal_z),  # 최소 z 값 0.17 보장 ! 
            )
            pre_pos = (goal_pos[0], goal_pos[1], goal_pos[2] + self.args.approach_offset)
            lift_pos = (goal_pos[0], goal_pos[1], goal_pos[2] + self.args.lift_offset)

            current_pose = self.ee_pose_from_fk()
            goal_quat = self._resolve_goal_orientation(current_pose.orientation)

            q_pre = self.compute_ik(pre_pos, goal_quat)
            apply_gripper(q_pre, self.args.open_finger)
            current_js = copy_joint_state(self._latest_js)

            self._phase = "pre_grasp"
            self.move_smooth(current_js, q_pre, duration=self.args.duration, phase="pre_grasp_move")
            self._latest_js = copy_joint_state(q_pre)
            self._pause_for_observation(self.args.pre_grasp_pause_sec, hold_joint_state=q_pre)

            self._phase = "policy"
            self.run_diffusion_policy_loop()

            if self._policy_requested_close:
                self._phase = "lift"
                q_lift = self.compute_ik(lift_pos, goal_quat)
                apply_gripper(q_lift, self.args.close_finger)
                    # 👉 디퓨전 루프에서도 그리퍼 닫기 행동은 한다.
                    #     이 lift 부분에서도 닫힌 상태를 다시 명시해서 유지하는 거야.
                    # 즉,
                    #     집는 순간(close) 은 디퓨전 루프 안에서 발생
                    #     그 다음 lift 단계에서는 닫은 상태를 유지한 채 들어 올리기
                    # 라고 보면 돼.

                self.move_smooth(
                    copy_joint_state(self._latest_js),
                    q_lift,
                    duration=self.args.duration,
                    phase="lift",
                )
                self._latest_js = copy_joint_state(q_lift)

            self._executed = True
            self.get_logger().info(f"Diffusion policy run finished. log_path={self.policy_log_path}")
        except Exception as exc:
            self.get_logger().error(f"Diffusion policy failed at phase={self._phase}: {exc}")
        finally:
            self._execution_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    node = DiffusionPolicyExecutor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


## 실행 예시
# ros2 run robotarm_executor chair_grasp_moveit_diffusion_policy
