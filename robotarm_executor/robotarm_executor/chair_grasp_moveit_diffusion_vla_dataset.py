#!/usr/bin/env python3

import json
import random
import time

from robotarm_executor.chair_grasp_moveit_openvla_dataset import (
    OpenVLADatasetCollector,
    apply_gripper,
    copy_joint_state,
)


class DiffusionVLADatasetCollector(OpenVLADatasetCollector):
    """
    Diffusion/VLA 학습용 데이터셋 수집기.

    기존 OpenVLA 수집기와의 차이:
    - approach 구간을 far / mid / near / contact_ready 로 세분화
    - approach phase 에서 더 촘촘하게 프레임 저장
    - 목표 위치에 작은 jitter 를 줄 수 있게 해서 장면 다양성 확보
    """

    def __init__(self):
        super().__init__()

        extra_defaults = {
            "approach_mid_offset": 0.14,
            "approach_near_offset": 0.08,
            "contact_ready_offset": 0.04,
            "approach_pause_sec": 0.15,
            "approach_sample_period_sec": 0.05,
            "stage_pause_sec": 0.10,
            "goal_xy_jitter_m": 0.0,
            "goal_z_jitter_m": 0.0,
        }
        for name, value in extra_defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

        self.args.approach_mid_offset = float(self.get_parameter("approach_mid_offset").value)
        self.args.approach_near_offset = float(self.get_parameter("approach_near_offset").value)
        self.args.contact_ready_offset = float(self.get_parameter("contact_ready_offset").value)
        self.args.approach_pause_sec = max(0.0, float(self.get_parameter("approach_pause_sec").value))
        self.args.approach_sample_period_sec = max(
            0.01, float(self.get_parameter("approach_sample_period_sec").value)
        )
        self.args.stage_pause_sec = max(0.0, float(self.get_parameter("stage_pause_sec").value))
        self.args.goal_xy_jitter_m = max(0.0, float(self.get_parameter("goal_xy_jitter_m").value))
        self.args.goal_z_jitter_m = max(0.0, float(self.get_parameter("goal_z_jitter_m").value))

        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset_type": "diffusion_vla_stream",
                    "instruction_text": self.args.instruction_text,
                    "rgb_topic": self.args.rgb_topic,
                    "joint_state_topic": self.args.joint_state_topic,
                    "joint_command_topic": self.args.joint_command_topic,
                    "phase_layout": [
                        "pre_grasp",
                        "approach_far",
                        "approach_mid",
                        "approach_near",
                        "contact_ready",
                        "close",
                        "lift",
                    ],
                    "approach_offset": self.args.approach_offset,
                    "approach_mid_offset": self.args.approach_mid_offset,
                    "approach_near_offset": self.args.approach_near_offset,
                    "contact_ready_offset": self.args.contact_ready_offset,
                    "goal_xy_jitter_m": self.args.goal_xy_jitter_m,
                    "goal_z_jitter_m": self.args.goal_z_jitter_m,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        self.get_logger().info(
            f"Diffusion/VLA dataset collection ready. run_dir={self.run_dir}, rgb_topic={self.args.rgb_topic}"
        )

    def _phase_sample_period(self, phase: str) -> float:
        if phase in {"approach_far", "approach_mid", "approach_near", "contact_ready"}:
            return self.args.approach_sample_period_sec
        return self.args.frame_sample_period_sec

    def _maybe_record_frame(self, phase, joint_state, note="", force=False):
        if not self._recording_active and not force:
            return
        now = time.time()
        if not force and now < self._next_record_time:
            return
        self._save_openvla_sample(phase, joint_state, note)
        self._next_record_time = now + self._phase_sample_period(phase)

    def _jittered_goal_components(self, point_world):
        px = float(point_world[0])
        py = float(point_world[1])
        pz = float(point_world[2])
        if self.args.goal_xy_jitter_m > 0.0:
            px += random.uniform(-self.args.goal_xy_jitter_m, self.args.goal_xy_jitter_m)
            py += random.uniform(-self.args.goal_xy_jitter_m, self.args.goal_xy_jitter_m)
        if self.args.goal_z_jitter_m > 0.0:
            pz += random.uniform(-self.args.goal_z_jitter_m, self.args.goal_z_jitter_m)
        return px, py, pz

    def _capture_stage(self, phase, joint_state, note, pause_sec=None):
        pause_sec = self.args.stage_pause_sec if pause_sec is None else pause_sec
        if pause_sec > 0.0:
            self._pause_for_observation(pause_sec, hold_joint_state=joint_state)
        snapshot = self._settle_and_capture_joint_state(settle_sec=0.0)
        self._latest_js = copy_joint_state(snapshot)
        self._maybe_record_frame(phase, snapshot, note=note, force=True)
        return snapshot

    def _move_vertical_axis_locked(
        self,
        from_js,
        fixed_x,
        fixed_y,
        start_z,
        target_z,
        quat_xyzw,
        *,
        duration,
        rate_hz=40.0,
        phase=None,
        gripper_width=None,
    ):
        steps = max(1, int(duration * rate_hz))
        active_phase = self._phase if phase is None else phase
        current_js = copy_joint_state(from_js)

        for index in range(1, steps + 1):
            alpha = index / steps
            target_pos = (
                float(fixed_x),
                float(fixed_y),
                float((1.0 - alpha) * start_z + alpha * target_z),
            )
            solved_js = self.compute_ik(target_pos, quat_xyzw, seed_joint_state=current_js)
            if gripper_width is not None:
                apply_gripper(solved_js, gripper_width)
            self.pub.publish(solved_js)
            current_js = copy_joint_state(solved_js)
            self._latest_js = copy_joint_state(solved_js)
            self._maybe_record_frame(active_phase, self._latest_js, note="stream")
            time.sleep(1.0 / rate_hz)

        return current_js

    def try_execute(self):
        if self._executed or self._execution_in_progress or self._latest_js is None:
            return

        detection = self._locked_detection or self._latest_detection
        if detection is None:
            return

        if self.args.wait_for_rgb_before_start and self._latest_rgb is None:
            self.get_logger().warn(f"Waiting for RGB image on topic {self.args.rgb_topic} before starting")
            return

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

        self._execution_in_progress = True
        try:
            self._stop_continuous_recording()
            point_world = self.camera_point_to_world(detection)
            px, py, pz = self._jittered_goal_components(point_world)

            raw_goal_z = float(pz + self.args.grasp_offset)
            goal_z = max(raw_goal_z, self.args.min_goal_z)
            goal_pos = (px, py, goal_z)
            pre_pos = (px, py, goal_z + self.args.approach_offset)
            approach_mid_pos = (px, py, goal_z + self.args.approach_mid_offset)
            approach_near_pos = (px, py, goal_z + self.args.approach_near_offset)
            contact_ready_pos = (px, py, goal_z + self.args.contact_ready_offset)
            lift_pos = (px, py, goal_z + self.args.lift_offset)
            retreat_pos = (px, py, lift_pos[2] + self.args.retreat_offset)

            current_pose = self.ee_pose_from_fk()
            goal_quat = self._resolve_goal_orientation(current_pose.orientation)

            q_pre = self.compute_ik(pre_pos, goal_quat)
            q_mid = self.compute_ik(approach_mid_pos, goal_quat, seed_joint_state=q_pre)
            q_near = self.compute_ik(approach_near_pos, goal_quat, seed_joint_state=q_mid)
            q_ready = self.compute_ik(contact_ready_pos, goal_quat, seed_joint_state=q_near)
            q_goal_open = self.compute_ik(goal_pos, goal_quat, seed_joint_state=q_ready)

            for js in (q_pre, q_mid, q_near, q_ready, q_goal_open):
                apply_gripper(js, self.args.open_finger)

            current_js = copy_joint_state(self._latest_js)

            self._phase = "pre_grasp"
            self.move_smooth(current_js, q_pre, duration=self.args.duration, phase="pre_grasp_move")
            pre_js = self._capture_stage(
                "pre_grasp", q_pre, note="after_pre_grasp_pause", pause_sec=self.args.pre_grasp_pause_sec
            )

            self._start_continuous_recording()
            self._maybe_record_frame("pre_grasp", pre_js, note="pre_grasp_anchor", force=True)

            self._phase = "approach_far"
            q_mid = self._move_vertical_axis_locked(
                q_pre,
                px,
                py,
                pre_pos[2],
                approach_mid_pos[2],
                goal_quat,
                duration=self.args.duration * 0.75,
                phase="approach_far",
                gripper_width=self.args.open_finger,
            )
            mid_js = self._capture_stage("approach_far", q_mid, note="after_approach_far")

            self._phase = "approach_mid"
            q_near = self._move_vertical_axis_locked(
                mid_js,
                px,
                py,
                approach_mid_pos[2],
                approach_near_pos[2],
                goal_quat,
                duration=self.args.duration * 0.65,
                phase="approach_mid",
                gripper_width=self.args.open_finger,
            )
            near_js = self._capture_stage("approach_mid", q_near, note="after_approach_mid")

            self._phase = "approach_near"
            q_ready = self._move_vertical_axis_locked(
                near_js,
                px,
                py,
                approach_near_pos[2],
                contact_ready_pos[2],
                goal_quat,
                duration=self.args.duration * 0.55,
                phase="approach_near",
                gripper_width=self.args.open_finger,
            )
            ready_js = self._capture_stage("approach_near", q_ready, note="after_approach_near")

            self._phase = "contact_ready"
            q_goal_open = self._move_vertical_axis_locked(
                ready_js,
                px,
                py,
                contact_ready_pos[2],
                goal_pos[2],
                goal_quat,
                duration=self.args.duration * 0.45,
                phase="contact_ready",
                gripper_width=self.args.open_finger,
            )
            goal_open_js = self._capture_stage("contact_ready", q_goal_open, note="pre_close_ready")
            q_close = copy_joint_state(goal_open_js)
            apply_gripper(q_close, self.args.close_finger)

            self._phase = "close"
            self.move_smooth(goal_open_js, q_close, duration=self.args.gripper_motion_duration, phase="close")
            close_snapshot_js = self._settle_and_capture_joint_state()
            self._latest_js = copy_joint_state(close_snapshot_js)
            self._maybe_record_frame("close", close_snapshot_js, note="post_close", force=True)

            self._phase = "lift"
            q_lift = self.compute_ik(lift_pos, goal_quat)
            apply_gripper(q_lift, self.args.close_finger)
            self.move_smooth(q_close, q_lift, duration=self.args.duration, phase="lift")
            self._maybe_record_frame("lift", q_lift, note="post_lift", force=True)

            if self.args.enable_retreat:
                q_retreat = self.compute_ik(retreat_pos, goal_quat)
                apply_gripper(q_retreat, self.args.close_finger)
                self.move_smooth(q_lift, q_retreat, duration=self.args.duration, phase="retreat")
                self._maybe_record_frame("lift", q_retreat, note="post_retreat", force=True)

            self._executed = True
            self._stop_continuous_recording()
            self.get_logger().info(
                f"Diffusion/VLA dataset saved to {self.samples_path} with {self._sample_index} samples"
            )
        except Exception as exc:
            self._stop_continuous_recording()
            self.get_logger().error(f"Diffusion/VLA dataset collection failed at phase={self._phase}: {exc}")
        finally:
            self._execution_in_progress = False


def main(args=None):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=args)
    node = DiffusionVLADatasetCollector()
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


## 실행예시
# ros2 run robotarm_executor chair_grasp_moveit_diffusion_vla_dataset --ros-args \
#   -p dataset_root:=/home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   -p approach_offset:=0.20 \
#   -p approach_mid_offset:=0.14 \
#   -p approach_near_offset:=0.08 \
#   -p contact_ready_offset:=0.04 \
#   -p approach_sample_period_sec:=0.05 \
#   -p stage_pause_sec:=0.10 \
#   -p pre_grasp_pause_sec:=0.8 \
#   -p rgb_wait_timeout_sec:=1.0 \
#   -p max_rgb_staleness_sec:=0.0

## 이후 해야할 일
# 1. convert_jsonl_to_delta_joint.py 스크립트를 이용해서 raw joint state trajectory -> delta joint state trajectory 변환
# 2. merge_openvla_delta_jsonl.py 스크립트를 이용해서 여러 episode 의 delta jsonl 파일들을 하나로 합치기
