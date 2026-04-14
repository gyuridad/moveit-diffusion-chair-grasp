import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {line_num}: {e}")
    return items


def save_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_nested(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_episode_base_key(sample: Dict[str, Any]) -> str:
    """
    episode_id가 있으면 우선 사용.
    없으면 instruction + image parent dir 기준의 base 그룹 생성.
    note는 에피소드 내부 phase 변화 때문에 제외한다.
    """
    if "episode_id" in sample:
        return str(sample["episode_id"])

    instruction = str(sample.get("instruction", ""))
    image_path = str(sample.get("image", ""))
    parent_dir = str(Path(image_path).parent)

    return f"stream||{instruction}||{parent_dir}"


def infer_episode_suffix(input_path: str) -> str:
    """
    입력 파일 상위 폴더명(예: episode_0001, 1)에서 ep001 형태 suffix를 만든다.
    숫자를 찾지 못하면 ep001을 기본값으로 사용한다.
    """
    parent_name = Path(input_path).resolve().parent.name
    digits = "".join(ch for ch in parent_name if ch.isdigit())
    if digits:
        return f"ep{int(digits):03d}"
    return "ep001"


def assign_episode_keys(samples: List[Dict[str, Any]], input_path: str) -> None:
    """
    명시적인 episode_id가 없는 경우, 입력 파일이 속한 episode 폴더명을 이용해
    같은 파일의 모든 sample에 동일한 episode_key를 부여한다.
    """
    episode_suffix = infer_episode_suffix(input_path)

    for sample in samples:
        if "episode_id" in sample:
            episode_key = str(sample["episode_id"])
            sample["_episode_base_key"] = episode_key
            sample["_episode_key"] = episode_key
            continue

        base_key = infer_episode_base_key(sample)
        sample["_episode_base_key"] = base_key
        sample["_episode_key"] = f"{base_key}||{episode_suffix}"


def get_sort_timestamp(sample: Dict[str, Any]) -> float:
    """
    정렬 우선순위:
    1) timestamp
    2) rgb_timestamp
    3) 파일명 숫자
    """
    if "timestamp" in sample:
        try:
            return float(sample["timestamp"])
        except Exception:
            pass

    if "rgb_timestamp" in sample:
        try:
            return float(sample["rgb_timestamp"])
        except Exception:
            pass

    image_path = str(sample.get("image", ""))
    stem = Path(image_path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return float(digits)
        except Exception:
            pass

    return 0.0


def validate_ee_pose(sample: Dict[str, Any]) -> Tuple[bool, Optional[List[float]]]:
    ee_pose = get_nested(sample, ["action", "ee_pose"], default=None)
    if ee_pose is None or not isinstance(ee_pose, list):
        return False, None

    try:
        ee_pose = [float(x) for x in ee_pose]
    except Exception:
        return False, None

    if len(ee_pose) != 7:
        return False, None

    return True, ee_pose


def validate_joint(sample: Dict[str, Any], expected_dim: int = 7) -> Tuple[bool, Optional[List[float]]]:
    joint = get_nested(sample, ["action", "arm_joint_position"], default=None)
    if joint is None or not isinstance(joint, list):
        return False, None

    try:
        joint = [float(x) for x in joint]
    except Exception:
        return False, None

    if len(joint) != expected_dim:
        return False, None

    return True, joint


# -----------------------------
# Quaternion utilities
# -----------------------------
def quat_normalize(q: List[float]) -> List[float]:
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / n, y / n, z / n, w / n]


def quat_conjugate(q: List[float]) -> List[float]:
    x, y, z, w = q
    return [-x, -y, -z, w]


def quat_multiply(q1: List[float], q2: List[float]) -> List[float]:
    """
    quaternion format: [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2

    return [x, y, z, w]


def quat_to_euler_xyz(q: List[float]) -> List[float]:
    """
    quaternion [x, y, z, w] -> [roll, pitch, yaw]
    """
    x, y, z, w = quat_normalize(q)

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


def rotation_delta_from_quat(q_cur: List[float], q_next: List[float]) -> List[float]:
    """
    q_delta = q_next * conjugate(q_cur)
    return [droll, dpitch, dyaw]
    """
    q_cur = quat_normalize(q_cur)
    q_next = quat_normalize(q_next)

    q_delta = quat_multiply(q_next, quat_conjugate(q_cur))
    q_delta = quat_normalize(q_delta)

    # quaternion sign ambiguity 보정
    if q_delta[3] < 0:
        q_delta = [-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3]]

    return quat_to_euler_xyz(q_delta)


def build_openvla_like_sample(
    current_sample: Dict[str, Any],
    next_sample: Dict[str, Any],
    current_joint: Optional[List[float]],
    current_ee_pose: List[float],
    next_ee_pose: List[float],
    terminate_episode: float,
) -> Dict[str, Any]:
    # position delta
    cur_pos = current_ee_pose[:3]
    nxt_pos = next_ee_pose[:3]
    world_vector = [n - c for c, n in zip(cur_pos, nxt_pos)]

    # rotation delta
    cur_quat = current_ee_pose[3:]   # [qx, qy, qz, qw]
    nxt_quat = next_ee_pose[3:]
    rotation_delta = rotation_delta_from_quat(cur_quat, nxt_quat)

    # gripper action
    current_gripper = get_nested(current_sample, ["action", "gripper_width"], default=None)
    next_gripper = get_nested(next_sample, ["action", "gripper_width"], default=None)

    gripper_closedness_action = [0.0]
    if current_gripper is not None and next_gripper is not None:
        try:
            current_gripper = float(current_gripper)
            next_gripper = float(next_gripper)

            # width가 줄어들면 "닫힘"으로 해석 -> positive
            gripper_closedness_action = [current_gripper - next_gripper]
        except Exception:
            gripper_closedness_action = [0.0]

    output = {
        # OpenVLA-style에 가까운 입력 표현
        "image": current_sample.get("image"),
        "instruction": current_sample.get("instruction"),

        "observation": {
            "image": current_sample.get("image"),
            "natural_language_instruction": current_sample.get("instruction"),
        },

        "action": {
            "world_vector": world_vector,                       # [dx, dy, dz]
            "rotation_delta": rotation_delta,                  # [droll, dpitch, dyaw]
            "gripper_closedness_action": gripper_closedness_action,  # [dg]
            "terminate_episode": float(terminate_episode),
        },

        # optional: 디버깅/추적용 현재 상태
        "state": {
            "arm_joint_position": current_joint,
            "gripper_width": current_gripper,
            "ee_pose": current_ee_pose,
        },

        "meta": {
            "phase": current_sample.get("phase"),
            "note": current_sample.get("note"),
            "timestamp": current_sample.get("timestamp"),
            "rgb_timestamp": current_sample.get("rgb_timestamp"),
            "next_image": next_sample.get("image"),
            "next_timestamp": next_sample.get("timestamp"),
            "next_rgb_timestamp": next_sample.get("rgb_timestamp"),
        }
    }

    return output


def convert_jsonl_to_openvla_style(
    input_path: str,
    output_path: str,
    expected_joint_dim: int = 7,
    min_group_size: int = 2,
) -> None:
    samples = load_jsonl(input_path)
    assign_episode_keys(samples, input_path)
    print(f"[INFO] Loaded {len(samples)} samples")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    valid_count = 0
    invalid_count = 0

    for sample in samples:
        ok_ee, ee_pose = validate_ee_pose(sample)
        if not ok_ee:
            invalid_count += 1
            continue

        ok_joint, joint = validate_joint(sample, expected_dim=expected_joint_dim)
        if not ok_joint:
            joint = None  # joint가 없더라도 OpenVLA-like 변환은 가능

        sample["_ee_pose_cache"] = ee_pose
        sample["_joint_cache"] = joint

        key = str(sample.get("_episode_key", infer_episode_base_key(sample)))
        groups.setdefault(key, []).append(sample)
        valid_count += 1

    print(f"[INFO] Valid samples: {valid_count}")
    print(f"[INFO] Invalid samples skipped: {invalid_count}")
    print(f"[INFO] Number of groups: {len(groups)}")

    converted_items: List[Dict[str, Any]] = []
    skipped_small_groups = 0

    for group_key, group_samples in groups.items():
        if len(group_samples) < min_group_size:
            skipped_small_groups += 1
            continue

        group_samples.sort(key=get_sort_timestamp)

        for i in range(len(group_samples) - 1):
            cur = group_samples[i]
            nxt = group_samples[i + 1]

            cur_joint = cur.get("_joint_cache", None)
            cur_ee_pose = cur["_ee_pose_cache"]
            nxt_ee_pose = nxt["_ee_pose_cache"]

            terminate_episode = 1.0 if i == len(group_samples) - 2 else 0.0

            item = build_openvla_like_sample(
                current_sample=cur,
                next_sample=nxt,
                current_joint=cur_joint,
                current_ee_pose=cur_ee_pose,
                next_ee_pose=nxt_ee_pose,
                terminate_episode=terminate_episode,
            )
            item["meta"]["episode_key"] = group_key
            item["meta"]["episode_base_key"] = cur.get("_episode_base_key")
            item["meta"]["step_index"] = i
            converted_items.append(item)

    print(f"[INFO] Small groups skipped: {skipped_small_groups}")
    print(f"[INFO] Converted samples created: {len(converted_items)}")

    save_jsonl(output_path, converted_items)
    print(f"[INFO] Saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input jsonl path")
    parser.add_argument("--output", type=str, required=True, help="output jsonl path")
    parser.add_argument("--expected_joint_dim", type=int, default=7)
    parser.add_argument("--min_group_size", type=int, default=2)

    args = parser.parse_args()

    convert_jsonl_to_openvla_style(
        input_path=args.input,
        output_path=args.output,
        expected_joint_dim=args.expected_joint_dim,
        min_group_size=args.min_group_size,
    )

## 실행 예시
# python /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/openVLA_dataset/convert_jsonl_to_delta_joint.py \
#   --input /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/episode_0001/samples_openvla.jsonl \
#   --output /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/episode_0001/samples_openvla_delta.jsonl


# python3 /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/openVLA_dataset/convert_jsonl_to_delta_joint.py \
#   --input /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/episode_0001/samples_openvla.jsonl \
#   --output /home/lst7910/isaac_ros2_ws/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/episode_0001/samples_openvla_delta.jsonl