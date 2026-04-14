import argparse
import json
from pathlib import Path
from typing import Dict, List


def _flatten_float_list(x) -> List[float]:
    if isinstance(x, list):
        out: List[float] = []
        for v in x:
            if isinstance(v, list):
                out.extend(_flatten_float_list(v))
            else:
                out.append(float(v))
        return out
    return [float(x)]


def _normalize_image_path(ep_name: str, image_path: str) -> str:
    if image_path and not Path(image_path).is_absolute():
        return str(Path(ep_name) / image_path).replace("\\", "/")
    return image_path


def _read_jsonl_rows(jsonl_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] JSON 파싱 실패, 건너뜀: {jsonl_path} line {line_num} | {exc}")
                continue
            row["_line_num"] = line_num
            rows.append(row)
    return rows


def _state_from_openvla_row(row: Dict) -> Dict:
    action = row.get("action", {})
    return {
        "arm_joint_position": _flatten_float_list(action.get("arm_joint_position", [])),
        "gripper_width": float(_flatten_float_list(action.get("gripper_width", 0.0))[0]),
        "ee_pose": _flatten_float_list(action.get("ee_pose", [])),
    }


def _phase_from_row(row: Dict) -> str:
    phase = row.get("phase")
    if phase is None:
        phase = row.get("meta", {}).get("phase", "")
    return str(phase or "")


def _note_from_row(row: Dict) -> str:
    note = row.get("note")
    if note is None:
        note = row.get("meta", {}).get("note", "")
    return str(note or "")


def _ee_z_from_state(state: Dict):
    ee_pose = state.get("ee_pose", [])
    if isinstance(ee_pose, list) and len(ee_pose) >= 3:
        return float(ee_pose[2])
    return None


def merge_openvla_delta_jsonl(
    root_dir: str,
    output_jsonl: str = "merged_samples_openvla_delta.jsonl",
    input_jsonl_name: str = "samples_openvla_delta.jsonl",
) -> None:
    root = Path(root_dir).resolve()
    output_path = root / output_jsonl
    episode_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")])

    if not episode_dirs:
        raise FileNotFoundError(f"episode_* 폴더를 찾지 못했습니다: {root}")

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for ep_dir in episode_dirs:
            ep_name = ep_dir.name
            jsonl_path = ep_dir / input_jsonl_name
            if not jsonl_path.exists():
                print(f"[WARN] 파일이 없어서 건너뜀: {jsonl_path}")
                continue

            print(f"[INFO] Cartesian delta 읽는 중: {jsonl_path}")
            for row in _read_jsonl_rows(jsonl_path):
                row["episode_key"] = ep_name
                if "frame_index" not in row:
                    row["frame_index"] = int(row["_line_num"] - 1)

                obs = row.get("observation", {})
                if isinstance(obs, dict):
                    obs["image"] = _normalize_image_path(ep_name, obs.get("image", ""))
                    row["observation"] = obs

                row.pop("_line_num", None)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"[DONE] Cartesian delta 병합 완료: {output_path}")
    print(f"[DONE] 총 저장된 샘플 수: {total_written}")


def merge_openvla_joint_delta_jsonl(
    root_dir: str,
    output_jsonl: str = "merged_samples_openvla_joint_delta.jsonl",
    input_jsonl_name: str = "samples_openvla.jsonl",
    gripper_label_mode: str = "first_close_only",
    gripper_epsilon: float = 1e-6,
) -> None:
    root = Path(root_dir).resolve()
    output_path = root / output_jsonl
    episode_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")])

    if not episode_dirs:
        raise FileNotFoundError(f"episode_* 폴더를 찾지 못했습니다: {root}")

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for ep_dir in episode_dirs:
            ep_name = ep_dir.name
            jsonl_path = ep_dir / input_jsonl_name
            if not jsonl_path.exists():
                print(f"[WARN] 파일이 없어서 건너뜀: {jsonl_path}")
                continue

            rows = _read_jsonl_rows(jsonl_path)
            if not rows:
                print(f"[WARN] 비어 있어서 건너뜀: {jsonl_path}")
                continue

            print(f"[INFO] Joint delta 읽는 중: {jsonl_path}")
            first_close_idx = None
            if gripper_label_mode == "first_close_only":
                for idx, row in enumerate(rows):
                    phase = _phase_from_row(row)
                    cur_state = _state_from_openvla_row(row)
                    next_state = _state_from_openvla_row(rows[idx + 1] if idx + 1 < len(rows) else row)
                    raw_gripper_delta = float(next_state["gripper_width"] - cur_state["gripper_width"])
                    if phase == "close" and raw_gripper_delta < -gripper_epsilon:
                        first_close_idx = idx
                        break

            for idx, row in enumerate(rows):
                next_row = rows[idx + 1] if idx + 1 < len(rows) else row
                cur_state = _state_from_openvla_row(row)
                next_state = _state_from_openvla_row(next_row)

                cur_joint = cur_state["arm_joint_position"]
                next_joint = next_state["arm_joint_position"]
                if len(cur_joint) != len(next_joint):
                    raise ValueError(f"arm_joint_position 길이 불일치: {jsonl_path} line {row['_line_num']}")

                joint_delta = [float(n - c) for c, n in zip(cur_joint, next_joint)]
                raw_gripper_delta = float(next_state["gripper_width"] - cur_state["gripper_width"])
                phase = _phase_from_row(row)
                next_phase = _phase_from_row(next_row)

                # The executor closes immediately once it decides to close, so repeated
                # negative labels across many "close" frames teach an early time-based close.
                if gripper_label_mode == "first_close_only":
                    if idx == first_close_idx and raw_gripper_delta < -gripper_epsilon:
                        gripper_delta = raw_gripper_delta
                    elif raw_gripper_delta < 0.0:
                        gripper_delta = 0.0
                    elif phase not in {"pre_grasp", "grasp", "close"}:
                        gripper_delta = 0.0
                    else:
                        gripper_delta = raw_gripper_delta if raw_gripper_delta > gripper_epsilon else 0.0
                elif gripper_label_mode == "phase_aware":
                    if phase != "close" and raw_gripper_delta < 0.0:
                        gripper_delta = 0.0
                    else:
                        gripper_delta = raw_gripper_delta if abs(raw_gripper_delta) > gripper_epsilon else 0.0
                else:
                    gripper_delta = raw_gripper_delta if abs(raw_gripper_delta) > gripper_epsilon else 0.0

                terminate = 1.0 if idx == len(rows) - 1 else 0.0

                image_path = _normalize_image_path(ep_name, row.get("image", ""))
                instruction = row.get("instruction") or "do the task"
                note = _note_from_row(row)
                timestamp = row.get("timestamp")
                rgb_timestamp = row.get("rgb_timestamp")
                ee_z = _ee_z_from_state(cur_state)

                merged_row = {
                    "image": image_path,
                    "instruction": instruction,
                    "phase": phase,
                    "note": note,
                    "observation": {
                        "image": image_path,
                        "natural_language_instruction": instruction,
                    },
                    "state": cur_state,
                    "action": {
                        "joint_delta": joint_delta,
                        "gripper_delta": [gripper_delta],
                        "terminate_episode": terminate,
                    },
                    "meta": {
                        "phase": phase,
                        "next_phase": next_phase,
                        "note": note,
                        "timestamp": timestamp,
                        "rgb_timestamp": rgb_timestamp,
                        "episode_key": ep_name,
                        "step_index": idx,
                        "ee_z": ee_z,
                        "gripper_label_mode": gripper_label_mode,
                    },
                    "episode_key": ep_name,
                    "frame_index": idx,
                }

                fout.write(json.dumps(merged_row, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"[DONE] Joint delta 병합 완료: {output_path}")
    print(f"[DONE] 총 저장된 샘플 수: {total_written}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="joint_delta",
        choices=["joint_delta", "cartesian_delta"],
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--input_jsonl_name", type=str, default=None)
    parser.add_argument(
        "--gripper_label_mode",
        type=str,
        default="first_close_only",
        choices=["first_close_only", "phase_aware", "raw"],
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "joint_delta":
        merge_openvla_joint_delta_jsonl(
            root_dir=args.root_dir,
            output_jsonl=args.output_jsonl or "merged_samples_openvla_joint_delta.jsonl",
            input_jsonl_name=args.input_jsonl_name or "samples_openvla.jsonl",
            gripper_label_mode=args.gripper_label_mode,
        )
    else:
        merge_openvla_delta_jsonl(
            root_dir=args.root_dir,
            output_jsonl=args.output_jsonl or "merged_samples_openvla_delta.jsonl",
            input_jsonl_name=args.input_jsonl_name or "samples_openvla_delta.jsonl",
        )


## 실행
# python3 src/robotarm_project/diffusion_vla/merge_openvla_delta_jsonl.py \
#   --mode joint_delta \
#   --root_dir src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_jsonl merged_samples_openvla_joint_delta.jsonl

# python3 src/robotarm_project/diffusion_vla/merge_openvla_delta_jsonl.py \
#   --mode joint_delta \
#   --root_dir src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_jsonl merged_samples_openvla_joint_delta_phase_aware.jsonl \
#   --input_jsonl_name samples_openvla.jsonl \
#   --gripper_label_mode first_close_only
