import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np

# diffusion_vla_pretrained.py 안의 predict_single 함수 import
from diffusion_vla_pretrained import predict_single


def flatten_float_list(x):
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, list):
                out.extend(flatten_float_list(v))
            else:
                out.append(float(v))
        return out
    return [float(x)]


def build_state_vector(row: Dict[str, Any]) -> List[float]:
    state = row.get("state", {})
    obs = row.get("observation", {})

    state_vec = []
    if "arm_joint_position" in state:
        state_vec.extend(flatten_float_list(state["arm_joint_position"]))
    if "gripper_width" in state:
        state_vec.extend(flatten_float_list(state["gripper_width"]))
    if "ee_pose" in state:
        state_vec.extend(flatten_float_list(state["ee_pose"]))

    if not state_vec and isinstance(obs, dict):
        for k in ["state", "EEF_state", "gripper_state"]:
            if k in obs:
                state_vec.extend(flatten_float_list(obs[k]))

    return state_vec


def infer_action_format(action: Dict[str, Any]) -> str:
    if any(k in action for k in ["world_vector", "rotation_delta"]):
        return "cartesian_delta"
    if any(k in action for k in ["joint_delta", "arm_joint_delta"]):
        return "joint_delta"
    raise ValueError(f"Could not infer action format from keys: {sorted(action.keys())}")


def build_action_vector(row: Dict[str, Any], action_format: str = "auto") -> Tuple[List[float], str]:
    action = row.get("action", {})
    resolved_format = infer_action_format(action) if action_format == "auto" else action_format
    action_vec = []

    if resolved_format == "cartesian_delta":
        if "world_vector" in action:
            action_vec.extend(flatten_float_list(action["world_vector"]))
        if "rotation_delta" in action:
            action_vec.extend(flatten_float_list(action["rotation_delta"]))
        if "gripper_closedness_action" in action:
            action_vec.extend(flatten_float_list(action["gripper_closedness_action"]))
    elif resolved_format == "joint_delta":
        if "joint_delta" in action:
            action_vec.extend(flatten_float_list(action["joint_delta"]))
        elif "arm_joint_delta" in action:
            action_vec.extend(flatten_float_list(action["arm_joint_delta"]))
        else:
            raise ValueError("joint_delta action format requires 'joint_delta' or 'arm_joint_delta'")

        if "gripper_delta" in action:
            action_vec.extend(flatten_float_list(action["gripper_delta"]))
        elif "gripper_width_delta" in action:
            action_vec.extend(flatten_float_list(action["gripper_width_delta"]))
        elif "gripper_action" in action:
            action_vec.extend(flatten_float_list(action["gripper_action"]))
        elif "gripper_closedness_action" in action:
            action_vec.extend(flatten_float_list(action["gripper_closedness_action"]))
        else:
            raise ValueError("joint_delta action format requires gripper delta/action field")
    else:
        raise ValueError(f"Unsupported action format: {resolved_format}")

    terminate = action.get("terminate_episode", 0.0)
    action_vec.extend(flatten_float_list(terminate))
    return action_vec, resolved_format


def first_action_to_vector(first_action: Dict[str, Any], action_format: str = "auto") -> List[float]:
    resolved_format = infer_action_format(first_action) if action_format == "auto" else action_format
    vec = []

    if resolved_format == "cartesian_delta":
        vec.extend(flatten_float_list(first_action["world_vector"]))
        vec.extend(flatten_float_list(first_action["rotation_delta"]))
        vec.extend(flatten_float_list(first_action["gripper_closedness_action"]))
    elif resolved_format == "joint_delta":
        vec.extend(flatten_float_list(first_action["joint_delta"]))
        vec.extend(flatten_float_list(first_action["gripper_delta"]))
    else:
        raise ValueError(f"Unsupported action format: {resolved_format}")

    vec.extend(flatten_float_list(first_action["terminate_episode"]))
    return vec


def load_rows(jsonl_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_idx"] = line_idx
            rows.append(row)
    return rows


def pick_rows(rows: List[Dict[str, Any]], num_samples: int, mode: str, seed: int) -> List[Dict[str, Any]]:
    if num_samples <= 0 or num_samples >= len(rows):
        return rows

    if mode == "first":
        return rows[:num_samples]

    if mode == "random":
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        indices = indices[:num_samples]
        indices.sort()
        return [rows[i] for i in indices]

    raise ValueError(f"Unsupported mode: {mode}")


def l1(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def l2(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def split_action(vec: List[float], action_format: str) -> Dict[str, List[float]]:
    if action_format == "joint_delta":
        return {
            "joint_delta": vec[:7],
            "gripper_delta": vec[7:8],
            "terminate": vec[8:9],
        }

    return {
        "world_vector": vec[:3],
        "rotation_delta": vec[3:6],
        "gripper": vec[6:7],
        "terminate": vec[7:8],
    }


def evaluate(
    checkpoint_path: str,
    jsonl_path: str,
    image_root: str,
    output_path: str,
    num_samples: int,
    sample_mode: str,
    seed: int,
    steps: int,
    device: str,
    image_size: int = None,
    lora_path: str = None,
    action_format: str = "auto",
):
    rows = load_rows(jsonl_path)
    selected_rows = pick_rows(rows, num_samples=num_samples, mode=sample_mode, seed=seed)

    all_results = []
    all_gt = []
    all_pred = []

    for i, row in enumerate(selected_rows):
        obs = row.get("observation", {})
        image_rel = obs.get("image", "")
        image_path = image_rel if os.path.isabs(image_rel) else os.path.join(image_root, image_rel)

        instruction = obs.get("natural_language_instruction") or obs.get("instruction") or "do the task"
        state_vec = build_state_vector(row)
        gt_action, row_action_format = build_action_vector(row, action_format=action_format)
        expected_action_dim = 9 if row_action_format == "joint_delta" else 8

        if len(state_vec) == 0:
            print(f"[WARN] skip line {row['_line_idx']} - empty state")
            continue
        if len(gt_action) != expected_action_dim:
            print(
                f"[WARN] skip line {row['_line_idx']} - gt action dim {len(gt_action)} "
                f"!= {expected_action_dim} ({row_action_format})"
            )
            continue
        if not os.path.exists(image_path):
            print(f"[WARN] skip line {row['_line_idx']} - image not found: {image_path}")
            continue

        pred = predict_single(
            checkpoint_path=checkpoint_path,
            image_path=image_path,
            instruction=instruction,
            state=state_vec,
            image_size=image_size,
            steps=steps,
            device=device,
            lora_path=lora_path,
        )
        pred_action = first_action_to_vector(pred["first_action"], action_format=pred.get("action_format", "auto"))

        if len(pred_action) != len(gt_action):
            print(
                f"[WARN] skip line {row['_line_idx']} - pred action dim {len(pred_action)} "
                f"!= gt dim {len(gt_action)}"
            )
            continue

        result = {
            "sample_index": i,
            "jsonl_line_index": row["_line_idx"],
            "image_path": image_path,
            "instruction": instruction,
            "action_format": row_action_format,
            "state": state_vec,
            "gt_action": {
                "full": gt_action,
                **split_action(gt_action, row_action_format),
            },
            "pred_action": {
                "full": pred_action,
                **split_action(pred_action, row_action_format),
            },
            "error": {
                "l1_full": l1(gt_action, pred_action),
                "rmse_full": l2(gt_action, pred_action),
            },
        }

        if row_action_format == "joint_delta":
            result["error"].update({
                "l1_joint_delta": l1(gt_action[:7], pred_action[:7]),
                "l1_gripper_delta": l1(gt_action[7:8], pred_action[7:8]),
                "l1_terminate": l1(gt_action[8:9], pred_action[8:9]),
            })
        else:
            result["error"].update({
                "l1_world_vector": l1(gt_action[:3], pred_action[:3]),
                "l1_rotation_delta": l1(gt_action[3:6], pred_action[3:6]),
                "l1_gripper": l1(gt_action[6:7], pred_action[6:7]),
                "l1_terminate": l1(gt_action[7:8], pred_action[7:8]),
            })

        all_results.append(result)
        all_gt.append(gt_action)
        all_pred.append(pred_action)

        print(
            f"[{i+1}/{len(selected_rows)}] "
            f"line={row['_line_idx']} "
            f"l1={result['error']['l1_full']:.6f} "
            f"rmse={result['error']['rmse_full']:.6f}"
        )

    summary = {}
    if len(all_results) > 0:
        gt_arr = np.array(all_gt, dtype=np.float32)
        pred_arr = np.array(all_pred, dtype=np.float32)
        abs_err = np.abs(gt_arr - pred_arr)

        summary = {
            "num_evaluated": len(all_results),
            "action_format": all_results[0]["action_format"],
            "mean_l1_full": float(abs_err.mean()),
            "rmse_full": float(np.sqrt(np.mean((gt_arr - pred_arr) ** 2))),
        }
        if summary["action_format"] == "joint_delta":
            summary.update({
                "mean_l1_joint_delta": float(abs_err[:, :7].mean()),
                "mean_l1_gripper_delta": float(abs_err[:, 7:8].mean()),
                "mean_l1_terminate": float(abs_err[:, 8:9].mean()),
            })
        else:
            summary.update({
                "mean_l1_world_vector": float(abs_err[:, :3].mean()),
                "mean_l1_rotation_delta": float(abs_err[:, 3:6].mean()),
                "mean_l1_gripper": float(abs_err[:, 6:7].mean()),
                "mean_l1_terminate": float(abs_err[:, 7:8].mean()),
            })

    out = {
        "summary": summary,
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_mode", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--action_format", type=str, default="auto", choices=["auto", "cartesian_delta", "joint_delta"])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        output_path=args.output_path,
        num_samples=args.num_samples,
        sample_mode=args.sample_mode,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        image_size=args.image_size,
        lora_path=args.lora_path,
        action_format=args.action_format,
    )

# 도커 안에서 실행 예시

# /home/lst7910/isaac_ros2_ws 를 /workspace 로 mount한 기존 컨테이너 재사용:
# docker start openvla_diffusion
# docker exec -it openvla_diffusion /bin/bash

# cd /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/diffusion_vla

# python eval_diffusion_predictions.py \
#   --checkpoint /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/diffusion_vla/diffusion_policy_output/policy_final.pt \
#   --jsonl_path /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla/merged_samples_openvla_joint_delta.jsonl \
#   --image_root /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/debug_runs/chair_grasp_openvla \
#   --output_path /workspace/IsaacSim-ros_workspaces/humble_ws/src/robotarm_project/diffusion_vla/eval_results_10.json \
#   --num_samples 10 \
#   --sample_mode first \
#   --steps 50 \
#   --device cuda \
#   --action_format joint_delta

