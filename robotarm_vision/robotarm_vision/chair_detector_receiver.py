import argparse
import json
import socket
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

from robotarm_common.chair_grasp_common import (
    UdpChunkAssembler,
    bbox_center_xyxy,
    extract_crop_pca_quaternion,
    parse_frame_payload,
    pixel_to_camera_xyz,
    robust_depth_at,
)


def resolve_classes_filter(classes_filter, model):
    """
    사용자가 문자열로 적은 클래스 이름을 YOLO 모델의 클래스 ID 리스트로 바꿔주는 역할

    사용자가 "person, car" 같은 문자열 입력
    모델이 가지고 있는 클래스 목록과 비교
    해당하는 숫자 ID로 변환
    최종적으로 [0, 2] 같은 형태 반환
    """
    if classes_filter is None:
        return None
    # 모델의 클래스 이름 정보를 항상 딕셔너리 형태로 맞추는 작업
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    # 클래스 이름 → 클래스 번호로 뒤집는 작업
    name_to_id = {name.lower(): idx for idx, name in names.items()}
        # 예를 들어:
        #     {0: 'person', 1: 'bicycle', 2: 'car'}를 
        #     {'person': 0, 'bicycle': 1, 'car': 2}로 바꿔줌

    tokens = [token.strip().lower() for token in classes_filter.split(",")]
        # classes_filter = "person, car, dog" 같은 문자열이 들어오면
        # tokens = ['person', 'car', 'dog']로 리스트로 만들어줌

    ids = [name_to_id[token] for token in tokens if token in name_to_id]
    return ids or None 

def save_depth_preview(depth: np.ndarray, path: Path) -> None:
    """
    이 함수는 depth 배열을 사람이 보기 쉬운 컬러 이미지로 저장하는 함수야.

    즉 원래 depth는 숫자 배열이라서 그냥 보면 잘 안 보이는데,
    이 함수를 쓰면 가까운 곳/먼 곳이 색으로 보이는 미리보기 이미지로 바꿔서 파일로 저장해줘

    이 함수는 이렇게 동작해:

    depth 값을 float 형태로 복사
    유효한 depth 값만 골라냄
    너무 작은 값/너무 큰 값의 영향을 줄이기 위해 1%~99% 구간만 사용
    그 범위를 0~255로 정규화
    컬러맵(JET) 적용
    png/jpg 파일로 저장
    """
    depth_vis = depth.astype(np.float32).copy()
    valid = np.isfinite(depth_vis) & (depth_vis > 0)  # 정상적인 양수 depth만 True
    if np.any(valid):    # 유효한 값이 하나라도 있으면 아래 작업 수행
        dmin, dmax = np.percentile(depth_vis[valid], [1, 99])
        if dmax <= dmin:
            dmax = dmin + 1e-3

        norm = np.zeros_like(depth_vis, dtype=np.float32)
        norm[valid] = (np.clip(depth_vis[valid], dmin, dmax) - dmin) / (dmax - dmin)  # 값 범위가 0~1로 바뀜
        depth_u8 = (norm * 255.0).astype(np.uint8)   # 이제 0~1 범위를 0~255 범위로 바꿈
    else:     # 유효한 값이 하나도 없으면 그냥 검은색 이미지로 만들어줌
        depth_u8 = np.zeros_like(depth_vis, dtype=np.uint8)

    cv2.imwrite(str(path), cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET))

def clamp_xyxy(box, width: int, height: int):
    """
    바운딩박스 좌표를 이미지 범위 안으로 안전하게 잘라주는 함수
    즉, 객체 탐지 결과 박스가 이미지 밖으로 조금 튀어나오거나 소수점 좌표로 들어와도,
    실제로 crop 하거나 인덱싱할 수 있는 안전한 정수 좌표로 바꿔주는 역할
    """
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x2))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y2))))
    return x1, y1, x2, y2

def frame_age_sec(camera_info: dict, key: str, now_sec: float) -> Optional[float]:
    """
    카메라 프레임이 지금 시점 기준으로 얼마나 오래됐는지(몇 초 지났는지) 계산하는 함수

    ROS/sim stamp가 Unix epoch 기준이 아닐 수 있어서
    epoch처럼 보이는 값일 때만 age를 계산한다.
    """
    stamp = camera_info.get(key)
    if stamp is None:
        return None
    try:
        stamp = float(stamp)
    except (TypeError, ValueError):
        return None
    if stamp < 1_000_000_000.0:
        return None
    return max(0.0, now_sec - stamp)


def packet_transport_age_sec(packet: dict, now_sec: float) -> Optional[float]:
    """
    패킷 안에 들어 있는 "stamp"가
    **“이 데이터가 송신측에서 만들어진 시간”**이라고 보면 돼.

    그리고 now_sec는
    **“지금 수신측 현재 시간”**이야.
    """
    stamp = packet.get("stamp")
    if stamp is None:
        return None
    try:
        stamp = float(stamp)
    except (TypeError, ValueError):
        return None
    if stamp < 1_000_000_000.0:
        return None
    return max(0.0, now_sec - stamp)

# “지금 받은 패킷이 너무 오래된 예전 데이터인지 확인해서, 오래됐으면 버릴지 결정하는 역할”
def should_skip_packet(packet, now_sec: float, max_frame_age_sec: float, max_rgb_depth_skew_sec: float) -> bool:
    """
    고정 카메라/고정 대상 조건에서는 RGB-Depth skew 때문에 publish를 막지 않고,
    실제 transport stale 프레임만 건너뛴다.

    판단 기준:
        transport_age_sec > max_frame_age_sec
    """
    if max_frame_age_sec <= 0:
        return False

    transport_age_sec = packet_transport_age_sec(packet, now_sec)
    if transport_age_sec is not None and transport_age_sec > max_frame_age_sec:
        return True
    return False

def maybe_save_artifacts(rgb_bgr: np.ndarray, depth: np.ndarray, target, outdir: str) -> None:
    """
    시각화된 RGB 이미지를 저장
    depth를 .npy 파일로 저장
    depth 배열을 보기 쉬운 컬러 이미지로 바꿔 저장
    탐지된 물체 영역만 잘라서 따로 저장
    """
    # 1. 저장 시간 문자열 만들기
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # 2. 저장 폴더 만들기
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # 3. RGB 복사본 만들기
    vis = rgb_bgr.copy()

    # target이 있을 때 하는 일
    # 즉 탐지된 목표가 있을 때만
    # bbox나 중심점 같은 정보를 이미지에 표시하고 crop도 저장해.
    if target is not None:
        # bbox를 이미지 범위 안으로 안전하게 보정
        x1, y1, x2, y2 = clamp_xyxy(target["xyxy"], rgb_bgr.shape[1], rgb_bgr.shape[0])
        # target["pixel"]에 들어있는 중심 좌표 가져오기
        u, v = map(int, map(round, target["pixel"]))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (u, v), 3, (0, 0, 255), -1)
        label = (
            f"{target['label']} {target['conf']:.2f} Z={target['depth']:.2f}m"
            if target["depth"] is not None
            else f"{target['label']} {target['conf']:.2f}"
        )
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    
    cv2.imwrite(str(outdir_path / f"yolo_{stamp}.png"), vis)
    np.save(outdir_path / f"depth_{stamp}.npy", depth.astype(np.float32))
    save_depth_preview(depth, outdir_path / f"depth_{stamp}.png")

    if target is not None:
        x1, y1, x2, y2 = clamp_xyxy(target["xyxy"], rgb_bgr.shape[1], rgb_bgr.shape[0])
        crop_rgb = rgb_bgr[y1:y2, x1:x2]
        crop_depth = depth[y1:y2, x1:x2]
        cv2.imwrite(str(outdir_path / f"crop_{stamp}_chair.png"), crop_rgb)
        np.save(outdir_path / f"crop_depth_{stamp}_chair.npy", crop_depth.astype(np.float32))
        save_depth_preview(crop_depth, outdir_path / f"crop_depth_{stamp}_chair.png")


class ChairDetectorReceiver(Node):
    def __init__(self, args):
        super().__init__("chair_detector_receiver")
        self.args = args
        self.model = YOLO(args.weights)
        self.inference_device = 0 if torch.cuda.is_available() else "cpu"
        self.assembler = UdpChunkAssembler(stale_after_sec=10.0)
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((args.listen_host, args.listen_port))
        self.recv_sock.setblocking(False)
        self.publisher = self.create_publisher(String, args.publish_topic, 10)
        self.timer = self.create_timer(0.01, self.on_timer)
        self.artifacts_saved = True

        self.get_logger().info(
            f"Listening UDP on {args.listen_host}:{args.listen_port}, publishing detections to {args.publish_topic} "
            f"with YOLO device={self.inference_device}"
        )

    def detect_single_chair(self, packet):
        rgb = packet["rgb"]
        depth = packet["depth"]
        camera_info = packet["camera_info"]
        k = np.asarray(camera_info["k"], dtype=np.float32).reshape(3, 3)

        results = self.model.predict(
            source=rgb,
            conf=self.args.conf,
            imgsz=self.args.imgsz,
            iou=self.args.iou,
            verbose=False,
            device=self.inference_device,
            **({"classes": resolve_classes_filter(self.args.classes, self.model)} if self.args.classes else {}),
        )

        boxes = []
        if results and results[0].boxes is not None:
            raw = results[0].boxes
            xyxy = raw.xyxy.cpu().numpy()
            conf = raw.conf.cpu().numpy()
            cls = raw.cls.cpu().numpy().astype(int)
            names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
            for index in range(xyxy.shape[0]):
                box = xyxy[index].astype(float).tolist()
                u, v = bbox_center_xyxy(box)
                z = robust_depth_at(depth, u, v, patch=self.args.depth_patch, box=box)
                xyz = None if z is None else pixel_to_camera_xyz(k, u, v, z).tolist()
                pca_quat = None
                if self.args.compute_pca and xyz is not None:
                    quat = extract_crop_pca_quaternion(depth, box, k)
                    pca_quat = None if quat is None else quat.tolist()
                boxes.append(
                    {
                        "xyxy": box,
                        "conf": float(conf[index]),
                        "cls": int(cls[index]),
                        "label": names[int(cls[index])],
                        "pixel": [float(u), float(v)],
                        "depth": None if z is None else float(z),
                        "xyz_camera": xyz,
                        "quat_camera": pca_quat,
                    }
                )

        # boxes 안에 있는 여러 탐지 결과 중에서 confidence가 가장 높은 1개를 target으로 고르는 코드
        target = max(boxes, key=lambda item: item["conf"], default=None)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.args.save_artifacts and target is not None and not self.artifacts_saved:
            maybe_save_artifacts(rgb_bgr, depth, target, self.args.outdir)
            self.artifacts_saved = True

        return {
            "stamp": packet["stamp"],
            "camera_info": camera_info,
            "t_world_camera": None if packet["t_world_camera"] is None else packet["t_world_camera"].tolist(),
            "detection": target,
        }
            # 예시값
            #     "camera_info": {
            #         "rgb_stamp": 1712345678.10,
            #         "depth_stamp": 1712345678.05,
            #         "rgb_age_sec": 0.02,
            #         "depth_age_sec": 0.07
            #     }
            #     "detection": {
            #         "label": "chair",
            #         "conf": 0.91,
            #         "xyxy": [120, 40, 260, 300],
            #         "pixel": [190, 170],
            #         "depth": 2.35,
            #         "camera_xyz": [0.12, -0.08, 2.35]
            #     }

    def _process_datagram(self, datagram: bytes, now: float):
        payload = self.assembler.push(datagram)
        if payload is None:
            return

        packet = parse_frame_payload(payload)
        if should_skip_packet(packet, now, self.args.max_frame_age_sec, self.args.max_rgb_depth_skew_sec):
            return

        result = self.detect_single_chair(packet)
        if result["detection"] is None:
            return

        msg = String()
        msg.data = json.dumps(result)
        self.publisher.publish(msg)

    def on_timer(self):
        while True:
            now = time.time()
            try:
                datagram, _ = self.recv_sock.recvfrom(65_535)
            except BlockingIOError:
                break
            except Exception as exc:
                self.get_logger().warn(f"UDP receive error: {exc}")
                break

            try:
                self._process_datagram(datagram, now)
            except Exception as exc:
                self.get_logger().warn(f"Detection pipeline error: {exc}")

    def destroy_node(self):
        try:
            self.recv_sock.close()
        except Exception:
            pass
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(description="UDP RGB-D receiver with chair detection publisher")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=9002)
    parser.add_argument("--publish-topic", default="/chair_detection_json")
    parser.add_argument("--weights", default="src/robotarm_project/weights/yolov8s.pt")
    parser.add_argument("--classes", default="chair")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--depth-patch", type=int, default=7)
    parser.add_argument("--outdir", default="src/robotarm_project/yolo_results")
    parser.add_argument("--compute-pca", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--max-frame-age-sec", type=float, default=5.0)
    parser.add_argument("--max-rgb-depth-skew-sec", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init(args=None)
    node = ChairDetectorReceiver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



