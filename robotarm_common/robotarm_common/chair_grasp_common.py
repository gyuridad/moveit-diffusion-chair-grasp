import io
import json
import socket
import struct
import time
import zlib
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


MAGIC = b"IGRP"
HEADER = struct.Struct("!4sIHH")
MAX_UDP_PAYLOAD = 60_000


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("RGB JPEG encoding failed")
    return encoded.tobytes()


def decode_rgb_jpeg(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("RGB JPEG decoding failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_depth_array(depth: np.ndarray) -> bytes:
    with io.BytesIO() as buf:
        np.save(buf, depth.astype(np.float32))
        return zlib.compress(buf.getvalue(), level=3)


def decode_depth_array(payload: bytes) -> np.ndarray:
    with io.BytesIO(zlib.decompress(payload)) as buf:
        return np.load(buf)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported type: {type(value)!r}")


def make_frame_payload(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_info: Dict,
    t_world_camera: Optional[np.ndarray],
    stamp: float,
) -> bytes:
    packet = {
        "stamp": float(stamp),
        "camera_info": camera_info,
        "t_world_camera": None if t_world_camera is None else np.asarray(t_world_camera, dtype=np.float32).tolist(),
        "rgb_jpeg": encode_rgb_jpeg(rgb).hex(),
        "depth_zlib_npy": encode_depth_array(depth).hex(),
    }
    return json.dumps(packet, default=_json_default).encode("utf-8")


def parse_frame_payload(payload: bytes) -> Dict:
    """
    네트워크로 받은 프레임 payload를 실제 사용 가능한 데이터로 복원하는 함수

    보낼 때는 RGB와 Depth를 그대로 보내지 않고:
        RGB는 JPEG로 압축
        Depth는 numpy 배열을 저장 후 zlib 압축
        전체를 JSON 문자열 형태로 묶어서 전송
    받는 쪽에서는 이 함수를 통해 다시:
        JSON 문자열을 파싱하고
        rgb_jpeg → 실제 RGB 이미지 배열
        depth_zlib_npy → 실제 depth 배열
        t_world_camera → numpy 배열
    로 되돌려.
    """
    packet = json.loads(payload.decode("utf-8"))
    packet["rgb"] = decode_rgb_jpeg(bytes.fromhex(packet.pop("rgb_jpeg")))
    packet["depth"] = decode_depth_array(bytes.fromhex(packet.pop("depth_zlib_npy")))
    if packet.get("t_world_camera") is not None:
        packet["t_world_camera"] = np.asarray(packet["t_world_camera"], dtype=np.float32)
    return packet


def chunk_payload(frame_id: int, payload: bytes, max_payload: int = MAX_UDP_PAYLOAD) -> List[bytes]:
    header_size = HEADER.size
    chunk_size = max_payload - header_size
    chunks = [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)] or [b""]
    total = len(chunks)
    return [
        HEADER.pack(MAGIC, frame_id, total, index) + chunk
        for index, chunk in enumerate(chunks)
    ]


class UdpChunkAssembler:
    """
    이 클래스가 하는 일은:

    패킷 헤더를 읽어서 frame_id, total, index를 확인하고
    같은 frame_id끼리 조각을 모으고
    전부 모이면 순서대로 이어서 원래 payload를 만들어 반환하는 것

    그리고 너무 오래 안 완성된 조각 묶음은 _drop_stale()로 버려.
    """

    def __init__(self, stale_after_sec: float = 10.0):
        self.stale_after_sec = stale_after_sec
        self._frames: Dict[int, Dict] = {}

    def push(self, datagram: bytes) -> Optional[bytes]:
        if len(datagram) < HEADER.size:
            return None
        magic, frame_id, total, index = HEADER.unpack(datagram[:HEADER.size])
        if magic != MAGIC:
            return None
        now = time.time()
        self._drop_stale(now)
        frame = self._frames.setdefault(frame_id, {"chunks": {}, "total": total, "ts": now})
        frame["chunks"][index] = datagram[HEADER.size:]
        frame["ts"] = now
        if len(frame["chunks"]) != frame["total"]:
            return None
        payload = b"".join(frame["chunks"][i] for i in range(frame["total"]))
        self._frames.pop(frame_id, None)
        return payload

    def _drop_stale(self, now: float) -> None:
        stale_ids = [frame_id for frame_id, frame in self._frames.items() if now - frame["ts"] > self.stale_after_sec]
        for frame_id in stale_ids:
            self._frames.pop(frame_id, None)


def send_udp_chunks(
    sock: socket.socket,
    target: Tuple[str, int],
    payload: bytes,
    frame_id: int,
    max_payload: int = MAX_UDP_PAYLOAD,
) -> None:
    for datagram in chunk_payload(frame_id, payload, max_payload=max_payload):
        sock.sendto(datagram, target)


def recv_complete_payload(sock: socket.socket, assembler: UdpChunkAssembler, timeout: Optional[float] = None) -> bytes:
    sock.settimeout(timeout)
    while True:
        datagram, _ = sock.recvfrom(65_535)
        payload = assembler.push(datagram)
        if payload is not None:
            return payload


def bbox_center_xyxy(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return float(0.5 * (x1 + x2)), float(0.5 * (y1 + y2))


def robust_depth_at(
    depth: np.ndarray,
    u: float,
    v: float,
    patch: int = 7,
    box: Optional[Iterable[float]] = None,
    fallback_box_median: bool = True,
) -> Optional[float]:
    """
    특정 픽셀의 depth 값을 좀 더 안정적으로 구하는 함수

    (u, v) 주변의 작은 패치(기본 7x7)를 잘라서
    그 안에서 유효한 depth 값만 모아
    중앙값(median) 을 반환해

    “한 픽셀의 깊이를 대충 찍는 게 아니라, 주변 상황을 보고 믿을 만한 깊이를 뽑아내는 함수”
    """
    height, width = depth.shape[:2]
    cx, cy = int(round(u)), int(round(v))
    half = max(1, patch // 2)
    u1 = max(0, cx - half)
    v1 = max(0, cy - half)
    u2 = min(width, cx + half + 1)
    v2 = min(height, cy + half + 1)
    roi = depth[v1:v2, u1:u2]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size:
        return float(np.median(valid))
    if fallback_box_median and box is not None:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            roi = depth[y1:y2, x1:x2]
            valid = roi[np.isfinite(roi) & (roi > 0)]
            if valid.size:
                return float(np.median(valid))
    return None

# 이미지 픽셀 좌표(u, v)와 depth(z)를 이용해서 카메라 3D 좌표계의 점 (x, y, z)를 계산하는 함수
def pixel_to_camera_xyz(k: np.ndarray, u: float, v: float, z: float) -> np.ndarray:
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def camera_to_world(point_camera: np.ndarray, t_world_camera: np.ndarray) -> np.ndarray:
    point_h = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0], dtype=np.float32)
    return (t_world_camera @ point_h)[:3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def world_to_robot(point_world: np.ndarray, t_world_robot: np.ndarray) -> np.ndarray:
    return (invert_transform(t_world_robot) @ np.array([*point_world, 1.0], dtype=np.float32))[:3]


def depth_crop_to_point_cloud(depth_crop: np.ndarray, k: np.ndarray, u0: int, v0: int) -> np.ndarray:
    height, width = depth_crop.shape[:2]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    points = []
    for v in range(height):
        for u in range(width):
            z = depth_crop[v, u]
            if not np.isfinite(z) or z <= 0:
                continue
            u_img = u + u0
            v_img = v + v0
            x = (u_img - cx) * z / fx
            y = (v_img - cy) * z / fy
            points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def pca_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    order = np.argsort(-eigenvalues)
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.real(eigenvectors[:, order])
    for index in range(3):
        eigenvectors[:, index] /= np.linalg.norm(eigenvectors[:, index]) + 1e-8
    return centroid, eigenvectors, eigenvalues


def rotation_from_pca(pc1: np.ndarray, pc2: np.ndarray) -> np.ndarray:
    axis_x = pc1 / (np.linalg.norm(pc1) + 1e-8)
    axis_y = pc2 / (np.linalg.norm(pc2) + 1e-8)
    axis_z = np.cross(axis_x, axis_y)
    axis_z /= np.linalg.norm(axis_z) + 1e-8
    return np.stack([axis_x, axis_y, axis_z], axis=1)


def mat_to_quat(rotation_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rotation_matrix).as_quat()


def extract_crop_pca_quaternion(depth: np.ndarray, box_xyxy: Iterable[float], k: np.ndarray) -> Optional[np.ndarray]:
    """
    바운딩박스 안의 depth 데이터를 이용해서 물체의 대략적인 3D 방향(자세)을 쿼터니언으로 추정하는 함수

    바운딩박스 영역을 depth에서 잘라냄
    그 crop을 3D 포인트클라우드로 변환
    그 점들의 퍼진 방향을 PCA로 분석
    가장 길게 퍼진 방향, 두 번째로 퍼진 방향을 축으로 사용
    회전행렬 생성
    마지막에 쿼터니언으로 변환
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    crop = depth[y1:y2, x1:x2]
    point_cloud = depth_crop_to_point_cloud(crop, k, x1, y1)
    if len(point_cloud) < 20:
        return None
    _, eigvecs, _ = pca_points(point_cloud)
    rotation_matrix = rotation_from_pca(eigvecs[:, 0], eigvecs[:, 1])
    return mat_to_quat(rotation_matrix)


def extract_roll_delta_from_pca(goal_quat_xyzw: np.ndarray, quat_pca_xyzw: np.ndarray) -> float:
    current = R.from_quat(goal_quat_xyzw)
    desired = R.from_quat(quat_pca_xyzw)
    relative = desired * current.inv()
    rotvec = relative.as_rotvec()
    ee_z_world = current.apply([0.0, 0.0, 1.0])
    ee_z_world /= np.linalg.norm(ee_z_world) + 1e-9
    return float(np.dot(rotvec, ee_z_world))
