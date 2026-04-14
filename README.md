## 프로젝트 개요
이 프로젝트는 ROS 2, MoveIt2, Isaac Sim, RGB-D 기반 객체 인식, 그리고 Diffusion Policy를 결합하여 로봇팔이 의자를 인식하고 접근한 뒤 파지 및 들어올리기까지 수행하는 end-to-end manipulation pipeline입니다.

핵심 실행 로직은 [`chair_grasp_moveit_diffusion_policy.py`]에 구현되어 있으며, 다음 과정을 하나의 파이프라인으로 연결합니다.

- RGB-D 기반 의자 탐지
- world 좌표계 기준 3D 목표 위치 계산
- MoveIt2 IK/FK 기반 pre-grasp pose 생성
- Diffusion Policy 기반 최종 접근 및 grasp 제어
- grasp 이후 lift 동작 수행

또한 프로젝트에는 커스텀 Diffusion Policy 학습 코드인 [`diffusion_vla_pretrained.py`]와,
접근 구간을 촘촘히 기록하도록 설계한 데이터 수집 코드 [`chair_grasp_moveit_diffusion_vla_dataset.py`]도 포함되어 있습니다.

이 프로젝트에서 중점적으로 개선한 부분은 “로봇팔이 실제로 의자 가까이까지 안정적으로 접근하도록 만드는 것”입니다.
초기 정책은 grasp 직전 구간의 데이터가 충분히 촘촘하지 않아, 의자 가까이까지 내려가지 못하거나 접근 동작이 불안정한 문제가 있었습니다.
이를 해결하기 위해 접근 구간을 더 세분화하고 중간 프레임을 촘촘히 저장하는 데이터셋을 새롭게 구성했고,
그 결과 Diffusion Policy가 접근 동작 자체를 더 잘 학습할 수 있도록 개선했습니다.

---

## 기술 스택
- ROS 2 `rclpy`
- MoveIt2 `GetPositionIK`, `GetPositionFK`
- TF2 `TransformListener`, `Buffer`
- Isaac Sim
- Python
- NumPy
- OpenCV
- PyTorch
- RGB-D Camera + YOLO 기반 객체 탐지
- Panda 로봇팔 JointState 기반 제어

---

## 주요 기능
- RGB-D + YOLO 기반 의자 탐지 및 3D 위치 추정
- 카메라 좌표계에서 world 좌표계로의 변환
- MoveIt2 IK/FK를 활용한 pre-grasp / grasp / lift pose 계산
- Diffusion Policy 기반 최종 접근 및 grasp 제어
- 이미지 + instruction + robot state 기반 policy 추론
- 높이 조건을 반영한 gripper close gating
- 접근 구간을 촘촘히 기록하는 dense dataset 수집
- Diffusion Policy 학습 및 추론 파이프라인 구현
- reproducible한 학습/실행 로그 저장 구조

---

## 시스템 아키텍처

### 1. Perception
- RGB-D 기반으로 의자를 탐지합니다.
- 2D detection 결과와 depth를 이용해 3D camera 좌표를 계산합니다.
- TF 또는 `t_world_camera`를 사용해 world 좌표계로 변환합니다.

### 2. Planning
- `goal`, `pre-grasp`, `lift` 위치를 계산합니다.
- MoveIt2 IK/FK 서비스를 통해 각 pose에 대응하는 joint target을 생성합니다.
- gripper orientation은 기본적으로 top-down grasp 전략을 사용합니다.

### 3. Policy Control
- 먼저 로봇팔을 pre-grasp pose로 이동시킵니다.
- 이후 Diffusion Policy 루프를 실행합니다.
- policy는 다음 입력을 기반으로 action을 예측합니다.
  - 현재 RGB 이미지
  - 자연어 instruction
  - 현재 joint / gripper / end-effector state
- 예측된 action은 joint delta 및 gripper action으로 해석되어 실제 제어에 반영됩니다.
- close 동작은 단순 threshold만 사용하는 것이 아니라 다음 조건을 함께 고려합니다.
  - end-effector 높이 조건
  - 연속적인 close signal 확인

### 4. Post-Grasp Motion
- policy가 gripper close를 요청하면 lift 동작을 수행합니다.
- 전체 구조는 pre-grasp planning, policy 기반 접근, post-grasp lift로 역할이 분리되어 있습니다.

---

## 데이터셋 및 학습

### Dense Approach Dataset
초기 데이터는 grasp 전후 핵심 구간만 상대적으로 성기게 저장되어 있었기 때문에, policy가 “의자에 어떻게 접근해야 하는지”보다는 일부 상태에서의 동작만 제한적으로 학습하는 경향이 있었습니다.
이로 인해 실제 추론 시 의자에 충분히 가까이 접근하지 못하거나, 접근 도중 멈추는 문제가 발생했습니다.

이 문제를 해결하기 위해 [`chair_grasp_moveit_diffusion_vla_dataset.py`]에서 접근 구간을 더 촘촘하게 수집하도록 데이터셋 구성을 개선했습니다.

### 개선 내용
- approach 구간을 다음과 같이 세분화
  - `approach_far`
  - `approach_mid`
  - `approach_near`
  - `contact_ready`
- 접근 단계에서 더 짧은 간격으로 frame 저장
- grasp 직전 상태를 더 많이 포함하도록 중간 상태 데이터 확보
- policy가 단순 grasp 순간뿐 아니라 접근 과정 자체를 더 잘 학습하도록 데이터 분포 개선

### Diffusion Policy 학습
학습 및 추론 코드는 [`diffusion_vla_pretrained.py`]에 구현되어 있습니다.

학습 입력:
- image
- instruction
- robot state

학습 목표:
- future action sequence

모델 구조:
- conditional diffusion 기반 action sequence prediction
- joint delta, gripper delta, terminate signal 예측

---

## 결과

### 정성적 결과
- MoveIt2를 이용해 pre-grasp pose까지 안정적으로 이동
- 최종 접근 구간은 Diffusion Policy가 제어
- policy는 visual input과 robot state를 함께 사용해 grasp 행동을 결정
- dense approach dataset을 통해 의자에 더 가깝게 접근하는 동작 품질 향상

### 핵심 개선점
- 이 프로젝트에서 가장 중요하게 개선한 부분은 “로봇이 실제로 의자 근처까지 충분히 접근하도록 만드는 것”이었습니다.
  초기에는 policy가 의자 근처까지 안정적으로 내려오지 못하거나,
  grasp 가능한 높이에 도달하기 전에 멈추는 문제가 있었습니다.
  이를 해결하기 위해:

    - 접근 구간을 촘촘하게 수집한 dense dataset을 구성하고
    - grasp 직전 상태를 더 많이 학습에 포함시키며
    - 실행 시 close 동작에 높이 조건과 연속 신호 확인 로직을 추가했습니다.
    - 그 결과 policy가 단순히 grasp 순간만 맞추는 것이 아니라, 의자 근처까지 접근하는 과정 자체를 더 안정적으로 수행할 수 있게 되었습니다.
