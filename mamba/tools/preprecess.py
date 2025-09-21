import csv
import numpy as np
import pickle
from tqdm import tqdm
from typing import Tuple, Optional

MAX_BRAKE_PRESSURE = 2757.89990234  # 브레이크 최대 압력 상수
SAMPLING_TIME = 0.04                # 샘플링 주기 (초)

class DatasetConverter:
    """
    CSV → (dyn_in, labels, poses) 변환기 클래스
    dyn_in: (N, T=horizon, F=8)
        [vx, vy, vtheta, throttle, steering, throttle_cmd, steering_cmd, vx_future(+future_offset)]
    labels: (N, 3) = [vx, vy, vtheta] at t + horizon
    poses:  (M, 8) = [x, y, phi, vx, vy, vtheta, throttle, steering] (로그 전구간)
    """

    def __init__(self,
                 csv_path: str,
                 horizon: int,
                 save: bool = True,
                 future_offset: int = 5,
                 min_speed: float = 5.0):
        """
        Args:
            csv_path (str): CSV 파일 경로
            horizon (int): 시퀀스 길이 (T)
            save (bool): pkl 저장 여부
            future_offset (int): dyn_in 마지막 채널로 넣을 vx의 미래 오프셋 (기본 5 step)
            min_speed (float): 주행 시작/종료를 판단할 최소 속도 (m/s)
        """
        self.csv_path = csv_path            # csv 파일 위치
        self.horizon = horizon              # 시퀀스 길이
        self.save = save                    # 변환한 결과를 pkl로 저장할지 여부
        self.future_offset = future_offset  # 미래 vx 추가할 오프셋
        self.min_speed = min_speed          # 차량이 주행 시작했다고 판단하는 최소 속도

    def csv_convert(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CSV를 읽어 데이터셋 변환 수행
        Returns:
            dyn_in: (N, T, 8) float32               학습 입력
            labels: (N, 3)    float32               학습 정답
            poses:  (M, 8)    float32 (로그 전구간)   원본 궤 (비교, 시각화 용도)
        """
        with open(self.csv_path, newline="") as f:
            csv_reader = csv.reader(f, delimiter=",")

            # 누적 버퍼
            odometry = []       # [vx, vy, vtheta, throttle, steering]
            throttle_cmds = []
            steering_cmds = []
            poses = []          # [x, y, phi, vx, vy, vtheta, throttle, steering]

            column_idxs = dict()
            previous_throttle = 0.0
            previous_steer = 0.0
            started = False

            for row in csv_reader:
                # 헤더 처리
                if not column_idxs:
                    for i in range(len(row)):
                        column_idxs[row[i].split("(")[0]] = i       # vx(m/s) ->  "vx"로 변환
                    # 필수 컬럼 존재 체크(필요 시 더 추가)
                    required = ["vx", "vy", "omega", "delta", "brake_ped_cmd", "throttle_ped_cmd",
                                "x", "y", "phi"]
                    missing = [k for k in required if k not in column_idxs]
                    if missing:
                        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
                    continue

                vx = float(row[column_idxs["vx"]])

                # 정지 상태 처리(주행 시작 전/후 컷오프)
                if abs(vx) < self.min_speed:
                    if started:
                        # 주행 구간 종료
                        break
                    # 시작 전: 초기화용 값만 갱신
                    brake = float(row[column_idxs["brake_ped_cmd"]])
                    throttle = float(row[column_idxs["throttle_ped_cmd"]])

                    if brake > 0.0:
                        previous_throttle = -brake / MAX_BRAKE_PRESSURE
                    else:
                        previous_throttle = throttle / 100.0

                    previous_steer = float(row[column_idxs["delta"]])
                    continue    # abs(vx) >= self.min_spped가 되기 전까지 아래의 코드를 실행하지 않고 위 코드에서 계속 초기화만 진행함

                # 주행 데이터 읽기
                vy = float(row[column_idxs["vy"]])
                vtheta = float(row[column_idxs["omega"]])
                steering = float(row[column_idxs["delta"]])
                brake = float(row[column_idxs["brake_ped_cmd"]])

                if brake > 0.0:
                    throttle = -brake / MAX_BRAKE_PRESSURE
                else:
                    throttle = float(row[column_idxs["throttle_ped_cmd"]]) / 100.0

                steering_cmd = steering - previous_steer
                throttle_cmd = throttle - previous_throttle

                odometry.append(np.array([vx, vy, vtheta, throttle, steering], dtype=np.float32))

                poses.append(np.array([
                    float(row[column_idxs["x"]]),
                    float(row[column_idxs["y"]]),
                    float(row[column_idxs["phi"]]),
                    vx, vy, vtheta, throttle, steering
                ], dtype=np.float32))

                previous_throttle += throttle_cmd
                previous_steer += steering_cmd

                if started:
                    throttle_cmds.append(throttle_cmd)
                    steering_cmds.append(steering_cmd)

                started = True

        # NumPy 배열 변환
        odometry = np.asarray(odometry, dtype=np.float32)           # (M, 5)
        throttle_cmds = np.asarray(throttle_cmds, dtype=np.float32) # (M-1,)
        steering_cmds = np.asarray(steering_cmds, dtype=np.float32) # (M-1,)
        poses = np.asarray(poses, dtype=np.float32)                 # (M, 8)

        # 유효 길이 확인
        # 필요한 최소 길이: horizon + 1 + future_offset + 1 (명령 벡터가 시작 한 스텝 짧음)
        min_required = self.horizon + 1 + self.future_offset
        if len(throttle_cmds) < (self.horizon + 1 + self.future_offset):
            raise ValueError(
                f"유효 주행 데이터가 부족합니다. throttle_cmds 길이={len(throttle_cmds)}, "
                f"필요 최소={min_required}"
            )

        # 샘플 개수 N 계산 (루프 범위와 정확히 일치하도록)
        N = len(throttle_cmds) - self.horizon - 1 - self.future_offset
        if N <= 0:
            raise ValueError(f"N가 0 이하입니다. 파라미터(horizon={self.horizon}, future_offset={self.future_offset})와 데이터 길이를 확인하세요.")

        # 동적 배열 생성
        dyn_in = np.zeros((N, self.horizon, 8), dtype=np.float32)
        labels = np.zeros((N, 3), dtype=np.float32)

        # 시퀀스 컴파일
        # i 기준:
        #   - 시퀀스: odometry[i : i+horizon]
        #   - 명령:   cmd[i : i+horizon]
        #   - 미래 vx: odometry[i+future_offset : i+horizon+future_offset, 0]
        #   - 라벨:   odometry[i+horizon][:3]
        for i in tqdm(range(N), desc="Compiling dataset"):
            seq_odo = odometry[i:i+self.horizon]                      # (T,5)
            seq_thr = throttle_cmds[i:i+self.horizon]                 # (T,)
            seq_str = steering_cmds[i:i+self.horizon]                 # (T,)
            seq_vxf = odometry[i+self.future_offset:i+self.horizon+self.future_offset, 0]  # (T,)

            # (T, 8): [vx, vy, vtheta, throttle, steering, thr_cmd, str_cmd, vx_future]
            dyn_in[i] = np.column_stack([seq_odo, seq_thr, seq_str, seq_vxf]).astype(np.float32)
            labels[i] = odometry[i+self.horizon, :3].astype(np.float32)

        print("Final dyn_in shape:", dyn_in.shape)
        print("Final labels shape:", labels.shape)
        print("Sampling time:", SAMPLING_TIME, "seconds")

        if self.save:
            pkl_path = self.csv_path.replace(".csv", f"_{self.horizon}.pkl")
            payload = {
                "dyn_in": dyn_in,
                "labels": labels,
                "poses": poses,
                # 멀티모달 확장 여지 (필요 시 외부에서 병합 저장 가능):
                # "images": ..., "semantics": ..., "bboxes": ..., "targets": ...
                "meta": {
                    "horizon": self.horizon,
                    "sampling_time": SAMPLING_TIME,
                    "future_offset": self.future_offset,
                    "min_speed": self.min_speed,
                    "source_csv": self.csv_path,
                }
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved dataset to {pkl_path}")

        return dyn_in, labels, poses


if __name__ == "__main__":
    converter = DatasetConverter(
        csv_path="/workspace/mamba/local_data/Putnam_park2023_run4_1.csv",
        horizon=50,
        save=True,
        future_offset=5,   # 기존 로직 유지
        min_speed=5.0      # 주행 구간 판단 속도
    )
    dyn_in, labels, poses = converter.csv_convert()
