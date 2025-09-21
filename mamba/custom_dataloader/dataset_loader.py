# dataset_loader.py
import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any, List, Set

# ===============================
# Collate: 텐서는 stack, 리스트는 그대로 유지
# ===============================
def dict_collate_keep_lists(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(v0, dict):
            out[k] = dict_collate_keep_lists(vals)
        else:
            out[k] = vals  # 리스트/None/문자열 등은 리스트로 유지
    return out


# ===============================
# 특성 선택기(기존 DeepDynamics/DeepPacejka 규칙)
# ===============================
def apply_feature_selector(features: np.ndarray, selector: str) -> np.ndarray:
    """
    features: (N, T, F) 원본
    selector:
      - "DeepDynamics": 앞 7개 특성만 사용
      - "DeepPacejka":   axis=2에서 인덱스 3(throttle), 5(steering_cmd) 제거
      - "none":          그대로 사용
    """
    if selector == "DeepDynamics":
        return features[:, :, :7]
    elif selector == "DeepPacejka":
        return np.delete(features, [3, 5], axis=2)
    elif selector == "none":
        return features
    else:
        raise ValueError(f"Unknown feature_selector: {selector}. Use one of ['DeepDynamics','DeepPacejka','none'].")


# ===============================
# 멀티모달 Dataset (딕셔너리 반환)
# ===============================
class MultiModalDataset(torch.utils.data.Dataset):
    """
    포함 가능한 키 (include로 선택):
      "x", "x_norm", "y_ctrl", "targets", "images", "semantics", "bboxes"
    내부 보관:
      X_data, X_norm, y_ctrl, targets, images, semantics, bboxes_list
    """
    def __init__(
        self,
        data_dict: Dict[str, Any],
        include: Set[str],
        scaler: Optional[StandardScaler],
        device: torch.device,
        feature_selector: str = "none",
    ):
        self.include = set(include)
        self.device = device

        # ---- 필수 입력 dyn_in/features ----
        if "dyn_in" in data_dict:
            feats = data_dict["dyn_in"]
        elif "features" in data_dict:
            feats = data_dict["features"]
        else:
            raise KeyError("Input must contain 'dyn_in' or 'features' key.")

        assert isinstance(feats, np.ndarray) and feats.ndim == 3, "dyn_in/features must be (N,T,F)"
        feats = apply_feature_selector(feats, feature_selector)

        # ---- y_ctrl (labels) ----
        y_ctrl = data_dict.get("labels", None)
        if ("y_ctrl" in self.include) and (y_ctrl is None):
            raise KeyError("include에 'y_ctrl'가 있지만 데이터에 'labels'가 없습니다.")

        # ---- scaler & 정규화 ----
        N, T, F = feats.shape
        flat = feats.reshape((-1, F))
        if scaler is None:
            self.scaler = StandardScaler()
            norm_flat = self.scaler.fit_transform(flat)
        else:
            self.scaler = scaler
            norm_flat = self.scaler.transform(flat)

        self.X_data = torch.from_numpy(feats).float().to(device)
        self.X_norm = torch.from_numpy(norm_flat.reshape((N, T, F))).float().to(device)

        # ---- y_ctrl ----
        if y_ctrl is not None:
            assert isinstance(y_ctrl, np.ndarray) and y_ctrl.ndim == 2, "labels must be (N, Y)"
            self.y_ctrl = torch.from_numpy(y_ctrl).float().to(device)
        else:
            self.y_ctrl = None

        # ---- targets ----
        targets = data_dict.get("targets", None)
        if targets is not None:
            if np.issubdtype(targets.dtype, np.integer):
                self.targets = torch.from_numpy(targets).long().to(device)
            else:
                self.targets = torch.from_numpy(targets).float().to(device)
        else:
            self.targets = None

        # ---- images ----
        images = data_dict.get("images", None)
        if images is not None:
            # (N, T, C, H, W)
            img_t = torch.from_numpy(images)
            if img_t.dtype != torch.float32:
                img_t = img_t.float()
                # uint8 등은 0..255 → 0..1 스케일
                img_t = img_t / 255.0
            self.images = img_t.to(device)
        else:
            self.images = None

        # ---- semantics ----
        semantics = data_dict.get("semantics", None)
        if semantics is not None:
            sem_t = torch.from_numpy(semantics).long().to(device)  # class ID
            self.semantics = sem_t
        else:
            self.semantics = None

        # ---- bboxes (가변 길이) ----
        bboxes = data_dict.get("bboxes", None)
        self.bboxes_list: Optional[List[List[torch.Tensor]]] = None
        if bboxes is not None:
            # 기대형태: bboxes[n][t] = np.ndarray(Mt, 4/5)
            # object 배열일 수 있으므로 파싱
            parsed: List[List[torch.Tensor]] = []
            for n in range(len(bboxes)):
                seq = bboxes[n]
                if isinstance(seq, np.ndarray) and seq.dtype == object:
                    seq = list(seq)
                cur_list: List[torch.Tensor] = []
                for arr in seq:
                    if arr is None:
                        cur_list.append(torch.empty((0, 5), dtype=torch.float32, device=device))
                        continue
                    tarr = torch.as_tensor(arr, dtype=torch.float32, device=device)
                    # 4열만 있으면 cls=-1 붙이기
                    if tarr.numel() > 0 and tarr.shape[-1] < 5:
                        pad = torch.full((tarr.shape[0], 5 - tarr.shape[1]), -1.0, device=device)
                        tarr = torch.cat([tarr, pad], dim=-1)
                    cur_list.append(tarr)
                parsed.append(cur_list)
            self.bboxes_list = parsed

        # 길이 체크(가능한 키들 중 하나 기준)
        self.N = N

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "x" in self.include:
            out["x"] = self.X_data[idx]         # (T, F)
        if "x_norm" in self.include:
            out["x_norm"] = self.X_norm[idx]    # (T, F)
        if "y_ctrl" in self.include and self.y_ctrl is not None:
            out["y_ctrl"] = self.y_ctrl[idx]    # (Y,)
        if "targets" in self.include and self.targets is not None:
            out["targets"] = self.targets[idx]
        if "images" in self.include and self.images is not None:
            out["images"] = self.images[idx]           # (T, C, H, W)
        if "semantics" in self.include and self.semantics is not None:
            out["semantics"] = self.semantics[idx]     # (T, H, W)
        if "bboxes" in self.include and self.bboxes_list is not None:
            out["bboxes"] = self.bboxes_list[idx]      # List[Tensor(Mt,5)]
        return out


# ===============================
# 공통: Train/Val 분할
# ===============================
def _split_np_dict(np_dict: Dict[str, Any], train_ratio: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    np_dict의 각 키에 대해 첫 축(N) 기준으로 분할.
    존재하는 키만 분할하여 반환.
    """
    # N 추출(가능한 키 중 하나 기준)
    for k in ["dyn_in", "features", "labels", "targets", "images", "semantics", "bboxes"]:
        if k in np_dict:
            N = len(np_dict[k])
            break
    else:
        raise ValueError("No valid keys found to infer N.")
    split_id = int(N * train_ratio)

    def split_val(v):
        if isinstance(v, np.ndarray):
            return v[:split_id], v[split_id:]
        else:
            # 리스트/오브젝트 배열 등
            return v[:split_id], v[split_id:]

    train_dict, val_dict = {}, {}
    for k, v in np_dict.items():
        v_tr, v_va = split_val(v)
        train_dict[k] = v_tr
        val_dict[k] = v_va
    return train_dict, val_dict


# ===============================
# 파일 로더 (npz / pkl) → dict
# ===============================
def _load_any_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in ["dyn_in", "features", "labels", "targets", "images", "semantics", "bboxes"]:
        if k in data:
            out[k] = data[k]
    if not out:
        raise ValueError("Empty or unsupported .npz contents.")
    return out

def _load_any_pkl(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Pickle must contain a dict.")
    out: Dict[str, Any] = {}
    for k in ["dyn_in", "features", "labels", "targets", "images", "semantics", "bboxes"]:
        if k in data:
            out[k] = data[k]
    if not out:
        raise ValueError("Empty or unsupported .pkl contents.")
    return out


# ===============================
# Public API: 로더 (딕셔너리 구성 + DataLoader)
# ===============================
def load_npz_dataset(
    npz_path: str,
    include: List[str] = ("x", "x_norm", "y_ctrl"),
    feature_selector: str = "none",
    train_ratio: float = 0.8,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, StandardScaler]:
    """
    .npz에서 필요한 키만 골라(dict) DataLoader 생성
    include: 반환할 키 목록 ["x","x_norm","y_ctrl","targets","images","semantics","bboxes"]
    """
    raw = _load_any_npz(npz_path)
    train_raw, val_raw = _split_np_dict(raw, train_ratio)

    # train으로 scaler fit
    train_ds = MultiModalDataset(train_raw, include=set(include), scaler=None, device=device, feature_selector=feature_selector)
    scaler = train_ds.scaler
    # val은 같은 scaler 사용
    val_ds   = MultiModalDataset(val_raw,   include=set(include), scaler=scaler, device=device, feature_selector=feature_selector)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=(device.type == "cuda" and pin_memory),
        collate_fn=dict_collate_keep_lists,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda" and pin_memory),
        collate_fn=dict_collate_keep_lists,
    )
    return train_loader, val_loader, scaler


def load_pkl_dataset(
    pkl_path: str,
    include: List[str] = ("x", "x_norm", "y_ctrl"),
    feature_selector: str = "none",
    train_ratio: float = 0.8,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, StandardScaler]:
    """
    .pkl에서 필요한 키만 골라(dict) DataLoader 생성
    include: 반환할 키 목록 ["x","x_norm","y_ctrl","targets","images","semantics","bboxes"]
    """
    raw = _load_any_pkl(pkl_path)
    train_raw, val_raw = _split_np_dict(raw, train_ratio)

    train_ds = MultiModalDataset(train_raw, include=set(include), scaler=None, device=device, feature_selector=feature_selector)
    scaler = train_ds.scaler
    val_ds   = MultiModalDataset(val_raw,   include=set(include), scaler=scaler, device=device, feature_selector=feature_selector)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=(device.type == "cuda" and pin_memory),
        collate_fn=dict_collate_keep_lists,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda" and pin_memory),
        collate_fn=dict_collate_keep_lists,
    )
    return train_loader, val_loader, scaler


def load_dataset_auto(
    path: str,
    include: List[str] = ("x", "x_norm", "y_ctrl"),
    feature_selector: str = "none",
    **kwargs,
):
    """
    확장자에 따라 .npz / .pkl 자동 로드 (딕셔너리 배치)
    """
    if path.lower().endswith(".npz"):
        return load_npz_dataset(path, include=include, feature_selector=feature_selector, **kwargs)
    elif path.lower().endswith(".pkl"):
        return load_pkl_dataset(path, include=include, feature_selector=feature_selector, **kwargs)
    else:
        raise ValueError("지원하지 않는 확장자입니다. .npz 또는 .pkl 파일을 사용하세요.")
