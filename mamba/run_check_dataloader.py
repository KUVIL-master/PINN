# run_check_dataloader.py
import torch

# 패키지 경로에 맞춰 import 경로를 조정하세요.
# 예) from custom_dataloader.dataset_loader import load_pkl_dataset
from custom_dataloader.dataset_loader import load_pkl_dataset

def main():
    # ==== 사용자 설정 ====
    pkl_path = "/workspace/mamba/local_data/Putnam_park2023_run4_1_50.pkl"

    # MultiModal include 키:
    #   "x", "x_norm", "y_ctrl", "targets", "images", "semantics", "bboxes"
    include = ["x", "x_norm", "y_ctrl"]  # 필요 시 ["images","semantics","bboxes","targets"] 추가
    feature_selector = "DeepDynamics"     # "DeepDynamics", "DeepPacejka", "none"

    batch_size = 64
    batch_check_count = 2
    train_ratio = 0.8
    num_workers = 0
    pin_memory = False
    shuffle = True
    drop_last = True
    # ====================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, scaler = load_pkl_dataset(
        pkl_path=pkl_path,
        include=include,
        feature_selector=feature_selector,
        train_ratio=train_ratio,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers, 
        pin_memory=pin_memory and (device.type == "cuda"),
        device=device,
    )

    print(f"Device: {device}")
    print(f"Scaler mean shape: {scaler.mean_.shape}, var shape: {scaler.var_.shape}")

    def describe_batch(bname: str, batch: dict):
        print(f"\n=== {bname} keys: {list(batch.keys())} ===")

        def p(name):
            if name in batch and isinstance(batch[name], torch.Tensor):
                print(f"{name:>10}: shape={tuple(batch[name].shape)}, device={batch[name].device}, dtype={batch[name].dtype}")
                # 값 검증
                if torch.isnan(batch[name]).any():
                    print(f"  ⚠ NaN detected in {name}")
                if batch[name].numel() > 0:
                    print(f"  min={batch[name].min().item():.4f}, max={batch[name].max().item():.4f}")
            elif name in batch and isinstance(batch[name], list):
                L = len(batch[name])
                inner = None
                if L > 0 and isinstance(batch[name][0], list):
                    inner = len(batch[name][0])
                print(f"{name:>10}: list(len={L}, inner_len={inner})")
                if L == 0:
                    print(f"  ⚠ Empty list in {name}")

        for k in ["x", "x_norm", "y_ctrl", "targets", "images", "semantics", "bboxes"]:
            p(k)

    def check_dataloader(loader, name):
        print(f"\n=== {name} DataLoader Check ===")
        total_batches = 0
        for i, batch in enumerate(loader):
            describe_batch(f"{name} Batch {i+1}", batch)
            total_batches += 1
            if i + 1 >= batch_check_count:
                break
        if total_batches == 0:
            print(f"⚠ {name} DataLoader is empty!")

    # 실행
    check_dataloader(train_loader, "Train")
    for i, batch in enumerate(train_loader):
        
        for key in ["x", "x_norm", "y_ctrl"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    print(f"{key}: shape={tuple(val.shape)}, dtype={val.dtype}, device={val.device}")
                    print(val)  # 실제 값 출력
                else:
                    print(f"{key}: (not a tensor) type={type(val)}")

        a= batch["x"]
        b= batch["x_norm"]
        c= batch["y_ctrl"]
        

        
    check_dataloader(val_loader, "Val")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
