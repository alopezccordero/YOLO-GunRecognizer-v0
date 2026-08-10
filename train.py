from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_ROOT / "gun_dataset" / "dataset.yaml"


def main() -> None:
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Dataset:", DATASET_YAML)
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo26s.pt")

    model.train(
        data=str(DATASET_YAML),
        epochs=200,
        imgsz=960,
        batch=16,
        device=device,
        workers=8,
        patience=30,
        save=True,
        save_period=10,
        project=str(PROJECT_ROOT / "runs" / "gun_detector"),
        name="yolo26s_960_aug_2",
        seed=42,
        verbose=True,
        mixup=0.1,
        copy_paste=0.2,
        cos_lr=True
    )


if __name__ == "__main__":
    main()