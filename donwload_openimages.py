from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


# This is the YOLO-GunRecognizer-v0 repository folder
REPO_ROOT = Path(__file__).resolve().parent

# open-images-v7 is directly inside the repository,
# so the repository itself is the dataset zoo directory
fo.config.dataset_zoo_dir = str(REPO_ROOT)

dataset_path = REPO_ROOT / "open-images-v7"
train_path = dataset_path / "train"

print("Repository:", REPO_ROOT)
print("Dataset path:", dataset_path)
print("Dataset exists:", dataset_path.exists())
print("Train folder exists:", train_path.exists())
print()
print("Downloaded zoo datasets:")
print(foz.list_downloaded_zoo_datasets())

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Handgun", "Rifle", "Shotgun"],
    dataset_name="open-images-guns",
    download_if_necessary=False,
    persistent=True,
)

print("\nDataset loaded successfully")
print("Dataset name:", dataset.name)
print("Number of images:", len(dataset))
print("Registered datasets:", fo.list_datasets())

print("\nFields:")
print(dataset.get_field_schema())

print("\nWeapon-label counts:")
print(
    dataset.count_values(
        "ground_truth.detections.label"
    )
)