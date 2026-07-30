from pathlib import Path
import shutil

import fiftyone as fo
import fiftyone.utils.random as four
from fiftyone import ViewField as F


DATASET_NAME = "open-images-guns"
EXPORT_DIR = Path("gun_dataset").resolve()

WEAPON_CLASSES = ["Handgun", "Rifle", "Shotgun"]


def main() -> None:
    dataset = fo.load_dataset(DATASET_NAME)
    dataset.reload()

    print(f"Loaded dataset: {dataset.name}")
    print(f"Dataset count: {dataset.count()}")
    print("Existing tags:", dataset.count_sample_tags())

    downloaded_classes = dataset.distinct(
        "ground_truth.detections.label"
    )

    print("Downloaded classes:", downloaded_classes)

    # Keep only images containing a firearm and remove all
    # non-firearm bounding boxes from those images
    weapon_view = dataset.filter_labels(
        "ground_truth",
        F("label").is_in(WEAPON_CLASSES),
        only_matches=True,
    )

    print(f"Images containing firearms: {weapon_view.count()}")

    # Convert all firearm categories into one YOLO class
    weapon_view = weapon_view.map_values(
        "ground_truth.detections.label",
        {
            "Handgun": "gun",
            "Rifle": "gun",
            "Shotgun": "gun",
        },
    )

    # Remove tags from any previous failed attempt
    weapon_view.untag_samples(
        ["yolo_train", "yolo_val", "yolo_test"]
    )

    # Use unique tags that do not conflict with the existing
    # Open Images "train" tag
    four.random_split(
        weapon_view,
        {
            "yolo_train": 0.80,
            "yolo_val": 0.10,
            "yolo_test": 0.10,
        },
        seed=42,
    )

    split_mapping = {
        "train": "yolo_train",
        "val": "yolo_val",
        "test": "yolo_test",
    }

    print("\nSplit sizes:")

    for export_split, fiftyone_tag in split_mapping.items():
        split_view = weapon_view.match_tags(fiftyone_tag)
        print(f"  {export_split}: {split_view.count()}")

    # Remove incomplete output from the interrupted export
    if EXPORT_DIR.exists():
        print(f"\nRemoving old export: {EXPORT_DIR}")
        shutil.rmtree(EXPORT_DIR)

    for export_split, fiftyone_tag in split_mapping.items():
        split_view = weapon_view.match_tags(fiftyone_tag)

        print(
            f"\nExporting {export_split}: "
            f"{split_view.count()} images"
        )

        split_view.export(
            export_dir=str(EXPORT_DIR),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=export_split,
            classes=["gun"],
        )

    print("\nExport completed.")
    print(f"Dataset location: {EXPORT_DIR}")
    print(f"YAML file: {EXPORT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()