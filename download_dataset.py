import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Handgun", "Rifle", "Shotgun"],
    max_samples=5000,
    shuffle=True,
    seed=42,
)

print(dataset)