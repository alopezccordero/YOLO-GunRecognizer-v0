from ultralytics import YOLO
import argparse



def main(SRC):
    model = YOLO("runs/gun_detector/yolo26s_1280_v3/weights/best.pt")
    src = f"./{SRC}"

    model.predict(source=src, save=True, conf=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src",
        help="Path to the input video, image or webcam index"
    )
    args = parser.parse_args()

    main(args.src)