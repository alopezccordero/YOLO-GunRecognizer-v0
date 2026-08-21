
from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)   # same key as in add_roboflow_data.py

workspace = rf.workspace("alejandro-lopez-cordero")

workspace.deploy_model(
    model_type="yolo26",   # ⚠️ see note below — your weights are yolo26s
    model_path="./runs/gun_detector/yolo26s_1280_v2/",
    project_ids=["yolo-gundetection-lzsoh"],
    model_name="gun-detector-v2",
)