
from roboflow import Roboflow

rf = Roboflow(api_key="Q1rmJssbcuaaFdFnoBpz")   # same key as in add_roboflow_data.py
workspace = rf.workspace("alejandro-lopez-cordero")

workspace.deploy_model(
    model_type="yolo26",   # ⚠️ see note below — your weights are yolo26s
    model_path="./runs/gun_detector/yolo26s_1280_diverse_2_downsample/",
    project_ids=["yolo-gundetection"],
    model_name="gun-detector-downsample",
)