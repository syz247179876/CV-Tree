"""
YoloV8 baseline
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8n-AFPN.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.01, name=f'train-{DATASET_EXDARK}-AFPN')