"""
YoloV8 baseline + Bi-Level Routing Attention(neck)
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8n-BRA-ODConv.yaml')
    model.train(data=DATASET_EXDARK, epochs=250, batch=32, lr0=0.08, name=f'train-{DATASET_EXDARK}-BRA-ODConv')