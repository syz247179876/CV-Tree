"""
YoloV8 baseline + ODConv(neck)
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8n-ODConv-neck.yaml')
    model.train(data=DATASET_COCO8, epochs=8, batch=32, lr0=0.1, name=f'train-{DATASET_EXDARK}-ODConv-neck')