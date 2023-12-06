"""
YoloV8 baseline + EfficientViT(backbone) smaller
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8-EfficientVit-M4-ODConv.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=16, lr0=0.01,
                name=f'train-{DATASET_EXDARK}-EfficientVit-M4-ODConv')