"""
YoloV8 baseline + EfficientViT(backbone) optimize
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8-EfficientVit-M4-backbone-optimize.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.06,
                name=f'train-{DATASET_EXDARK}-EfficientVit-M4-backbone-optimize')