"""
YoloV8 baseline + MobileViT(backbone)
"""

from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8-MobileViT-XSS.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.1, name=f'train-{DATASET_EXDARK}-MobileViT-XSS')