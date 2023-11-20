"""
YoloV8 baseline + Bottleneck Transformer(backbone + neck)
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8n-BoT-backbone-neck.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.1, name=f'train-{DATASET_EXDARK}-BoT-backbone-neck')