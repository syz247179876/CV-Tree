"""
YoloV8 baseline + EfficientViT(backbone) smaller
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    # last_model = r'C:\yolov8\runs\detect\train-ExDark.yaml-EfficientVit-M5-backbone-optimize\weights\best.pt'
    yaml_file = r'yolov8-EfficientViT-M5-backbone-optimize.yaml'
    model = YOLO(yaml_file)
    model.train(data=DATASET_EXDARK, epochs=300, batch=16, lr0=0.01,
                name=f'train-{DATASET_EXDARK}-EfficientVit-M5-backbone-optimize', resume=True)