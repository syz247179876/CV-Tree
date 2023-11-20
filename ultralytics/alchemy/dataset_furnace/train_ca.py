"""
YoloV8 baseline + Coordinate Attention(neck)
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8-CA-neck.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.1, pretrained='yolov8n.pt',
                name=f'train-{DATASET_EXDARK}-CA-neck-pretrained', freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 18, 20, 20, 22, 24])