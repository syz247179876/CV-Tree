"""
YoloV8 baseline + SEBlock(neck)
"""

from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8n-SE-neck.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.1, name=f'train-{DATASET_EXDARK}SE-neck-pretrained',
                pretrained='yolov8n.pt', freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 18, 20, 22, 24])