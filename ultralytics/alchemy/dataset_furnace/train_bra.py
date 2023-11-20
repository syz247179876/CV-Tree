"""
YoloV8 baseline + Bi-Level Routing Attention(neck)
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *

if __name__ == '__main__':
    model = YOLO('yolov8n-BiFormer-neck.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.1, name=f'train-{DATASET_EXDARK}-BiFormer-neck-pretrained',
                pretrained='yolov8n.pt', freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 19, 20, 22])