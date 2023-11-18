"""
YoloV8 baseline + SKBlock(neck)
"""

from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-SK-neck.yaml')
    model.train(data='ExDark.yaml', epochs=300, batch=32, lr0=0.1)