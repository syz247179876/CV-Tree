"""
YoloV8 baseline + Coordinate Attention(neck)
"""
from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-CA-neck.yaml')
    model.train(data='ExDark.yaml', epochs=300, batch=32, lr0=0.1, name='train_CA-neck')