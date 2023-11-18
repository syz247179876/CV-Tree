"""
YoloV8 baseline + PConv(backbone and neck)
"""
from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-Faster-backbone-neck.yaml')
    model.train(data='ExDark.yaml', epochs=300, batch=32, lr0=0.1, name='train_Faster-backbone-neck')