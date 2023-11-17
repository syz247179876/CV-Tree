"""
YoloV8 baseline + FasterNet(backbone)
"""
from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-FasterNetV1.yaml')
    model.train(data='ExDark.yaml', epochs=300, batch=1, lr0=0.1)