"""
YoloV8 baseline + FasterNet(backbone)
"""
from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-FasterNetVs.yaml')
    model.train(data='ExDark.yaml', epochs=300, batch=32, lr0=0.1, name='train_FasterNetVs')