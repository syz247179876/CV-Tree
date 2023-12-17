"""
YoloV8 baseline + MobileNetV2(backbone)
"""

from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=2)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO('yolov8-MobileNetV2.yaml')
    return model


if __name__ == '__main__':
    model = get_model('yolov8-MobileNetV2.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=64, lr0=0.02, name=f'train-{DATASET_EXDARK}-MobileNetV2-1.0')