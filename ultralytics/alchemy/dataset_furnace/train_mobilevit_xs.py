"""
YoloV8 baseline + MobileViT(backbone)
"""

from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=2)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    model = get_model('yolov8-MobileViT-XS.yaml')
    model.train(data=DATASET_EXDARK, epochs=300, batch=32, lr0=0.01, name=f'train-{DATASET_EXDARK}-baseline-pretrained')