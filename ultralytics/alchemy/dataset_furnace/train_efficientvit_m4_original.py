"""
YoloV8 baseline + EfficientViT(backbone) 原版作为backbone
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=2)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    model = get_model('yolov8s-EfficientVit-M4-backbone.yaml')
    model.train(data=DATASET_VOC, epochs=300, batch=32, lr0=0.02,
                name=f'train-{DATASET_VOC}-EfficientVit-M4-backbone')