"""
YoloV8 baseline + EfficientFormer(backbone)
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=1)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    model = get_model('EfficientViT-M4-AFPN.yaml')
    model.train(data=DATASET_VOC, epochs=300, batch=64, lr0=0.02,
                name=f'train-{DATASET_VOC}-EfficientViT-M4-AFPN')