"""
YoloV8 baseline + EfficientViT(backbone) smaller
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=2)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    yaml_file = r'yolov8-EfficientViT-M5-backbone-optimize.yaml'
    model = get_model(last_model)
    model.train(data=DATASET_EXDARK, epochs=300, batch=16, lr0=0.01,
                name=f'train-{DATASET_EXDARK}-EfficientVit-M5-backbone-optimize', resume=True)