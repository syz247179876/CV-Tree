"""
YoloV8 baseline + SKBlock(neck)
"""

from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=2, show_detail=True)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    model = get_model(r'C:\yolov8\runs\detect\train-VOC2.yaml-SK-neck\weights\best.pt')
    model.train(data=DATASET_VOC, epochs=300, batch=32, lr0=0.01, name=f'train-{DATASET_VOC}-SK-neck', resume=True)