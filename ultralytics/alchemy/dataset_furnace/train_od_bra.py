"""
YoloV8 baseline + Bi-Level Routing Attention(neck)
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=2)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    model = get_model('yolov8n-BRA-ODConv.yaml')
    model.train(data=DATASET_EXDARK, epochs=250, batch=32, lr0=0.02, name=f'train-{DATASET_EXDARK}-BRA-ODConv')