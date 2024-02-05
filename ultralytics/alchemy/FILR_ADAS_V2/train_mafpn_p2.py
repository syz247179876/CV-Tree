"""
YoloV8 baseline
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=2, show_detail=True)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    model = get_model(r'yolov8n-MAFPN-p2-FILR.yaml')
    model.train(data=DATASET_FLIR_ADAS_V2, epochs=300, batch=16, lr0=0.01, name=f'train-{DATASET_FLIR_ADAS_V2}-MAFPN-v8n-p2', resume=False)