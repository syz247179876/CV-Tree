"""
YoloV8 baseline
"""
from ultralytics.models import YOLO
from ultralytics.alchemy.settings import *
from ultralytics.utils.wraps import log_profile

@log_profile('thop', shape=(3, 640, 640), batch_size=1, show_detail=True)
def get_model(model: str, device: str = 'cuda'):
    model = YOLO(model)
    return model

if __name__ == '__main__':
    model = get_model(r'C:\yolov8\runs\detect\train-FLIR-ADAS-V2.yaml-GhostConv2-16-woeca\weights\last.pt')
    print(model)
    # model.train(data=DATASET_FLIR_ADAS_V2, epochs=300, batch=16, lr0=0.01, name=f'train-{DATASET_FLIR_ADAS_V2}-GhostConv2-16-woeca', resume=True)