from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-CA.yaml')
    model.train(data='VOC2.yaml', epochs=300, batch=32)