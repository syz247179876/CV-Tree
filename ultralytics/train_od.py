from ultralytics.models import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-ODConv.yaml')
    model.train(data='VOC2.yaml', epochs=300, batch=32, lr0=0.015)
    # results = model.train(data='coco128.yaml', epochs=3)
    # model.val()