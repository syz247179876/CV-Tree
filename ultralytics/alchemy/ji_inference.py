"""
ji inference testing script
"""
import sys
if '/project/train/src_repo/ultralytics-main' not in sys.path:
    sys.path.insert(0, '/project/train/src_repo/ultralytics-main')
    sys.path.insert(0, '/project/train/src_repo/ultralytics-main/ultralytics/')
import cv2
import torch
import json
import time

from ultralytics.utils import ops
from ultralytics.data import augment

device = torch.device("cuda:0")

model_path = r'C:\yolov8\runs\detect\train-ExDark.yaml-EfficientVit-M5-backbone-optimize\weights\best.pt'
# model_path = '/project/train/models/train/train-non-motorized-vehicle-yolov8n-BRA-ODConv/weights/best.pt'

half = False # use FP16 half-precision inference
@torch.no_grad()
def init():
    weights = model_path
    device = 'cuda:0'
    ckpt = torch.load(weights, map_location='cpu')
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()
    model = model.fuse().eval()
    if half:
        model.half()
    # model.eval()
    return model

def process_image(handle=None, input_image=None, args=None, **kwargs):
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold

    max_det = 1000  # maximum detections per image
    imgsz = [1024, 1024]
    names = {
      # 非机动车
        0: 'own_bicycle',
        1: 'meituan_bicycle',
        2: 'haluo_bicycle',
        3: 'qingju_bicycle',
        4: 'other_type_bicycle',
        5: 'motorbike',
        6: 'electric_scooter',
        7: 'tricycle',
        8: 'auto_tricycle',

        # 无人骑的非机动车
        9: 'none_person_own_bicycle',
        10: 'none_person_meituan_bicycle',
        11: 'none_person_haluo_bicycle',
        12: 'none_person_qingju_bicycle',
        13: 'none_person_other_type_bicycle',
        14: 'none_person_motorbike',
        15: 'none_person_electric_scooter',
        16: 'none_person_tricycle',
        17: 'none_person_auto_tricycle',

        # 未知是否有人骑的非机动车
        18: 'unknown_person_own_bicycle',
        19: 'unknown_person_meituan_bicycle',
        20: 'unknown_person_haluo_bicycle',
        21: 'unknown_person_qingju_bicycle',
        22: 'unknown_person_other_type_bicycle',
        23: 'unknown_person_motorbike',
        24: 'unknown_person_electric_scooter',
        25: 'unknown_person_tricycle',
        26: 'unknown_person_auto_tricycle',

        # 有人骑的非机动车
        27: 'own_bicycle_person',
        28: 'meituan_bicycle_person',
        29: 'haluo_bicycle_person',
        30: 'qingju_bicycle_person',
        31: 'other_type_bicycle_person',
        32: 'motorbike_person',
        33: 'electric_scooter_person',
        34: 'tricycle_person',
        35: 'auto_tricycle_person',
        36: 'person',
        37: 'rider',
    }

    stride = 32
    fake_result = {}
    fake_result["model_data"] = {"objects": []}

    # resize img and padding
    letterbox = augment.LetterBox(imgsz, center=True, stride=stride)
    img = letterbox(image=input_image)
    img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
    img = img / 255  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, dim=0).to(device)
    if half:
        img = img.half()
    else:
        img = img.float()

    pred = handle(img, augment=False, visualize=False)[0]
    pred = ops.non_max_suppression(
        pred, conf_thres, iou_thres, None, False, max_det=max_det)

    for det in pred:
        det[:, :4] = ops.scale_boxes(
            img.shape[2:], det[:, :4], input_image.shape).round()
        for *xyxy, conf, cls in det:
            xyxy_list = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
            conf_list = conf.tolist()
            label = names[int(cls)]
            fake_result['model_data']['objects'].append({
                "xmin": int(xyxy_list[0]),
                "ymin": int(xyxy_list[1]),
                "xmax": int(xyxy_list[2]),
                "ymax": int(xyxy_list[3]),
                "confidence": conf_list,
                "name": label
            })

    return json.dumps(fake_result, indent=4)

if __name__ == '__main__':
    img = cv2.imread(r'C:\dataset\OpenDataLab___ExDark\ExDark_yolo\ExDark_yolo\images\temp\2015_00009.jpg')
    model = init()
    t1 = time.time()
    process_image(model, input_image=img)
    t2 = time.time()
    print(t2 - t1)