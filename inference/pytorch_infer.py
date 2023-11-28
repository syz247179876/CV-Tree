import json
import os
import time


import torch
import typing as t

from inference.base_infer import BaseInference
from ultralytics.utils import ops


class PytorchInference(BaseInference):

    """
    Inference based on Pytorch
    """

    def __init__(
            self,
            model_path: str,
            image_path: str,
            conf_thres: float = 0.3,
            iou_thres: float = 0.05,
            data_classes: t.Union[t.Dict, str] = 'coco8.yaml',
            max_det: int = 1000,
            img_size: t.Tuple = (640, 640),
            half: bool = True,
            fuse: bool = True,
            use_gpu: bool = True,
    ):
        super().__init__(model_path, image_path, conf_thres, iou_thres, data_classes, max_det, img_size, half, fuse,
                         use_gpu)

    @torch.no_grad()
    def _init(self):
        """
        load model from model_path
        Returns: model
        """
        ckpt = torch.load(self.model_path, map_location=self.device)
        model = (ckpt.get('ema') or ckpt['model']).to(self.device).float()
        if self.fuse:
            model = model.fuse()
        if self.half:
            model.half()
        model.eval()
        return model

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.
        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        image_data = super(PytorchInference, self).preprocess()

        image_data = torch.from_numpy(image_data).to(self.device)
        if self.half:
            image_data = image_data.type(torch.cuda.HalfTensor)
        return image_data


    def main(self):
        """
         Performs inference using a different model or inference engine and returns the dict of output image

        Returns:
            output_img: The output image with drawn detections.

        """
        image = self.preprocess()
        pred = self.model(image, augment=False, visualize=False)[0]
        return self.postprocess(resize_shape=image.shape[2:], origin_shape=self.img.shape, pred=pred)



    def postprocess(self, resize_shape: t.Tuple, origin_shape: t.Tuple, pred: torch.Tensor):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
        Returns: dict
        """
        pred = ops.non_max_suppression(
            pred, self.conf_thres, self.iou_thres, None, False, max_det=self.max_det)
        fake_result = {"model_data": {"objects": []}}
        for i, det in enumerate(pred):
            det[:, :4] = ops.scale_boxes(
                origin_shape, det[:, :4], origin_shape).round()
            for *xyxy, conf, cls in reversed(det):
                xyxy_list = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                conf_list = conf.tolist()
                label = self.classes[int(cls)]
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
    model_path = r'D:\projects\yolov8\ultralytics\ultralytics\alchemy\dataset_furnace\yolov8n.pt'
    image_path = r'D:\projects\ultralytics\ultralytics\assets\bus.jpg'
    obj = PytorchInference(model_path, image_path,
                           data_classes=r'D:\projects\yolov8\ultralytics\ultralytics\cfg\datasets\coco8.yaml',
                           half=True, fuse=True, use_gpu=True)
    t1 = time.time()
    res = obj.main()
    t2 = time.time()
    print(t2 - t1)
