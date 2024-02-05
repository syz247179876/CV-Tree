"""
This file is used to visualize feature maps
"""
import cv2
import numpy as np
import torch.nn as nn
import torch
import typing as t

from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from pathlib import Path
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM



class Visualize(object):

    def __init__(
            self,
            weight: str,
    ):
        self.weight = weight
        self.model = None

    def load_model(self):
        # ckpt = YOLO(weight)
        # model = ckpt.model
        ckpt = torch.load(self.weight, map_location='cpu')
        model = (ckpt.get('ema') or ckpt['model']).to(0).float()
        model = model.eval()
        setattr(self, 'model', model)
        return model

    @staticmethod
    def read_img(image_path: str):
        origin_img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    @staticmethod
    def preprocess() -> transforms.Compose:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),
            # transforms.CenterCrop(224),
        ])
        # new_input = transforms.Normalize([0.476, 0.442, 0.349], [0.260, 0.237, 0.269])
        return trans

    def visualize_feature_map(self, image_path: str, layer_name: nn.Module):
        img = self.read_img(image_path)
        trans = self.preprocess()
        crop_img = trans(img)
        input_tensor = crop_img.unsqueeze(0)

        canvas_img = (crop_img * 255).byte().numpy().transpose(1, 2, 0)
        canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)
        canvas_img = np.float32(canvas_img) / 255
        with GradCAMPlusPlus(model=self.model, target_layers=[layer_name]) as cam:
            garyscale_cam = cam(input_tensor=input_tensor)[0, :]
        visualization_img = show_cam_on_image(canvas_img, garyscale_cam, use_rgb=False)
        cv2.imshow('feature_map', visualization_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    weight = r'C:\yolov8\runs\detect\train-VOC.yaml-CloFormer-XXS-ECA2\weights\best.pt'
    image = r'C:\dataset\VOC\images\test2007\000076.jpg'
    v = Visualize(weight)
    model = v.load_model().model
    v.visualize_feature_map(image, model[10])
