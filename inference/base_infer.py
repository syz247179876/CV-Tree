import os

import cv2
import numpy as np
import torch
import typing as t
from ultralytics.utils import ASSETS, yaml_load

__all__ = ['BaseInference', ]
class BaseInference(object):

    """
    Inference Base Class
    """

    def __init__(
            self,
            model_path: str,
            image_path: str,
            conf_thres: float,
            iou_thres: float,
            data_classes: t.Union[t.Dict, str],
            max_det: int = 1000,
            img_size: t.Tuple = (640, 640),
            half: bool = False,
            fuse: bool = False,
            use_gpu: bool = False,
    ):
        self.model_path = model_path
        self.image_path = image_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_width, self.input_height = img_size
        self.max_det = max_det
        self.half = half
        self.fuse = fuse

        if isinstance(data_classes, str):
            self.classes = yaml_load(data_classes)['names']
        else:
            self.classes = data_classes

        if torch.cuda.is_available() and use_gpu:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.model = self._init()

    def _init(self):
        """
        load model from model_path
        Returns: model
        """
        pass

    def preprocess(self, *args, **kwargs) -> t.Union[torch.Tensor, np.ndarray]:
        """
        Preprocesses the input image before performing inference.
        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.image_path)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data


    def main(self):
        """
         Performs inference using a different model or inference engine and returns the dict of output image

        Returns:
            output_img: The output image with drawn detections.

        """
        pass


    def postprocess(self, *args, **kwargs):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
        Returns: dict
        """
        pass
