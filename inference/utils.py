import os

import onnx
import torch
import numpy as np
import onnxruntime as rt
import cv2
import typing as t


__all__ = ['check_onnx', 'export_onnx', ]

from ultralytics import YOLO

onnx_storage = rf'{os.path.dirname(__file__)}\onnx_storage'
pt_storage = rf'{os.path.dirname(__file__)}\pt_storage'
trt_storage = rf'{os.path.dirname(__file__)}\trt_storage'

def export_onnx(
        pt_model: str,
        onnx_name: str,
        input_names: str = 'input',
        output_names: str = 'output'
):
    """
    export model from pytorch model
    """
    _x = torch.randn((1, 3, 640, 640))
    ckpt = torch.load(pt_model, map_location='cpu')
    model = (ckpt.get('ema') or ckpt['model']).float()
    torch.onnx.export(
        model,
        _x,
        rf'{onnx_storage}\{onnx_name}',
        opset_version=11,
        input_names=[input_names],
        output_names=[output_names],
        # dynamic_axes={'input': {0: 'batch_size', 1: 'channel', 2: 'height', 3: 'width'}},
    )

def export_test():
    # model = YOLO(r'D:\projects\yolov8\ultralytics\inference\pt_storage\yolov8n.pt')
    # # model.export(format='tensorrt')
    # model.benchmark()
    from ultralytics.utils.benchmarks import benchmark

    benchmark(model=r'D:\projects\yolov8\ultralytics\inference\pt_storage\yolov8n.pt', imgsz=640, )


def check_onnx(
        pt_model: str,
        onnx_model: str,
        image_path: str,
        input_shape: t.Tuple[int] = (640, 640)
):
    """
    Verify the rationality of the onnx model and pytorch model
    """
    # image preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = np.array(img) / 255.
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

    # build onnx session
    sess = rt.InferenceSession(onnx_model)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    # onnx_model output
    onnx_out = sess.run([output_name], {input_name: img})[0]

    # load pytorch model
    ckpt = torch.load(pt_model, map_location='cpu')
    pt_model = (ckpt.get('ema') or ckpt['model']).float()
    pt_out = pt_model(torch.from_numpy(img).float())[0]

    print(torch.max(torch.abs(pt_out - onnx_out)))




if __name__ == '__main__':

    # export_onnx(
    #     pt_model=rf'{pt_storage}\yolov8n.pt',
    #     onnx_name='yolov8n.onnx',
    # )

    # check_onnx(
    #     pt_model=rf'{pt_storage}\yolov8n.pt',
    #     onnx_model=rf'{onnx_storage}\yolov8n-sim.onnx',
    #     image_path=r'D:\projects\yolov8\ultralytics\ultralytics\assets\bus.jpg'
    # )
    export_test()
