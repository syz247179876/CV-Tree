"""
General Workflow
"""
import os

import torch.nn as nn
import argparse
from utils import check_inference_result, export_onnx, export_engine, infer_on_engine, \
    onnx_storage, pt_storage, trt_storage

analysis_storage = rf'{os.path.dirname(__file__)}\analysis'


class Args(object):
    """
    General Args
    """

    def __init__(self):

        self.parser = argparse.ArgumentParser()


        self.parser.add_argument('--dtype', type=str, default='fp32', help='data and model type')
        self.parser.add_argument('--shape', type=str, default='16,3,640,640',
                                 help='The maximum shape used for inference in dynamic tensorrt export')
        self.parser.add_argument('--test_image', type=str, default=None, help='test image path')
        self.parser.add_argument('--inference_shapes', type=str,
                                 default='1,3,224,224 1,3,480,480 1,3,640,640 1,3,1024,1024',
                                 help='different shapes used in inference')
        # pytorch use
        self.parser.add_argument('--pt_name', type=str, default='yolov8-EfficientViT-M5.pt', help='pytorch model name')

        # onnx use
        self.parser.add_argument('--onnx_name', type=str, default='yolov8-EfficientViT-M5.onnx', help='onnx file name')
        self.parser.add_argument('--input_names', type=str, default='images', help='onnx input_names')
        self.parser.add_argument('--output_names', type=str, default='output0', help='onnx output_names')
        self.parser.add_argument('--dynamic', action='store_true', default=False, help='whether enable dynamic shape')
        self.parser.add_argument('--task', type=str, default='detection', help='downstream task')
        self.parser.add_argument('--simplify', action='store_true', default=False, help='whether simplify onnx model')
        self.parser.add_argument('--opset_version', type=int, default=None, help='onnx opset version')
        self.parser.add_argument('--verbose', action='store_true', default=False, help='whether enable verbose logger')
        self.parser.add_argument('--device', type=str, default='cuda', help='device to use')
        self.parser.add_argument('--fuse', action='store_true', default=False, help='whether enable fuse')

        # engine use
        self.parser.add_argument('--engine_name', type=str, default='yolov8-EfficientViT-M5.engine', help='engine file name')
        self.parser.add_argument('--workspace', type=int, default=4,
                                 help='The allowed graphics memory capacity for running TensorRT')

        # check
        self.parser.add_argument('--analysis_name', type=str, default='yolov8-EfficientViT-M5.txt', help='analysis file name')

        self.opts = self.parser.parse_args()
        self.adjust_args()
        # self.inspect_args()


    def adjust_args(self) -> None:
        """
        adjust some args
        """
        _shape = tuple([int(s) for s in self.opts.shape.split(',')])
        self.opts.shape = _shape

        _input_names = self.opts.input_names.split(',')
        self.opts.input_names = _input_names

        _output_names = self.opts.output_names.split(',')
        self.opts.output_names = _output_names

        _inference_shapes = [tuple(int(s) for s in shape.split(',')) for shape in self.opts.inference_shapes.split(' ')]
        self.opts.inference_shapes = _inference_shapes


    def inspect_args(self) -> None:
        """
        inspect some args
        """
        assert self.opts.pt_name != '' and os.path.exists(self.opts.pt_name), \
            f'pt file: {self.opts.pt_name} is not exist'
        assert self.opts.onnx_name != '' and os.path.exists(self.opts.opts.onnx_name), \
            f'onnx file: {self.opts.onnx_name} is not exist'
        assert self.opts.engine_name != '' and os.path.exists(self.opts.opts.engine_name),\
            f'engine file: {self.opts.engine_name} is not exist'
        assert self.opts.analysis_name != '' and os.path.exists(self.opts.opts.analysis_name), \
            f'analysis file: {self.opts.analysis_name} is not exist'


class WorkFlow(nn.Module):
    """
    General WorkFlow
    """

    def __init__(self, opts: argparse.Namespace):
        if not os.path.exists(onnx_storage):
            os.mkdir(onnx_storage)
        if not os.path.exists(pt_storage):
            os.mkdir(pt_storage)
        if not os.path.exists(trt_storage):
            os.mkdir(trt_storage)
        if not os.path.exists(analysis_storage):
            os.mkdir(analysis_storage)

        save_path = rf'{analysis_storage}\{opts.analysis_name}'
        super(WorkFlow, self).__init__()
        self.opts = opts
        self.workflow = [
            (export_onnx.__name__, {
                'pt_name': self.opts.pt_name,
                'onnx_name': self.opts.onnx_name,
                'simplify': self.opts.simplify,
                'verbose': self.opts.verbose,
                'fuse': self.opts.fuse,
                'input_names': self.opts.input_names,
                'output_names': self.opts.output_names,
                'dynamic': self.opts.dynamic,
                'device': self.opts.device,
                'dtype': 'fp32',
                'opset_version': self.opts.opset_version,
                'shape': (1, *self.opts.shape[1:])
            }),
            (export_engine.__name__, {
                'onnx_name': self.opts.onnx_name,
                'verbose': self.opts.verbose,
                'workspace': self.opts.workspace,
                'dtype': self.opts.dtype,
                'dynamic': self.opts.dynamic,
                'shape': self.opts.shape,
            }),
            (check_inference_result.__name__, {
                'pt_name': self.opts.pt_name,
                'onnx_name': self.opts.onnx_name,
                'engine_name': self.opts.engine_name,
                'analysis_name': self.opts.analysis_name,
                'image_path': self.opts.test_image,
            }),
        ]

        workflow_inference = []
        for shape in self.opts.inference_shapes:
            workflow_inference.extend([
                (infer_on_engine.__name__, {
                    'model': self.opts.pt_name,
                    'engine': 'pytorch',
                    'device': 'cpu',
                    'dtype': 'fp32',
                    'fuse': False,
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.pt_name,
                    'engine': 'pytorch',
                    'device': 'cpu',
                    'dtype': 'fp32',
                    'fuse': True,
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.pt_name,
                    'engine': 'pytorch',
                    'device': 'cuda',
                    'dtype': 'fp32',
                    'fuse': False,
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.pt_name,
                    'engine': 'pytorch',
                    'device': 'cuda',
                    'dtype': 'fp32',
                    'fuse': True,
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.pt_name,
                    'engine': 'pytorch',
                    'device': 'cuda',
                    'dtype': 'fp16',
                    'fuse': False,
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.pt_name,
                    'engine': 'pytorch',
                    'device': 'cuda',
                    'dtype': 'fp16',
                    'fuse': True,
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.onnx_name,
                    'engine': 'onnxruntime',
                    'device': 'cpu',
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.onnx_name,
                    'engine': 'onnxruntime',
                    'device': 'cuda',
                    'shape': shape
                }),
                (infer_on_engine.__name__, {
                    'model': self.opts.engine_name,
                    'engine': 'tensorrt',
                    'device': 'cuda',
                    'shape': shape
                }),
            ])
        self.workflow.extend(workflow_inference)

        # add extra param
        for flow_name, flow_args in self.workflow:
            flow_args['file_path'] = save_path


    def forward(self):
        """
        The workflow is executed in the following order of steps:
        1.trans pt to onnx/engine/...

        2.check onnx/engine/... result, compared with pt

        3.execute inference respectively

        note:
            log all above result of each model into ../inference/analysis/
        """


        for idx, (func_name, func_args) in enumerate(self.workflow):
            func = globals().get(func_name)
            func(**func_args)


if __name__ == '__main__':
    args = Args()
    workflow = WorkFlow(args.opts)
    workflow()