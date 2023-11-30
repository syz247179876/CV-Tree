import json
import os
import time
import typing as t
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
from tensorrt.tensorrt import BuilderFlag

from ultralytics.utils import ops

from inference.base_infer import BaseInference, Args
from inference.utils import trt_storage, onnx_storage

pt_storage = rf'{os.path.dirname(__file__)}\trt_storage'
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

cuda.init()
cfx = cuda.Device(0).make_context()
cfx.pop()
def trt_version():
    """
    get TensorRT version
    Returns: str
    """
    return trt.__version__

def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def torch_dtype_from_trt(dtype):
   if dtype == trt.int8:
       return torch.int8
   elif trt_version() >= '7.0' and dtype == trt.bool:
       return torch.bool
   elif dtype == trt.int32:
       return torch.int32
   elif dtype == trt.float16:
       return torch.float16
   elif dtype == trt.float32:
       return torch.float32
   else:
       raise TypeError("%s is not supported by torch" % dtype)

def build_engine(onnx_file: str, trt_file: str, save_trt: bool = True):
    """
    build engine based on onnx_file and store engine to trt file
    """

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        # allow TensorRT to use up to 1GB of GPU memory for tactic selection
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        # config.max_workspace_size  = 1 << 30

        # inference one picture for each time
        # builder.max_batch_size = 1

        # use FP16 mode if possible
        if builder.platform_has_fast_fp16:
            config.flags = 1 << int(BuilderFlag.FP16)

        # load onnx model
        assert os.path.exists(onnx_file), f'ONNX file {onnx_file} not found, plz generate it by using export_onnx() in ./utils.py'
        with open(onnx_file, 'rb') as onnx_f:
            print('Loading ONNX file from path {}...'.format(onnx_file))
            parser.parse(onnx_f.read())
        for error in range(parser.num_errors):
            print(parser.get_error(error)) # 打印错误（如果解析失败，根据打印的错误进行Debug）
            raise ValueError('parser onnx model failed')
        print('Completed parsing of ONNX file')
        print('Building an engine')

        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)
        # create engine use builder
        engine = builder.build_serialized_network(network, config)
        print('Completed creating Engine')

        if save_trt:
            with open(trt_file, 'wb') as f:
                f.write(engine)
            # 反序列化加载模型
        f = open(trt_file, "rb")  # 打开保存的序列化模型
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())  # 反序列化加载模
        return engine

class TRTModule(nn.Module):
    """
    load trt model and create runtime, and then inference
    """
    def __init__(
            self,
            engine: trt.tensorrt.ICudaEngine,
            input_names: t.List,
            output_names: t.List,
            device: str,
            classes_n: int
    ):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # use engine to create context
            self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_names = output_names
        self.device = device
        self.classes_n = classes_n


    def forward2(self, x: torch.Tensor):
        """
        only have one output and one input
        """
        batch_size = x.shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        # 创建输出tensor，并分配内存
        outputs = [None] * len(self.output_names)

        # 根据输入和输出分配指定大小的内存, 并绑定所使用的内存地址
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)  # 通过binding_name找到对应的input_id
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))  # 找到对应的数据类型
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))  # 找到对应的形状大小
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[1] = output.data_ptr()  # 绑定输出数据指针

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[0] = x.data_ptr()  # 绑定输入数据的内存地址(指针)
        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )  # 执行推理
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs.squeeze(dim=0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        bindings = []
        for idx, binding in enumerate(self.engine):
            # allocate mem to all input and output
            if self.engine.binding_is_input(binding):
                input_shape = self.engine.get_binding_shape(binding)
                input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
                # allocate GPU mem to input
                device_input = cuda.mem_alloc(input_size)
                bindings.append(int(device_input))
            # add one output
            else:
                output_shape = self.engine.get_binding_shape(binding)
                # allocate page-locked memory buffers to output(i.e. won't be swapped to disk)
                host_output = cuda.pagelocked_empty(
                    trt.volume(output_shape), dtype=np.float32
                )
                device_output = cuda.mem_alloc(host_output.nbytes)
                bindings.append(int(device_output))

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        host_input = np.ascontiguousarray(x, dtype=np.float32)
        # copy data from host to device (cpu to gpu)
        cuda.memcpy_htod_async(device_input, host_input, stream)

        # run inference
        cfx.push()
        # self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        execute = self.context.execute_v2(bindings=bindings)
        cfx.pop()

        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        # waiting for asynchronous task execution to end
        # stream.synchronize()
        output = torch.from_numpy(host_output).reshape((1, self.classes_n + 4, -1))
        return output


class TensorRTInference(BaseInference):

    def __init__(
            self,
            model_path: str,
            conf_thres: float = 0.3,
            iou_thres: float = 0.45,
            data_classes: t.Union[t.Dict, str] = 'coco8.yaml',
            max_det: int = 1000,
            img_size: t.Tuple = (640, 640),
            half: bool = True,
            fuse: bool = True,
            use_gpu: bool = True,
            onnx_path: t.Optional[str] = None
    ):
        super().__init__(model_path, conf_thres, iou_thres, data_classes, max_det, img_size, half, fuse,
                         use_gpu)
        self.onnx_path = onnx_path
        self.cfx = cuda.Device(0).make_context()


    def _init(self):
        """
        load model from trt model
        Returns: trt model
        """
        logger = trt.Logger(trt.Logger.INFO)
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f, trt.Runtime(logger) as runtime:
                # create engine/model
                engine = runtime.deserialize_cuda_engine(f.read())
        else:
            assert self.onnx_path, f'ONNX model path {self.onnx_path} is not exist'
            engine = build_engine(self.onnx_path, self.model_path, save_trt=True)
        self.engine = engine
        trt_model = TRTModule(engine, ['input'], ['output'], self.device, len(self.classes))
        return trt_model

    def preprocess(self, image_path: str, *args, **kwargs) -> t.Union[torch.Tensor, np.ndarray]:
        image_data = super(TensorRTInference, self).preprocess(image_path)
        # image_data = torch.from_numpy(image_data).to(self.device)
        return image_data

    def main(self, image_path: t.Union[str, t.List]) -> t.List:
        """
        Performs inference using a different model or inference engine and returns the dict of info of images

        Returns:
                a list, stores the coordinates, confidence, and category of the target box for each predicted image
        """

        res = []
        if isinstance(image_path, str):
            image_path = [image_path]
        for path in image_path:
            image = self.preprocess(path)
            pred = self.model(image)
            res.append(self.postprocess(resize_shape=image.shape[2:],
                                        origin_shape=(self.img_height, self.img_width), pred=pred))
        return res


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

def inspect_trt(model_path: str):
    """
    inspect TensorRT model
    Args:
        model_path:

    Returns:

    """
    model_all_names = []
    logger = trt.Logger(trt.Logger.INFO)
    with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        model_all_names.append(name)
        shape = engine.get_binding_shape(idx)
        print('input id:', idx, ' is input: ', is_input, ' binding name:', name, ' shape:', shape, 'type: ', op_type)

if __name__ == '__main__':
    args = Args()
    args.set_args()
    args.opts.model = r'D:\projects\yolov8\ultralytics\inference\trt_storage\yolov8n-fp163.trt'
    print(args.opts)
    images_path = os.listdir(args.opts.images_dir)
    images_path = [os.path.join(args.opts.images_dir, i) for i in images_path]
    obj = TensorRTInference(
        args.opts.model,
        conf_thres=args.opts.conf_thres,
        iou_thres=args.opts.iou_thres,
        data_classes=args.opts.data,
    )
    t1 = time.time()
    res = obj.main(images_path)
    # print(res)
    t2 = time.time()
    print(t2 - t1)

    # cfx.pop()
    # build_engine(
    #     onnx_file=rf'{onnx_storage}\yolov8n.onnx',
    #     trt_file=rf'{trt_storage}\yolov8n-fp162.trt',
    #     save_trt=True
    # )

    # inspect_trt(rf'{trt_storage}\yolov8n-fp16-sim(1).trt')
