import os
import time

import onnx
import torch
import numpy as np
import onnxruntime as rt
import cv2
import typing as t
import torch.nn as nn

from ultralytics.utils import TQDM

__all__ = ['check_onnx', 'export_onnx', ]

onnx_storage = rf'{os.path.dirname(__file__)}\onnx_storage'
pt_storage = rf'{os.path.dirname(__file__)}\pt_storage'
trt_storage = rf'{os.path.dirname(__file__)}\trt_storage'

def get_latest_opset():
    """Return second-most (for maturity) recently supported ONNX opset by this version of torch."""
    return max(int(k[14:]) for k in vars(torch.onnx) if 'symbolic_opset' in k) - 1  # opset


def iterative_sigma_clipping(data, sigma=2, max_iters=3):
    """Applies an iterative sigma clipping algorithm to the given data times number of iterations."""
    data = np.array(data)
    for _ in range(max_iters):
        mean, std = np.mean(data), np.std(data)
        clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
        if len(clipped_data) == len(data):
            break
        data = clipped_data
    return data

def export_onnx(
        pt_model: str,
        onnx_name: str,
        input_names: t.Optional[t.List] = None,
        output_names: t.Optional[t.List] = None,
        dynamic: bool = True,
        task: str = 'detection',
        simplify: bool = True,
        open_log: bool = False,
):
    """
    export model from pytorch model, mainly used in detection task, if exist, return directly
    """
    onnx_f_name = rf'{onnx_storage}\{onnx_name}'
    # if exist, return
    if os.path.exists(onnx_f_name):
        model_onnx = onnx.load(onnx_f_name)
        return onnx_f_name, model_onnx

    if input_names is None:
        input_names = ['images']
    if output_names is None:
        output_names = ['output0']
    # allow dynamic input
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # (1, 116, 8400)
        if task == 'detection':
            dynamic['output0'] = {0: 'batch', 2: 'anchors'} # (1, 84, 8400)
    opset_version = get_latest_opset()

    _x = torch.randn((1, 3, 640, 640))
    ckpt = torch.load(pt_model, map_location='cpu')
    model = (ckpt.get('ema') or ckpt['model']).float()

    # export model
    torch.onnx.export(
        model.cpu() if dynamic else model,
        _x.cpu() if dynamic else _x,
        onnx_f_name,
        verbose=open_log,
        do_constant_folding=True,  # 常量折叠会把常量表达式的值求出来作为常量嵌在最终生成的代码中
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic or None
    )

    # load model
    model_onnx = onnx.load(onnx_f_name)
    if simplify:
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
        except Exception as e:
            print(f'simplifier failure: {e}')

    onnx.save(model_onnx, onnx_f_name)
    return onnx_f_name, model_onnx


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
    pt_model.eval()
    img = torch.from_numpy(img).float()
    pt_out = pt_model(img)[0]

    print(torch.max(torch.abs(pt_out - onnx_out)))


def batchmark_common(
        model: t.Union[t.Callable, nn.Module],
        shape: t.Tuple,
        device: str = 'cuda',
        dtype: str = 'fp32',
        warmup_n: int = 50,
        run_num: int = 200,
        fuse: bool = True
):
    """
    only test model inference with fake data on CPU or GPU based on Pytorch, without postprocess
    """
    if fuse:
        model = model.fuse()
    model = model.to(device)
    x = torch.rand(size=shape).to(device)
    if dtype == 'fp32':
        x = x.float()
        model = model.float()
    elif dtype == 'fp16':
        x = x.half()
        model = model.half()
    else:
        raise ValueError(f'no supported dtype {dtype}')
    model.eval()

    """
    Firstly, after the model is built, entering Warmup is a stable stage for the model, as it will take some time 
    to load data after construction. If Warmup is not applied, it may affect performance.
    """
    print('---start warmup')
    with torch.no_grad():
        for _ in range(warmup_n):
            model(x)
    torch.cuda.synchronize()

    print(f'---start testing shape {shape[2]}x{shape[3]} on {device}, fuse: {fuse}, dtype: {dtype}')
    run_times = []
    with torch.no_grad():
        for _ in TQDM(range(1, run_num + 1),
                      desc=f'Pytorch {shape[2]}x{shape[3]} {device} ({"fused" if fuse else "no-fused"}) {dtype}'):
            t1 = time.time()
            model(x)
            torch.cuda.synchronize()
            run_times.append((time.time() - t1) * 1000)
        run_times = iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
    print(f'Input shape: {x.size()}')
    # print(f'Output shape: {res.size()}')
    print(f'Average time of each batch: {np.mean(run_times)}ms')

    print(f'-------------------------------------------------------')
    print('')


def batchmark_onnx(
        model: t.Union[t.Callable, nn.Module],
        shape: t.Tuple,
        device: str = 'cuda',
        warmup_n: int = 50,
        run_num: int = 200,
):
    import onnxruntime as ort
    import psutil
    sess_options = ort.SessionOptions()
    # 启用所有计算图优化策略
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 设置运行模型时的最大线程数
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    # 控制计算图内部的算子计算是串行还是并行, 如果模型包含多个分支, 启用并行性能会提升一些
    # sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(model, sess_options,
                                providers=['CPUExecutionProvider' if device == 'cpu' else 'CUDAExecutionProvider'])
    input_tensor = sess.get_inputs()[0]
    input_type = input_tensor.type

    if 'float16' in input_type:
        input_dtype = np.float16
    elif 'float' in input_type:
        input_dtype = np.float32
    elif 'double' in input_type:
        input_dtype = np.float64
    elif 'int64' in input_type:
        input_dtype = np.int64
    elif 'int32' in input_type:
        input_dtype = np.int32
    else:
        raise ValueError(f'Unsupported ONNX datatype {input_type}')
    input_data = np.random.rand(*shape).astype(input_dtype)
    input_name = input_tensor.name
    output_name = sess.get_outputs()[0].name

    # Warmup runs
    print('---start warmup')
    for _ in range(warmup_n):
        sess.run([output_name], {input_name: input_data})

    print(f'---start testing shape {shape[2]}x{shape[3]} on {device}, fused, dtype: {input_dtype}')
    run_times = []
    for _ in TQDM(range(1, run_num + 1),
                  desc=f'onnxruntime {shape[2]}x{shape[3]} {device} fused {input_dtype}'):
        t1 = time.time()
        sess.run([output_name], {input_name: input_data})
        run_times.append((time.time() - t1) * 1000)

    run_times = iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
    print(f'Input shape: {shape}')
    print(f'Average time of each batch: {np.mean(run_times)}ms')

    print(f'-------------------------------------------------------')
    print('')




def infer_on_engine(
        weights: str,
        engine: str = 'pytorch',
        device: str ='cpu',
        dtype: str = 'fp32',
        fuse: bool = True,
        shapes: t.List[t.Tuple] = [(1, 3, 224, 224), (1, 3, 480, 480), (1, 3, 640, 640), (1, 3, 1024, 1024)],
):
    """
    inference on different engine
    """
    print(f'---------------{weights.split("/")[-1]}-----------------')
    model = None
    if engine == 'pytorch':
        ckpt = torch.load(weights, map_location='cpu')
        model = (ckpt.get('ema') or ckpt['model']).float()
    elif engine == 'onnxruntime':
        model = weights

    for shape in shapes:
        if engine == 'pytorch':
            batchmark_common(
                model=model,
                shape=shape,
                device=device,
                dtype=dtype,
                fuse=fuse
            )
        elif engine == 'onnxruntime':
            batchmark_onnx(
                model=model,
                device=device,
                shape=shape,
            )
        elif engine == 'tensorrt':
            batchmark_onnx(
                model=model,
                device=device,
                shape=shape,
            )
    print(f'-------------------------------------------------------')
    print(f'-------------------------------------------------------')
    print('')


if __name__ == '__main__':

    # export_onnx(
    #     pt_model=r'C:\yolov8\runs\detect\train-ExDark.yaml-EfficientVit-M5-backbone-optimize\weights\best.pt',
    #     onnx_name='yolov8-EfficientViT-M5-original.onnx',
    #     simplify=False,
    #     open_log=True
    # )
    # export_onnx(
    #     pt_model=r'C:\yolov8\runs\detect\train-ExDark.yaml-EfficientVit-M5-backbone-optimize\weights\best.pt',
    #     onnx_name='yolov8-EfficientViT-M5-sim.onnx',
    #     simplify=True,
    #     open_log=True
    # )

    # check_onnx(
    #     pt_model=rf'C:\yolov8\runs\detect\train-ExDark.yaml-EfficientVit-M5-backbone-optimize\weights\best.pt',
    #     onnx_model=rf'{onnx_storage}\yolov8-EfficientViT-M5-sim.onnx',
    #     image_path=r'D:\projects\yolov8\ultralytics\ultralytics\assets\bus.jpg'
    # )
    # check_onnx(
    #     pt_model=rf'C:\yolov8\runs\detect\train-ExDark.yaml-EfficientVit-M5-backbone-optimize\weights\best.pt',
    #     onnx_model=rf'{onnx_storage}\yolov8-EfficientViT-M5-original.onnx',
    #     image_path=r'D:\projects\yolov8\ultralytics\ultralytics\assets\bus.jpg'
    # )

    weights_pt = r'C:\yolov8\runs\detect\train-ExDark.yaml-EfficientVit-M5-backbone-optimize\weights\best.pt'
    # Pytorch, cpu or cuda

    infer_on_engine(weights_pt, engine='pytorch', device='cpu', dtype='fp32', fuse=True)
    infer_on_engine(weights_pt, engine='pytorch', device='cpu', dtype='fp32', fuse=False)

    infer_on_engine(weights_pt, engine='pytorch', device='cuda', dtype='fp32', fuse=True)
    infer_on_engine(weights_pt, engine='pytorch', device='cuda', dtype='fp16', fuse=False)
    infer_on_engine(weights_pt, engine='pytorch', device='cuda', dtype='fp32', fuse=True)
    infer_on_engine(weights_pt, engine='pytorch', device='cuda', dtype='fp16', fuse=True)

    weight_onnx_original = rf'{onnx_storage}\yolov8-EfficientViT-M5-original.onnx'
    weight_onnx_sim = rf'{onnx_storage}\yolov8-EfficientViT-M5-sim.onnx'
    # ONNX, cpu or cuda, in ONNX, fuse is already implemented

    infer_on_engine(weight_onnx_original, engine='onnxruntime', device='cpu')
    infer_on_engine(weight_onnx_original, engine='onnxruntime', device='cuda')

    infer_on_engine(weight_onnx_sim, engine='onnxruntime', device='cpu')
    infer_on_engine(weight_onnx_sim, engine='onnxruntime', device='cuda')


    # TensorRT, cuda




