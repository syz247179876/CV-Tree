import os
import time
import warnings
from functools import wraps

import onnx
import torch
import numpy as np
import onnxruntime as ort
import psutil
import cv2
import typing as t
import torch.nn as nn

from collections import OrderedDict, namedtuple
from wraps import log_wrap
from ultralytics.utils import TQDM

__all__ = ['check_inference_result', 'export_onnx', 'export_engine', 'infer_on_engine']

onnx_storage = rf'{os.path.dirname(__file__)}\onnx_storage'
pt_storage = rf'{os.path.dirname(__file__)}\pt_storage'
trt_storage = rf'{os.path.dirname(__file__)}\trt_storage'


def from_numpy(x: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert a numpy array to a tensor.
    """
    return torch.tensor(x).to(device) if isinstance(x, np.ndarray) else x


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

@log_wrap(stream='file')
def export_onnx(
        onnx_name: str,
        pt_name: t.Optional[str] = None,
        input_names: t.Optional[t.List] = None,
        output_names: t.Optional[t.List] = None,
        dynamic: bool = True,
        task: str = 'detection',
        simplify: bool = True,
        device: str = 'cuda',
        fuse: bool = True,
        dtype: str = 'fp32',
        verbose: bool = False,
        opset_version: t.Optional[int] = None,
        shape: t.Tuple[int] = None,
) -> t.Tuple[str, t.List, t.List]:
    """
    export onnx from pytorch model, mainly used in detection task, if exist, return directly

    return:
        export file path, pre info, post info
    """
    onnx_path = rf'{onnx_storage}\{onnx_name}'
    # if exist, return
    if os.path.exists(onnx_path):
        return onnx_path, [], []

    assert pt_name is not None, 'pt_name should be str value'
    pt_path = rf'{pt_storage}\{pt_name}'
    if input_names is None:
        input_names = ['images']
    if output_names is None:
        output_names = ['output0']

    # allow dynamic input
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # (1, 116, 8400)
        if task == 'detection':
            dynamic['output0'] = {0: 'batch', 2: 'anchors'} # (1, 84, 8400)
    opset_version = opset_version or get_latest_opset()

    if shape is None:
        shape = (1, 3, 640, 640)
    x_ = torch.zeros(shape).to(device)

    ckpt = torch.load(pt_path, map_location='cpu')
    model = (ckpt.get('ema') or ckpt['model']).float().to(device)

    # Filter warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    warnings.filterwarnings('ignore', category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
    warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

    model.eval()

    if fuse:
        model = model.fuse()
    if dtype == 'fp16' and (device == 'cuda'):
        x_, model = x_.half(), model.half()

    # export model
    torch.onnx.export(
        model.cpu() if dynamic else model,
        x_.cpu() if dynamic else x_,
        onnx_path,
        verbose=verbose,
        do_constant_folding=True,  # 常量折叠会把常量表达式的值求出来作为常量嵌在最终生成的代码中
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic or None
    )

    # load model
    model_onnx = onnx.load(onnx_path)
    if simplify:
        try:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
        except Exception as e:
            print(f'simplifier failure: {e}')

    onnx.save(model_onnx, onnx_path)
    return onnx_path, [
        'starting export onnx model...'
    ], [
        'export onnx model successfully'
    ]

@log_wrap(stream='file')
def export_engine(
        onnx_name: str,
        task: str = 'detection',
        verbose: bool = False,
        workspace: int = 4,
        dtype: str = 'fp32',
        dynamic: bool = False,
        shape: t.Tuple[int] = None,
) -> t.Tuple[str, t.List, t.List]:
    """
    export TensorRT engine based on onnx-sim, builder period
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    if shape is None:
        shape = (16, 3, 640, 640)
    f_onnx, _, _ = export_onnx(onnx_name=onnx_name)
    assert os.path.exists(f_onnx), f'failed to export ONNX file: {f_onnx}'
    onnx_name = f_onnx.rsplit('\\')[-1]
    engine_name = onnx_name.replace('.onnx', '.engine')
    f = rf'{trt_storage}\{engine_name}'
    if verbose:
        # 包含优化过程相关信息
        logger.min_severity = trt.Logger.Severity.VERBOSE

    # 构建builder, 包含计算图的属性信息
    builder = trt.Builder(logger)
    # 负责设置模型的一些参数, 如运行的显存容量, 推理使用的数据类型等
    config = builder.create_builder_config()
    # TensorRT运行优化所需的最大内存
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4

    # 使用Explict Batch模式, 在构建中显示指定Batch大小
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # create network, 包含计算图的具体内容, 用于标记网络输入/输出张量和添加/删除层，以及onnx的中间导入
    network = builder.create_network(flag)
    # onnx parser, 解析onnx中间格式文件
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(f_onnx):
        raise RuntimeError(f'failed to load ONNX file: {f_onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        print(f'TensorRT input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'TensorRT output "{out.name}" with shape{out.shape} {out.dtype}')

    # 指定输出张量大小, 需要network使用Explicit Batch模式
    if dynamic:
        profile = builder.create_optimization_profile()
        for inp in inputs:
            # 分别为最小shape, 最常见shape, 和最大shape, 这里固定640x640, 只对batch做dynamic
            profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape)
        config.add_optimization_profile(profile)

    # 根据dtype和当前硬件判断是否开启半精度, 尝试插入Reformat节点
    if dtype == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    torch.cuda.empty_cache()

    # serialize network
    with builder.build_serialized_network(network, config) as engine_string, open(f, 'wb') as engine_f:
        engine_f.write(engine_string)

    return f, [
        'starting export TensorRT engine...',
    ],[
        'export TensorRT successfully'
    ]

@log_wrap(stream='file')
def check_inference_result(
        pt_name: str,
        onnx_name: str,
        engine_name: str,
        analysis_name: str,
        image_path: t.Optional[str] = None
) -> t.Tuple[t.List, t.List]:
    """
    Verify the rationality of the onnx model, tensorrt engine and pytorch model, test on cuda
    """
    # image preprocess
    device = 'cuda'

    pa_path = rf'{pt_storage}\{pt_name}'
    onnx_path = rf'{onnx_storage}\{onnx_name}'
    engine_path = rf'{trt_storage}\{engine_name}'

    model_names = [
        pt_name,
        onnx_name,
        engine_name,
    ]
    shape = (1, 3, 640, 640)
    if image_path is not None and os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, shape[2: ])
        img = np.array(img) / 255.
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
    else:
        img = np.random.randn(*shape).astype(np.float32)
    img_torch = torch.from_numpy(img).to(device)

    # test pytorch session
    pt_res = batchmark_common(pa_path, shape, device=device, warmup_n=3, fuse=True, inspect=True, test_data=img_torch)[0]

    # test onnx session
    onnx_res = batchmark_onnx(onnx_path, shape, device=device, warmup_n=3, inspect=True, test_data=img)

    # test tensorrt session
    engine_res = batchmark_engine(engine_path, shape, device=device, warmup_n=3, inspect=True, test_data=img_torch)[-1]

    return [
        'Compare the running results of the following engines to see if they are similar to those of Pytorch',
        'starting compare...'
    ],[
        'compared result:',
        f'[pytorch -- onnx] max diff: {torch.max(torch.abs(pt_res - onnx_res))}',
        f'[pytorch -- tensorrt] max diff: {torch.max(torch.abs(pt_res - engine_res))}',
        'compared finished'
    ]

@log_wrap(stream='file')
def batchmark_common(
        model: str,
        shape: t.Tuple,
        device: str = 'cuda',
        dtype: str = 'fp32',
        warmup_n: int = 50,
        run_num: int = 200,
        fuse: bool = True,
        inspect: bool = False,
        test_data: t.Optional[t.Union[torch.Tensor, np.ndarray]] = None
) -> t.Union[torch.Tensor, t.Tuple[t.List, t.List]]:
    """
    测试 Pytorch 推理性能
    only test model inference with fake data on CPU or GPU based on Pytorch, without postprocess
    """
    ckpt = torch.load(model, map_location='cpu')
    model = (ckpt.get('ema') or ckpt['model']).float()
    if fuse:
        model = model.fuse()
    model = model.to(device)

    if test_data is not None:
        x = test_data
    else:
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

    with torch.no_grad():
        for _ in range(warmup_n):
            model(x)
    torch.cuda.synchronize()

    run_times = []
    with torch.no_grad():
        for _ in TQDM(range(1, run_num + 1),
                      desc=f'Pytorch {shape[2]}x{shape[3]} {device} ({"fused" if fuse else "no-fused"}) {dtype}'):
            t1 = time.time()
            res = model(x)
            torch.cuda.synchronize()
            run_times.append((time.time() - t1) * 1000)
            if isinstance(res, (list, tuple)):
                res = from_numpy(res[0], device) if len(res) == 1 else [from_numpy(x, device) for x in res]
            else:
                res = from_numpy(res, device)
        if inspect:
            return res
        run_times = iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping

    return [
        'starting batchmark Pytorch...',
        f'---test shape {shape[2]}x{shape[3]} on {device}, fuse: {fuse}, dtype: {dtype}'
    ],[
        f'input shape: {shape}',
        f'average time of each batch: {np.mean(run_times)}ms\n'
        'batchmark Pytorch successfully',
    ]

@log_wrap(stream='file')
def batchmark_onnx(
        model: str,
        shape: t.Tuple,
        device: str = 'cuda',
        warmup_n: int = 50,
        run_num: int = 200,
        inspect: bool = False,
        test_data: t.Optional[t.Union[torch.Tensor, np.ndarray]] = None
) -> t.Union[torch.Tensor, t.Tuple[t.List, t.List]]:
    """
    测试 onnxruntime 推理性能

    """
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
    if test_data is not None:
        input_data = test_data
    else:
        input_data = np.random.rand(*shape).astype(input_dtype)
    input_name = input_tensor.name
    output_name = sess.get_outputs()[0].name

    # Warmup runs
    for _ in range(warmup_n):
        sess.run([output_name], {input_name: input_data})

    # start actual inference
    run_times = []
    for _ in TQDM(range(1, run_num + 1),
                  desc=f'onnxruntime {shape[2]}x{shape[3]} {device} fused {input_dtype}'):
        t1 = time.time()
        res = sess.run([output_name], {input_name: input_data})
        run_times.append((time.time() - t1) * 1000)
        if isinstance(res, (list, tuple)):
            res = from_numpy(res[0], device) if len(res) == 1 else [from_numpy(x, device) for x in res]
        else:
            res = from_numpy(res, device)

        if inspect:
            return res

    run_times = iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
    return [
       'starting batchmark ONNX...',
       f'---test shape {shape[2]}x{shape[3]} on {device}, fused, dtype: {input_dtype}'
    ], [
       f'input shape: {shape}',
       f'average time of each batch: {np.mean(run_times)}ms\n'
       'batchmark ONNX successfully',
    ]

@log_wrap(stream='file')
def batchmark_engine(
    model: str,
    shape: t.Tuple,
    dtype: str = 'fp32',
    device: str = 'cuda',
    warmup_n: int = 50,
    run_num: int = 200,
    batch_idx: int = 0,
    inspect: bool = False,
    test_data: t.Optional[t.Union[torch.Tensor, np.ndarray]] = None
) -> t.Union[torch.Tensor, t.Tuple[t.List, t.List]]:
    """
    inference based on TensorRT

    note:
        batch_idx: different dynamic batches set when exporting the engine, batch_idx means the index of selected batch

    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open(model, 'rb') as f:
        engine_string = f.read()
    # 反序列化读取模型, 生成engine
    engine = runtime.deserialize_cuda_engine(engine_string)

    # create context, 保存GPU运行期间的环境, 使用engine或context binding输入/输出张量
    context = engine.create_execution_context()
    fp16 = False
    dynamic = False
    output_names = []
    bindings = OrderedDict()
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))

    # 现根据engine中的配置设置输入/输出张量, 分配CPU内存和GPU显存, 同时后面会根据实际输出 + dynamic 进行 shape动态调整
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        _dtype = trt.nptype(engine.get_binding_dtype(i))
        if _dtype != dtype:
            dtype = _dtype
        # 处理input
        if engine.binding_is_input(i):
            # engine是否采用fp16
            if _dtype == np.float16:
                fp16 = True
            # engine是否使用dynamic shape
            # 如果任意维度为-1, 该shape即为 dynamic shape
            if -1 in tuple(context.get_binding_shape(i)):
                dynamic = True
                context.set_binding_shape(i, tuple(engine.get_profile_shape(0, i)[batch_idx]))
        else:
            # 处理output
            output_names.append(name)

        _shape = tuple(context.get_binding_shape(i))
        # 给输入/输出分配CPU端内存和GPU端显存, 并将分配的空间从CPU移动到GPU
        im = torch.from_numpy(np.empty(_shape, dtype=_dtype)).to(device)
        bindings[name] = Binding(name, _dtype, _shape, im, int(im.data_ptr()))

    # 保存输出/输出开辟的内存块的首地址
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

    # if dynamic, batch_size may not equal to max_batch_size
    batch_size = bindings['images'].shape[0]

    if test_data is not None:
        fake_img = test_data
    else:
        fake_img = torch.randn(shape, device=device, dtype=torch.float32)

    # Warmup runs
    print('---start warmup')
    for _ in range(warmup_n):
        if fp16 and fake_img.dtype != torch.float16:
            fake_img = fake_img.half()

        # 根据实际图片 + dynamic属性 进行shape动态调整, 包括修改context的shape和bindings中对应的shape
        if dynamic and fake_img.shape != bindings['images'].shape:
            i = engine.get_binding_index('images')
            context.set_binding_shape(i, fake_img.shape)
            # 不够pythonic操作
            bindings['images'] = bindings['images']._replace(shape=fake_img.shape)
            for name in output_names:
                i = engine.get_binding_index(name)
                bindings[name].data.resize_(tuple(context.get_binding_shape(i)))

        # 因为image尺寸和原先配置的engine尺寸不一致, 数据所在地址肯定不一致了, 因此需要重新设置地址指针
        binding_addrs['images'] = int(fake_img.data_ptr())
        # 同步推理
        context.execute_v2(list(binding_addrs.values()))


    # start actual inference
    run_times = []
    print(f'---start testing shape {shape[2]}x{shape[3]} on {device}, fused, dtype: {dtype}')
    for _ in TQDM(range(1, run_num + 1),
                  desc=f'onnxruntime {shape[2]}x{shape[3]} {device} fused {dtype}'):
        if fp16 and fake_img.dtype != torch.float16:
            fake_img = fake_img.half()

        # 根据实际图片 + dynamic属性 进行shape动态调整, 包括修改context的shape和bindings中对应的shape
        if dynamic and fake_img.shape != bindings['images'].shape:
            i = engine.get_binding_index('images')
            context.set_binding_shape(i, fake_img.shape)
            # 不够pythonic操作
            bindings['images'] = bindings['images']._replace(shape=fake_img.shape)
            for name in output_names:
                i = engine.get_binding_index(name)
                bindings[name].data.resize_(tuple(context.get_binding_shape(i)))

        # 因为image尺寸和原先配置的engine尺寸不一致, 数据所在地址肯定不一致了, 因此需要重新设置地址指针
        binding_addrs['images'] = int(fake_img.data_ptr())
        # 同步推理
        t1 = time.time()

        context.execute_v2(list(binding_addrs.values()))
        run_times.append((time.time() - t1) * 1000)
        y = [bindings[x].data for x in sorted(output_names)]

        # 推理完, 需要将数据从GPU移回CPU/GPU, 具体取决于函数形参device
        if isinstance(y, (list, tuple)):
            res = (y[0], device) if len(y) == 1 else [from_numpy(x, device) for x in y]
        else:
            res = from_numpy(y, device)

        if inspect:
            # 用于check 不同engine下的结果
            return res

    run_times = iterative_sigma_clipping(np.array(run_times), sigma=2, max_iters=3)  # sigma clipping
    return [
       'starting batchmark TensorRT...',
       f'---test shape {shape[2]}x{shape[3]} on {device}, fused, dtype: {dtype}'
    ], [
       f'input shape: {shape}',
       f'average time of each batch: {np.mean(run_times)}ms\n'
       'batchmark TensorRT successfully',
    ]

@log_wrap(stream='file')
def infer_on_engine(
        model: str,
        engine: str = 'pytorch',
        device: str ='cpu',
        dtype: str = 'fp32',
        fuse: bool = True,
        shape: t.Tuple = (1, 3, 224, 224)
) -> t.Tuple[t.List, t.List]:
    """
    inference on different engine
    """
    res = None
    if engine == 'pytorch':
        model = rf'{pt_storage}\{model}'
        res = batchmark_common(
            model=model,
            shape=shape,
            device=device,
            dtype=dtype,
            fuse=fuse
        )
    elif engine == 'onnxruntime':
        model = rf'{onnx_storage}\{model}'
        res = batchmark_onnx(
            model=model,
            device=device,
            shape=shape,
        )
    elif engine == 'tensorrt':
        model = rf'{trt_storage}\{model}'
        res = batchmark_engine(
            model=model,
            device=device,
            shape=shape,
            dtype=dtype,
        )

    return res
