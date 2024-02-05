from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, RepConv, Bottleneck, CPCA
import typing as t
import torch.nn as nn
import torch


__all__ = ['prune', 'prune_conv']


def prune_conv(
        conv1: t.Union[Conv, RepConv],
        conv2: t.Optional[t.Union[t.List, Conv, RepConv]],
        threshold: float
):
    """
    基于conv1中的BN的scale factor进行剪枝, 因为剪的是filter个数, 因此通道数会发生改变
    因此需要级联调整其后面卷积的输入通道

    """

    gamma = conv1.bn.weight.data.detach()
    beta = conv1.bn.bias.data.detach()
    # 记录保留的gamma索引
    keep_idxs = []
    while len(keep_idxs) < 8:
        keep_idxs = torch.where(gamma.abs() >= threshold)[0]
        threshold = threshold * 0.5
    n = len(keep_idxs)

    # 筛选保存下来的特征个数和权重
    conv1.bn.weight.data = gamma[keep_idxs]
    conv1.bn.bias.data = beta[keep_idxs]
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
    conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
    conv1.bn.num_features = n
    # 只改输出通道
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
    conv1.conv.out_channels = n

    if conv1.conv.bias is not None:
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

    # 修改后续卷积的输入通道
    if not isinstance(conv2, t.List):
        conv2 = [conv2]

    for item in conv2:
        if item is not None:
            if isinstance(item, (RepConv, Conv)):
                conv = item.conv
            else:
                conv = item
            conv.in_channels = n
            # 只改输出输入通道, 不变输出通道
            conv.weight.data = conv.weight.data[:, keep_idxs]

def prune(m1: nn.Module, m2: t.Union[t.List[nn.Module], None], threshold: float):
    """
    剪枝模块中的子模块（Conv, RepConv)
    处理1---1,  1---N的模块结构
    """
    if isinstance(m1, C2f):
        m1 = m1.cv2

    if not isinstance(m2, t.List):
        m2 = [m2]

    for i, item in enumerate(m2):
        if isinstance(item, C2f) or isinstance(item, SPPF):
            m2[i] = item.cv1
        elif isinstance(item, CPCA):
            m2[i] = item.ca.fc1
    prune_conv(m1, m2, threshold=threshold)


if __name__ == '__main__':
    # yolo = torch.load(r'C:\yolov8\runs\detect\train-VOC.yaml-baseline-v8n2\weights\last.pt', map_location='cpu')
    yolo = YOLO(r'C:\yolov8\runs\detect\train-VOC.yaml-baseline-v8n2\weights\last.pt')
    model = yolo.model

    ws = []
    bs = []

    # 计算剪枝阈值
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            w = m.weight.abs().detach()
            b = m.bias.abs().detach()
            ws.append(w)
            bs.append(b)

    factor = 0.8
    ws = torch.cat(ws)
    threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
    print(model)

    # 剪枝模型融合层的Bottleneck中的Conv
    for name, m in model.named_modules():
        if isinstance(m, Bottleneck):
            prune_conv(m.cv1, m.cv2, threshold)

    # 剪枝backbone中的Conv下采样层
    seq = model.model
    for i in range(3, 9):
        if i in [6, 4, 9]:
            continue
        prune(seq[i], seq[i + 1], threshold)

    # 剪枝neck中Detect, 作为Detect输入的C2f, 下采样
    detect = seq[-1]
    last_inputs = [seq[15], seq[18], seq[21]]
    downsample = [seq[16], seq[19], None]

    for last_input, ds, cv2, cv3 in zip(last_inputs, downsample, detect.cv2, detect.cv3):
        prune(last_input, [ds, cv2[0], cv3[0]], threshold)
        prune(cv2[0], cv2[1], threshold)
        prune(cv2[1], cv2[2], threshold)

        prune(cv3[0], cv3[1], threshold)
        prune(cv3[1], cv3[2], threshold)

    for name, p in yolo.model.named_parameters():
        p.requires_grad = True

    torch.save(yolo.ckpt, 'yolov8-prune.pt')
    print('successfully pruned!')