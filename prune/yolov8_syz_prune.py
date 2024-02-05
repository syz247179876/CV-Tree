from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f, CPCA
from yolov8_slim import prune, prune_conv
import typing as t
import torch.nn as nn
import torch

__all__ = ['prune_c2f_cpca']

def prune_c2f_cpca(m1: nn.Module, m2: nn.Module, threshold: float):
    """
    剪枝C2f的输出和修改对应CPCA的输入
    """
    if isinstance(m1, C2f):
        m1 = m1.cv2
    if not isinstance(m2, t.List):
        m2 = [m2]
    for i, item in enumerate(m2):
        if isinstance(item, CPCA):
            m2[i] = item.ca.fc1
    prune_conv(m1, m2, threshold=threshold)

if __name__ == '__main__':
    # yolo = torch.load(r'C:\yolov8\runs\detect\train-VOC.yaml-baseline-v8n2\weights\last.pt', map_location='cpu')
    yolo = YOLO(r'C:\yolov8\runs\detect\train-VOC.yaml-CloFormer-XXS-CPCA-Ultimate\weights\best.pt')
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

    # 目的剪枝80%
    factor = 0.8
    ws = torch.cat(ws)
    # 根据BN层的scale factor降序排列, 选择
    threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
    print(model)

    # 1.剪枝模型融合层的C2f中的Bottleneck中的Conv
    for name, m in model.named_modules():
        if isinstance(m, Bottleneck):
            prune_conv(m.cv1, m.cv2, threshold)

    seq = model.model
    # 2.剪枝模型融合层中前两层的C2f和CPCA
    fusion_c2f, fusion_cpca1 = seq[15], seq[16]
    prune_c2f_cpca(fusion_c2f, fusion_cpca1, threshold=threshold)

    # 3.剪枝neck中Detect, 作为Detect输入的C2f, 下采样
    detect = seq[-1]
    fusion_cpca2 = seq[20]
    last_inputs = [seq[19], seq[23], seq[26]]
    downsample = [seq[21], seq[24], None]

    for idx, (last_input, ds, cv2, cv3) in enumerate(zip(last_inputs, downsample, detect.cv2, detect.cv3)):
        if last_input is None:
            continue
        if idx == 0:
            prune(last_input, [ds, fusion_cpca2, cv2[0], cv3[0]], threshold)
        else:
            prune(last_input, [ds, cv2[0], cv3[0]], threshold)
        prune(cv2[0], cv2[1], threshold)
        prune(cv2[1], cv2[2], threshold)

        prune(cv3[0], cv3[1], threshold)
        prune(cv3[1], cv3[2], threshold)

    for name, p in yolo.model.named_parameters():
        p.requires_grad = True

    torch.save(yolo.ckpt, 'yolov8-syz-prune.pt')
    print('successfully pruned!')