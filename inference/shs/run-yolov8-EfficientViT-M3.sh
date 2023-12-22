# note: use git bash to execute

python ../workflow.py --pt_name='yolov8-EfficientViT-M3.pt' --onnx_name='yolov8-EfficientViT-M3.onnx' --engine_name='yolov8-EfficientViT-M3.engine' --analysis_name='yolov8-EfficientViT-M3.txt' --dynamic --simplify --verbose --fuse

echo 按任意键继续
# 监听一个字母 + 关闭回显
read -n 1 -s
echo 继续运行