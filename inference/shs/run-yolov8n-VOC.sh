# note: use git bash to execute

python ../workflow.py --pt_name='VOC-yolov8n.pt' --onnx_name='VOC-yolov8n.onnx' --engine_name='VOC-yolov8n.engine' --analysis_name='VOC-yolov8n.txt' --dynamic --simplify --verbose --fuse

echo 按任意键继续
# 监听一个字母 + 关闭回显
read -n 1 -s
echo 继续运行