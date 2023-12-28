# note: use git bash to execute

python ../workflow.py --pt_name='VOC-yolov8n-SK.pt' --onnx_name='VOC-yolov8n-SK.onnx' --engine_name='VOC-yolov8n-SK.engine' --analysis_name='VOC-yolov8n-SK.txt' --dynamic --simplify --verbose --fuse

echo 按任意键继续
# 监听一个字母 + 关闭回显
read -n 1 -s
echo 继续运行