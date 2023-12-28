# note: use git bash to execute

python ../workflow.py --pt_name='VOC-yolov8n-CondConv.pt' --onnx_name='VOC-yolov8n-CondConv.onnx' --engine_name='VOC-yolov8n-CondConv.engine' --analysis_name='VOC-yolov8n-CondConv.txt' --dynamic --simplify --verbose --fuse

echo 按任意键继续
# 监听一个字母 + 关闭回显
read -n 1 -s
echo 继续运行