# note: use git bash to execute

python ../workflow.py --pt_name='yolov8-CA.pt' --onnx_name='yolov8-CA.onnx' --engine_name='yolov8-CA.engine' --analysis_name='yolov8-CA.txt' --dynamic --simplify --verbose --fuse

echo 按任意键继续
# 监听一个字母 + 关闭回显
read -n 1 -s
echo 继续运行