# translate coco_json to xml
import os
import time
import json
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO


def trans_id(category_id):
    names = []
    namesid = []
    for i in range(0, len(cats)):
        names.append(cats[i]['name'])
        namesid.append(cats[i]['id'])
        # print('id:{1}\t {0}'.format(names[i], namesid[i]))
    index = namesid.index(category_id)
    return index


root = r''  # 你下载的 COCO 数据集所在目录
dataType = '2019'
anno = r'C:\dataset\FLIR_ADAS_v2\images_rgb_val\coco.json'  # annotation json 文件所在位置
xml_dir = r'C:\dataset\FLIR_ADAS_v2\images_rgb_val\xml'  # 导出的xml文件所在的位置
total_classes_map = r'C:\dataset\FLIR_ADAS_v2\images_rgb_val\classes.txt'  # 导出classes文件所在位置

if not os.path.exists(xml_dir):
    os.makedirs(rf'{xml_dir}\data')

coco = COCO(anno)  # 读文件
cats = coco.loadCats(coco.getCatIds())  # 这里loadCats就是coco提供的接口，获取类别

# Create anno dir
dttm = time.strftime("%Y%m%d%H%M%S", time.localtime())
# if os.path.exists(xml_dir):
#     os.rename(xml_dir, xml_dir + dttm)
# os.mkdir(xml_dir)

with open(anno, 'r') as load_f:
    f = json.load(load_f)

imgs = f['images']  # json文件的img_id和图片对应关系 imgs列表表示多少张图

cat = f['categories']
df_cate = pd.DataFrame(f['categories'])  # json中的类别
new_cate = {}
df_cate_sort = df_cate.sort_values(["id"], ascending=True)  # 按照类别id排序
categories = list(df_cate_sort['name'])  # 获取所有类别名称
print('categories = ', categories)
df_anno = pd.DataFrame(f['annotations'])  # json中的annotation

nums = 0
for i in tqdm(range(len(imgs))):  # 大循环是images所有图片
    xml_content = []
    file_name = imgs[i]['file_name']  # 通过img_id找到图片的信息
    height = imgs[i]['height']
    img_id = imgs[i]['id']
    width = imgs[i]['width']

    # xml文件添加属性
    xml_content.append("<annotation>")
    xml_content.append("	<folder>VOC2007</folder>")
    xml_content.append("	<filename>" + file_name.split('/')[1].split('.')[0] + '.jpg' + "</filename>")
    xml_content.append("	<size>")
    xml_content.append("		<width>" + str(width) + "</width>")
    xml_content.append("		<height>" + str(height) + "</height>")
    xml_content.append("	</size>")
    xml_content.append("	<segmented>0</segmented>")

    # 通过img_id找到annotations
    annos = df_anno[df_anno["image_id"].isin([img_id])]  # (2,8)表示一张图有两个框

    for index, row in annos.iterrows():  # 一张图的所有annotation信息
        bbox = row["bbox"]
        category_id = row["category_id"]
        # cate_name = categories[trans_id(category_id)]
        cate_name = cat[category_id - 1]['name']
        new_cate[cate_name] = new_cate.setdefault(cate_name, 0) + 1
        # add new object
        xml_content.append("<object>")
        xml_content.append("<name>" + cate_name + "</name>")
        xml_content.append("<pose>Unspecified</pose>")
        xml_content.append("<truncated>0</truncated>")
        xml_content.append("<difficult>0</difficult>")
        xml_content.append("<bndbox>")
        xml_content.append("<xmin>" + str(int(bbox[0])) + "</xmin>")
        xml_content.append("<ymin>" + str(int(bbox[1])) + "</ymin>")
        xml_content.append("<xmax>" + str(int(bbox[0] + bbox[2])) + "</xmax>")
        xml_content.append("<ymax>" + str(int(bbox[1] + bbox[3])) + "</ymax>")
        xml_content.append("</bndbox>")
        xml_content.append("</object>")
    xml_content.append("</annotation>")

    x = xml_content
    xml_content = [x[i] for i in range(0, len(x)) if x[i] != "\n"]
    ### list存入文件
    xml_path = os.path.join(xml_dir, file_name.replace('.jpg', '.xml'))
    with open(xml_path, 'w+', encoding="utf8") as f:
        f.write('\n'.join(xml_content))
    xml_content[:] = []
    nums += 1

print('new_category:', new_cate)

with open(total_classes_map, 'w+', encoding='utf-8') as f:
    f.write('\n'.join([f"{str(int(item.get('id')) - 1)}: {item.get('name')}" for item in cat]))
print('cat:', cat)
print('nums:', nums)

# ss = [{'id': 1, 'name': 'person', 'supercategory': 'unknown'}, {'id': 2, 'name': 'bike', 'supercategory': 'unknown'}, {'id': 3, 'name': 'car', 'supercategory': 'unknown'}, {'id': 4, 'name': 'motor', 'supercategory': 'unknown'}, {'id': 5, 'name': 'airplane', 'supercategory': 'unknown'}, {'id': 6, 'name': 'bus', 'supercategory': 'unknown'}, {'id': 7, 'name': 'train', 'supercategory': 'unknown'}, {'id': 8, 'name': 'truck', 'supercategory': 'unknown'}, {'id': 9, 'name': 'boat', 'supercategory': 'unknown'}, {'id': 10, 'name': 'light', 'supercategory': 'unknown'}, {'id': 11, 'name': 'hydrant', 'supercategory': 'unknown'}, {'id': 12, 'name': 'sign', 'supercategory': 'unknown'}, {'id': 13, 'name': 'parking meter', 'supercategory': 'unknown'}, {'id': 14, 'name': 'bench', 'supercategory': 'unknown'}, {'id': 15, 'name': 'bird', 'supercategory': 'unknown'}, {'id': 16, 'name': 'cat', 'supercategory': 'unknown'}, {'id': 17, 'name': 'dog', 'supercategory': 'unknown'}, {'id': 18, 'name': 'deer', 'supercategory': 'unknown'}, {'id': 19, 'name': 'sheep', 'supercategory': 'unknown'}, {'id': 20, 'name': 'cow', 'supercategory': 'unknown'}, {'id': 21, 'name': 'elephant', 'supercategory': 'unknown'}, {'id': 22, 'name': 'bear', 'supercategory': 'unknown'}, {'id': 23, 'name': 'zebra', 'supercategory': 'unknown'}, {'id': 24, 'name': 'giraffe', 'supercategory': 'unknown'}, {'id': 25, 'name': 'backpack', 'supercategory': 'unknown'}, {'id': 26, 'name': 'umbrella', 'supercategory': 'unknown'}, {'id': 27, 'name': 'handbag', 'supercategory': 'unknown'}, {'id': 28, 'name': 'tie', 'supercategory': 'unknown'}, {'id': 29, 'name': 'suitcase', 'supercategory': 'unknown'}, {'id': 30, 'name': 'frisbee', 'supercategory': 'unknown'}, {'id': 31, 'name': 'skis', 'supercategory': 'unknown'}, {'id': 32, 'name': 'snowboard', 'supercategory': 'unknown'}, {'id': 33, 'name': 'sports ball', 'supercategory': 'unknown'}, {'id': 34, 'name': 'kite', 'supercategory': 'unknown'}, {'id': 35, 'name': 'baseball bat', 'supercategory': 'unknown'}, {'id': 36, 'name': 'baseball glove', 'supercategory': 'unknown'}, {'id': 37, 'name': 'skateboard', 'supercategory': 'unknown'}, {'id': 38, 'name': 'surfboard', 'supercategory': 'unknown'}, {'id': 39, 'name': 'tennis racket', 'supercategory': 'unknown'}, {'id': 40, 'name': 'bottle', 'supercategory': 'unknown'}, {'id': 41, 'name': 'wine glass', 'supercategory': 'unknown'}, {'id': 42, 'name': 'cup', 'supercategory': 'unknown'}, {'id': 43, 'name': 'fork', 'supercategory': 'unknown'}, {'id': 44, 'name': 'knife', 'supercategory': 'unknown'}, {'id': 45, 'name': 'spoon', 'supercategory': 'unknown'}, {'id': 46, 'name': 'bowl', 'supercategory': 'unknown'}, {'id': 47, 'name': 'banana', 'supercategory': 'unknown'}, {'id': 48, 'name': 'apple', 'supercategory': 'unknown'}, {'id': 49, 'name': 'sandwich', 'supercategory': 'unknown'}, {'id': 50, 'name': 'orange', 'supercategory': 'unknown'}, {'id': 51, 'name': 'broccoli', 'supercategory': 'unknown'}, {'id': 52, 'name': 'carrot', 'supercategory': 'unknown'}, {'id': 53, 'name': 'hot dog', 'supercategory': 'unknown'}, {'id': 54, 'name': 'pizza', 'supercategory': 'unknown'}, {'id': 55, 'name': 'donut', 'supercategory': 'unknown'}, {'id': 56, 'name': 'cake', 'supercategory': 'unknown'}, {'id': 57, 'name': 'chair', 'supercategory': 'unknown'}, {'id': 58, 'name': 'couch', 'supercategory': 'unknown'}, {'id': 59, 'name': 'potted plant', 'supercategory': 'unknown'}, {'id': 60, 'name': 'bed', 'supercategory': 'unknown'}, {'id': 61, 'name': 'dining table', 'supercategory': 'unknown'}, {'id': 62, 'name': 'toilet', 'supercategory': 'unknown'}, {'id': 63, 'name': 'tv', 'supercategory': 'unknown'}, {'id': 64, 'name': 'laptop', 'supercategory': 'unknown'}, {'id': 65, 'name': 'mouse', 'supercategory': 'unknown'}, {'id': 66, 'name': 'remote', 'supercategory': 'unknown'}, {'id': 67, 'name': 'keyboard', 'supercategory': 'unknown'}, {'id': 68, 'name': 'cell phone', 'supercategory': 'unknown'}, {'id': 69, 'name': 'microwave', 'supercategory': 'unknown'}, {'id': 70, 'name': 'oven', 'supercategory': 'unknown'}, {'id': 71, 'name': 'toaster', 'supercategory': 'unknown'}, {'id': 72, 'name': 'sink', 'supercategory': 'unknown'}, {'id': 73, 'name': 'stroller', 'supercategory': 'unknown'}, {'id': 74, 'name': 'rider', 'supercategory': 'unknown'}, {'id': 75, 'name': 'scooter', 'supercategory': 'unknown'}, {'id': 76, 'name': 'vase', 'supercategory': 'unknown'}, {'id': 77, 'name': 'scissors', 'supercategory': 'unknown'}, {'id': 78, 'name': 'face', 'supercategory': 'unknown'}, {'id': 79, 'name': 'other vehicle', 'supercategory': 'unknown'}, {'id': 80, 'name': 'license plate', 'supercategory': 'unknown'}]
# res = []
# for item in ss:
#     res.append(item.get('name'))
# print(res)
