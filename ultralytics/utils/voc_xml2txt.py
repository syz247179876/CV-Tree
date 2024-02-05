import xml.etree.ElementTree as ET
import os, cv2
import typing as t
import numpy as np
from os import listdir
from os.path import join

classes = [
    'person',
    'bike',
    'car',
    'motor',
    'bus',
    'train',
    'truck',
    'light',
    'hydrant',
    'sign',
    'skateboard',
    'stroller',
    'scooter',
    'other vehicle',
]
# 统计每个类别的实例个数
instance_num = {}

empty_xml = []
def get_image_size(path: str) -> t.Tuple:
    image = cv2.imread(path)
    h, w, _ = image.shape
    return h, w

def convert(size, box) -> t.Tuple:
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(xmlpath: str, xmlname: str, height: t.Optional[int], width: t.Optional[int]):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        if size is not None:
            filename = root.find('filename')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

        # img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        # h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            instance_num[cls] = instance_num.setdefault(cls, 0) + 1
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((width, height), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))
        else:
            empty_xml.append(xmlpath)


if __name__ == "__main__":
    postfix = 'jpg'
    data_path = r'C:\dataset\FLIR_ADAS_v2\images_rgb_train'
    imgpath = rf'{data_path}\data'
    xmlpath = rf'{data_path}\xml\data'
    txtpath = rf'{data_path}\txt'
    success_num = 0
    fail_num = 0

    h, w = get_image_size(rf'{imgpath}\{listdir(imgpath)[0]}')
    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)

    list = os.listdir(xmlpath)
    error_file_list = []
    for i in range(0, len(list)):
        try:
            path = os.path.join(xmlpath, list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i], h, w)
                success_num += 1
                print(f'file {list[i]} convert success.')
            else:
                print(f'file {list[i]} is not xml format.')
                error_file_list.append(list[i])
                fail_num += 1

        except Exception as e:
            print(f'file {list[i]} convert error.')
            print(f'error message:\n{e}')
            error_file_list.append(list[i])
            fail_num += 1
    with open(rf'{data_path}\classes.txt', 'w') as f:
        f.write('\n'.join(classes))
    print('instance_num:', instance_num)
    print(f'this file convert failure\n{error_file_list}')
    print(f'Dataset Classes:{classes}')
    print('success num:', success_num)
    print('fail num:', fail_num)
    print('empty_xml:', empty_xml)
