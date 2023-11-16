import os, shutil
from sklearn.model_selection import train_test_split

val_size = 0.1
test_size = 0.2
postfix = 'jpg'
imgpath = r'C:\dataset\VOCdevkit\VOC2012\JPEGImages'
txtpath = r'C:\dataset\VOCdevkit\VOC2012\txt'

img_train_path = r'C:\dataset\VOCdevkit\VOC2012\images\train'
img_val_path = r'C:\dataset\VOCdevkit\VOC2012\images\val'
img_test_path = r'C:\dataset\VOCdevkit\VOC2012\images\test'

txt_train_path = r'C:\dataset\VOCdevkit\VOC2012\labels\train'
txt_val_path = r'C:\dataset\VOCdevkit\VOC2012\labels\val'
txt_test_path = r'C:\dataset\VOCdevkit\VOC2012\labels\test'

os.makedirs(img_train_path, exist_ok=True)
os.makedirs(img_val_path, exist_ok=True)
os.makedirs(img_test_path, exist_ok=True)
os.makedirs(txt_train_path, exist_ok=True)
os.makedirs(txt_val_path, exist_ok=True)
os.makedirs(txt_test_path, exist_ok=True)

listdir = [i for i in os.listdir(txtpath) if 'txt' in i]
train, test = train_test_split(listdir, test_size=test_size, shuffle=True, random_state=0)
train, val = train_test_split(train, test_size=val_size, shuffle=True, random_state=0)
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '{}\{}.{}'.format(img_train_path, i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '{}\{}'.format(txt_train_path, i))

for i in val:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), r'{}\{}.{}'.format(img_val_path, i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '{}\{}'.format(txt_val_path, i))

for i in test:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), r'{}\{}.{}'.format(img_test_path, i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '{}\{}'.format(txt_test_path, i))
