import os, shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = r'C:\dataset\Diverse-Weather_Dataset\Diverse-Weather Dataset\daytime_clear\VOC2007'
SPLIT_DIR = r'C:\dataset\Diverse-Weather_Dataset\Diverse-Weather Dataset\daytime_clear\VOC2007\data'

val_size = 0.2
test_size = 0.2
postfix = 'jpg'
postfix2 = 'png'
imgpath = rf'{DATASET_DIR}\JPEGImages'
txtpath = rf'{DATASET_DIR}\txt'

img_train_path = rf'{SPLIT_DIR}\images\train'
img_val_path = rf'{SPLIT_DIR}\images\val'
img_test_path = rf'{SPLIT_DIR}\images\test'

txt_train_path = rf'{SPLIT_DIR}\labels\train'
txt_val_path = rf'{SPLIT_DIR}\labels\val'
txt_test_path = rf'{SPLIT_DIR}\labels\test'

os.makedirs(img_train_path, exist_ok=True)
os.makedirs(img_val_path, exist_ok=True)
os.makedirs(img_test_path, exist_ok=True)
os.makedirs(txt_train_path, exist_ok=True)
os.makedirs(txt_val_path, exist_ok=True)
os.makedirs(txt_test_path, exist_ok=True)

listdir = [i for i in os.listdir(txtpath) if 'txt' in i]
train, test = train_test_split(listdir, test_size=test_size, shuffle=True, random_state=42)
train, val = train_test_split(train, test_size=val_size, shuffle=True, random_state=42)
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    try:
        shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '{}\{}.{}'.format(img_train_path, i[:-4], postfix))
    except FileNotFoundError:
        shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix2), '{}\{}.{}'.format(img_train_path, i[:-4], postfix2))
    shutil.copy('{}/{}'.format(txtpath, i), '{}\{}'.format(txt_train_path, i))

for i in val:
    try:
        shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), r'{}\{}.{}'.format(img_val_path, i[:-4], postfix))
    except FileNotFoundError:
        shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix2), r'{}\{}.{}'.format(img_val_path, i[:-4], postfix2))
    shutil.copy('{}/{}'.format(txtpath, i), '{}\{}'.format(txt_val_path, i))

for i in test:
    try:
        shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), r'{}\{}.{}'.format(img_test_path, i[:-4], postfix))
    except FileNotFoundError:
        shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix2), r'{}\{}.{}'.format(img_test_path, i[:-4], postfix2))
    shutil.copy('{}/{}'.format(txtpath, i), '{}\{}'.format(txt_test_path, i))
