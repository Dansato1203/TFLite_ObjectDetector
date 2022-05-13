import os
import shutil
import glob
import random

original_data_dir = '/src/images'
train_data_dir = '/src/dataset/train'
val_data_dir = '/src/dataset/val'

image_dir = '/src/dataset/images'
xml_dir = '/src/dataset/xml'

shutil.rmtree(train_data_dir, ignore_errors=True)
shutil.rmtree(val_data_dir, ignore_errors=True)
os.mkdir(train_data_dir)
os.mkdir(val_data_dir)
os.mkdir(image_dir)
os.mkdir(xml_dir)


# ファイル数カウント(Count the number of files)
file_count = len(glob.glob(original_data_dir + '/*.jpg'))
print('File count : ' + str(file_count))

# 学習データ/検証データ 分割(Split Training data/validation data.)
train_ratio = 0.7

file_list = glob.glob(original_data_dir + '/*.jpg')
random_sample_list = random.sample(file_list, file_count)

# ディレクトリへコピー(Copy to directory)
for index, filepath in enumerate(random_sample_list):
    if index < int(file_count * train_ratio):
        # 学習データ(Training data)
        shutil.copy2(filepath, train_data_dir)
        shutil.copy2(filepath, image_dir)
        shutil.copy2(os.path.splitext(filepath)[0] + ".xml", xml_dir)
    else:
        # 検証データ(Validation data)
        shutil.copy2(filepath, val_data_dir)
        shutil.copy2(filepath, image_dir)
        shutil.copy2(os.path.splitext(filepath)[0] + ".xml", xml_dir)

print("-------------File split done--------------")
