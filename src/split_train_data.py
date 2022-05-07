import os
import shutil
import glob
import random

original_data_dir = '/src/origin_data'
train_data_dir = '/src/models/research/train_data'
val_data_dir = '/src/models/research/val_data'

shutil.rmtree(train_data_dir, ignore_errors=True)
shutil.rmtree(val_data_dir, ignore_errors=True)
os.mkdir(train_data_dir)
os.mkdir(val_data_dir)

# ファイル数カウント(Count the number of files)
file_count = len(glob.glob(original_data_dir + '/*.tfrecord'))
print('File count : ' + str(file_count))

# 学習データ/検証データ 分割(Split Training data/validation data.)
train_ratio = 0.7

file_list = glob.glob(original_data_dir + '/*.tfrecord')
random_sample_list = random.sample(file_list, file_count)

# ディレクトリへコピー(Copy to directory)
for index, filepath in enumerate(random_sample_list):
    if index < int(file_count * train_ratio):
        # 学習データ(Training data)
        shutil.copy2(filepath, train_data_dir)
    else:
        # 検証データ(Validation data)
        shutil.copy2(filepath, val_data_dir)
