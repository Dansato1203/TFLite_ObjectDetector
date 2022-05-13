#! /bin/bash

#readlink -f /src/data/train/* > train_data.txt
#sed -i "s/.tfrecord/" train_data.txt
cd /src/dataset/train
for f in *.jpg ; do
	echo $f | sed 's/\.[^\.]*$//' >> ../train.txt
done

cd /src/dataset/val
for f in *.jpg ; do
	echo $f | sed 's/\.[^\.]*$//' >> ../val.txt
done
