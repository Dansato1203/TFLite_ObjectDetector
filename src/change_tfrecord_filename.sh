#!/bin/bash
i=0
for f in /src/data/train/*.tfrecord

do
	g=/src/data/train/`printf %04d $i`.tfrecord
	mv $f $g
	i=$((i+1))
done

i=0

for f in /src/data/val/*.tfrecord
do
	g=/src/data/val/`printf %04d $i`.tfrecord
	mv $f $g
	i=$((i+1))
done
