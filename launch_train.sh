#!/bin/bash

step=1000
batch_size=8
num_class=3

if ! command -v nvidia-smi &> /dev/null
then
	echo "NVIDIA Driver not be found"
	exit 0
fi

DOCKER_IMAGE="tf_detection:trainer"
docker run --gpus=all -it --rm \
	-e DISPLAY \
	--privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	--name tf_detection \
	-v $PWD/train_logs:/src/train_logs:rw \
	$DOCKER_IMAGE /src/run_train.sh $step $batch_size $num_class

DOCKER_IMAGE="tf_detection:converter"
docker run --gpus=all -it --rm \
	-e DISPLAY \
	--privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	--name tf_detection \
	-v $PWD/train_logs:/src/train_logs:rw \
	$DOCKER_IMAGE
