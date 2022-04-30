#!/bin/bash
DOCKER_IMAGE="dansato1203/tf_ball_detection:latest"

if ! command -v nvidia-smi &> /dev/null
then
	echo "NVIDIA Driver not be found"
	exit 0
fi

docker pull "$DOCKER_IMAGE"
docker run --gpus=all -it --rm -e DISPLAY --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	--name tf_ball_detection \
	$DOCKER_IMAGE /run_train.sh
