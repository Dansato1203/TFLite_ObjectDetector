FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y --no-install-recommends \
	vim \
	python3 \
	python3-pip \
	unzip \
	curl \
	wget \
	git \
	&& rm -rf /var/lib/apt/lists/*

RUN -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip \
	&& unzip protoc-3.2.0-linux-x86_64.zip -d protoc3 \
	&& mv protoc3/bin/* /usr/local/bin/ \
	&& mv protoc3/include/* /usr/local/include/ \
	&& -rf protoc3 protoc-3.2.0-linux-x86_64.zip

WORKDIR /src
RUN git clone --depth 1 https://github.com/tensorflow/models \
	&& cd /src/models/reserach \
	&& /usr/local/bin/protoc object_detection/protos/*.proto --python_out=.

RUN cp /content/models/research/object_detection/packages/tf2/setup.py . \
	&& python -m pip install .

RUN python /content/models/research/object_detection/builders/model_builder_tf2_test.py

RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz \
	&& tar -xf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz \
	&& rm ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz

RUN pip3 install --upgrade pip \
	&& pip3 install opencv-python==4.3.0.38 \
	-q tflite_support \
	git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

WORKDIR /src/models/research
RUN git clone https://github.com/karaage0703/object_detection_tools

WORKDIR /src/models/research/object_detection_tools/data
RUN "https://drive.google.com/uc?export=download&id=1ULi7WDxfckXgWLIVTLdQ_ibpIjLms01T" -O tf_label_map.pbtxt 

WORKDIR /content/models/research/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8
RUN "https://drive.google.com/uc?export=download&id=1ReEP1J9Rei9LVsrfiLzStZYjYZcA6R4R" -O pipeline.config

WORKDIR /src/models/research
