#FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt update && apt install -y --no-install-recommends \
	vim \
	python-dev \
	python3-dev \
	python3-pip \
	unzip \
	curl \
	wget \
	git \
	libgl1-mesa-dev \
	libglib2.0-0 \
	protobuf-compiler \
	python3-setuptools \
	&& rm -rf /var/lib/apt/lists/*

RUN curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip \
	&& unzip protoc-3.2.0-linux-x86_64.zip -d protoc3 \
	&& mv protoc3/bin/* /usr/local/bin/ \
	&& mv protoc3/include/* /usr/local/include/ \
	&& rm -rf protoc3 protoc-3.2.0-linux-x86_64.zip

RUN pip3 install --upgrade pip \
	cython \
	numpy \
	gdown \
	google-api-python-client \
	&& pip3 install opencv-python==4.3.0.38 \
	-q tflite_support \
	tf_slim \
	tensorflow-gpu==1.15 \
	git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

WORKDIR /src
RUN git clone --depth 1 https://github.com/tensorflow/models 
RUN gdown "https://drive.google.com/uc?export=download&id=1k6Nc2xiwB9d2ZRD4LLCS8ndCmPHWqBko" \
	&& unzip origin_data.zip \
	&& rm origin_data.zip 


WORKDIR /src/models/research 
RUN protoc object_detection/protos/*.proto --python_out=.

RUN cp /src/models/research/object_detection/packages/tf1/setup.py . \
	&& python3 -m pip install . \
	&& python3 /src/models/research/object_detection/builders/model_builder_test.py

WORKDIR /src/pretrained_model
RUN wget http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz \
	&& tar xvf ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz

WORKDIR /src/models/research 
RUN git clone https://github.com/karaage0703/object_detection_tools
RUN wget "https://drive.google.com/uc?export=download&id=1ULi7WDxfckXgWLIVTLdQ_ibpIjLms01T" -O tf_label_map.pbtxt \
	&& cp tf_label_map.pbtxt /src/models/research/object_detection_tools/data/

RUN mkdir -p /src/train_logs/inference_models

COPY scripts/run_train.sh /src/
COPY scripts/split_train_data.py /src/
COPY scripts/fix_pipeline.py /src/

WORKDIR /src/
CMD ["/src/run_train.sh"]
