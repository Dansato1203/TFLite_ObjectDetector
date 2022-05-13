#FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
#FROM tensorflow/tensorflow:1.15.5-gpu-py3


ENV DEBIAN_FRONTEND noninteractive

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

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
	google-api-python-client \
	&& pip3 install opencv-python==4.5.5.62 \
	gdown \
	tensorflow \
	-q tflite_support \
	tf_slim \
	git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

WORKDIR /src
RUN git clone --depth 1 https://github.com/tensorflow/models 
RUN gdown "https://drive.google.com/uc?export=download&id=1k6Nc2xiwB9d2ZRD4LLCS8ndCmPHWqBko" \
	&& unzip origin_data.zip \
	&& rm origin_data.zip 


WORKDIR /src/models/research 
RUN protoc object_detection/protos/*.proto --python_out=.

RUN cp /src/models/research/object_detection/packages/tf2/setup.py . \
	&& python3 -m pip install . \
	&& python3 /src/models/research/object_detection/builders/model_builder_tf2_test.py

WORKDIR /src/pretrained_model
#RUN wget http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz \
#	&& tar xvf ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz
RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz \
	&& tar xvzf efficientdet_d0_coco17_tpu-32.tar.gz \
	&& rm efficientdet_d0_coco17_tpu-32.tar.gz
RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz \
	&& tar xvzf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz \
	&& rm ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

WORKDIR /src/models/research 
RUN git clone https://github.com/karaage0703/object_detection_tools
RUN wget "https://drive.google.com/uc?export=download&id=1ULi7WDxfckXgWLIVTLdQ_ibpIjLms01T" -O tf_label_map.pbtxt \
	&& cp tf_label_map.pbtxt /src/models/research/object_detection_tools/data/

RUN mkdir -p /src/train_logs/inference_models

COPY src/run_train.sh /src/
COPY src/split_train_data.py /src/
COPY src/fix_pipeline.py /src/
COPY src/convert_tflite.py /src/

ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/slim
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/object_detection/utils/
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/object_detection

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update \
	&& apt-get install edgetpu-compiler

WORKDIR /src/
CMD ["/src/run_train.sh"]
