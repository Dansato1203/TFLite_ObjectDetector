FROM nvcr.io/nvidia/tensorflow:22.04-tf1-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt update && apt install -y --no-install-recommends \
	vim \
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

RUN python3 -m pip install --upgrade pip 

RUN pip3 install opencv-python \
	-q tflite_support \
	tf_slim \
	git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI \
	pycocotools \
	&& pip3 install --upgrade pip \
	cython \
	gdown \
	google-api-python-client 

WORKDIR /src
RUN git clone --depth 1 https://github.com/tensorflow/models 
RUN gdown "https://drive.google.com/uc?export=download&id=19-kOt9khSikXOWVb5o5WCqd6_7j6FRuV" \
	&& unzip *.zip -d tfrecord_data\
	&& rm *.zip 

WORKDIR /src/models/research 
RUN protoc object_detection/protos/*.proto --python_out=.

RUN cp /src/models/research/object_detection/packages/tf1/setup.py . \
	&& python3 -m pip install . \
	&& python3 /src/models/research/object_detection/builders/model_builder_test.py

WORKDIR /src/pretrained_model
RUN wget http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz \
	&& tar xvf ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz
RUN mkdir pbtxt \
	&& wget "https://drive.google.com/uc?export=download&id=1ULi7WDxfckXgWLIVTLdQ_ibpIjLms01T" -O tf_label_map.pbtxt \
	&& cp tf_label_map.pbtxt /src/pretrained_model/pbtxt/

WORKDIR /src/models/research 
RUN mkdir -p /src/train_logs/inference_models

COPY src/run_train.sh /src/
COPY src/split_train_data.py /src/
COPY src/fix_pipeline.py /src/

ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/slim
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/object_detection/utils/
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/object_detection 

WORKDIR /src/
CMD ["/src/run_train.sh"]
