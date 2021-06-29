FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ENV CUDA_PATH /usr/local/cuda
ENV PATH ${CUDA_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH ${CUDA_PATH}/bin64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH ${CUDA_PATH}/include

RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         vim \
         libopencv-dev \
         python-pip \
         python-opencv \
         git \
         libzmq3-dev \
         libhdf5-serial-dev \
         libboost-all-dev \
         curl \
         unzip

RUN apt-get update && apt-get install -y --no-install-recommends \
	python3
	
RUN curl -O  https://bootstrap.pypa.io/pip/3.5/get-pip.py
RUN python3 get-pip.py
RUN python3 -m pip install --upgrade "pip < 19.2"


RUN apt-get update && apt-get install -y --no-install-recommends \
         python3-dev \
         python3-tk

RUN pip3 install --upgrade pip 
RUN pip3 install -U setuptools
RUN pip3 install Cython \ 
                scikit-image \
                tensorflow-gpu==1.12.0 \
                keras==2.1.1 \
                configobj \
                IPython \
                tqdm \
                pandas \
                opencv-python \
                zmq \ 
                scikit-learn \
                h5py
RUN pip3 install pillow

# start like this
# sudo docker run --runtime=nvidia -v '/home/manuel/Desktop/CDCL-human-part-segmentation':'/CDCL'   -it cdcl:v1 bash
# Auf Cluster sudo docker run --runtime=nvidia -v '/tank/Manuel/CDCL-human-part-segmentation':'/CDCL'   -it cdcl:v1 bash
# python3 inference_15parts.py --scale=1
# ssh manuel@10.34.58.83
# /CDCL/input/01christ.jpg
# /CDCL/input/MonaLisa.jpg
