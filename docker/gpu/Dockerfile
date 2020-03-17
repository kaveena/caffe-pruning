FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer caffe-maint@googlegroups.com

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libopenblas-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
	libnccl-dev \
        protobuf-compiler \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-setuptools \
        python3-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=master

RUN git clone -b ${CLONE_TAG} --depth 1 https://bitbucket.org/STG-TCD/trinnity-caffe.git . && \
    python3 -m pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do python3 -m pip install $req; done && cd .. && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DBLAS=Open -Dpython_version=3 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
