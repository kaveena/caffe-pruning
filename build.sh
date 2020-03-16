#!/bin/bash

export CMAKE_BUILD_TYPE="Release" 

export CMAKE_PARALLEL_LEVEL=`grep processor /proc/cpuinfo | wc -l`

CMAKE_OPTIONS="-DCPU_ONLY=ON \
               -DUSE_NCCL=OFF \
               -DBUILD_tools=ON \
               -DBUILD_SHARED_LIBS=OFF \
               -DBUILD_python=ON \
               -DBUILD_matlab=OFF \
               -DBUILD_docs=OFF \
               -DBUILD_python_layer=ON \
               -DUSE_OPENCV=ON \
               -DUSE_LEVELDB=OFF \
               -DUSE_LMDB=OFF \
               -DUSE_HDF5=ON \
               -DALLOW_LMDB_NOLOCK=OFF \
               -DUSE_OPENMP=ON \
               -DBLAS=CBLAS \
               -D python_version=3"
               
mkdir -p build

mkdir -p install

cd build 

cmake $CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX:PATH=$(realpath ../install/usr) -DCMAKE_INSTALL_LIBDIR=$(realpath ../install/lib) .. 

make -j`grep processor /proc/cpuinfo | wc -l` clean caffe caffeproto pycaffe python

make install

cd ..
