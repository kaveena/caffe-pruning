#!/bin/bash

export CMAKE_BUILD_TYPE="Release" 

export CMAKE_PARALLEL_LEVEL=`grep processor /proc/cpuinfo | wc -l`

CMAKE_OPTIONS="-DCPU_ONLY=OFF \
               -DUSE_NCCL=ON \
               -DBUILD_tools=OFF \
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
               -Dpython_version=3"
               
rm -rf build && mkdir -p build

rm -rf install && mkdir -p install 

cmake -Bbuild $CMAKE_OPTIONS -DCMAKE_INSTALL_PREFIX=./install .

cd build

make -j`grep processor /proc/cpuinfo | wc -l` clean caffe caffeproto pycaffe python

make install

cd ..
