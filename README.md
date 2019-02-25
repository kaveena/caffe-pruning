# triNNity-caffe

Caffe is a deep learning framework made with expression, speed, and
modularity in mind. It is developed by Berkeley AI Research
([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning
Center (BVLC) and community contributors.

This is a customized distribution of Caffe developed by the Software
Tools Group at Trinity College Dublin. It is maintained as a fork that
cleanly merges over the BVLC Caffe master branch.

triNNity-caffe includes:

- Dynamic LR adjustment (triangular and triangular2)
- Saliency Computation during training
- Flexible Quantization to arbitrary bit precision
- Pruning (Sparsification), both pointwise and channel pruning
- General quality-of-life fixes and performance improvements

## Building Caffe

There are many CMake options to be aware of when building Caffe for different
uses. Here is a summary of the most important build options. You can control
any of these on the command line when invoking CMake by saying
`-DOPTION=value`. For a complete option listing, use `cmake -LA`.

`BLAS:STRING=Open`

Which BLAS library to use for matrix operations

`BUILD_SHARED_LIBS:BOOL=ON`

Whether to build `libcaffe.so` (if you're linking Caffe into another app)

`BUILD_docs:BOOL=ON`

Whether to build the doxygen docs

`BUILD_matlab:BOOL=OFF`

Whether to build the matlab interface

`BUILD_python:BOOL=ON`

Whether to build the python interface (pycaffe)

`BUILD_python_layer:BOOL=ON`

Whether to enable the python layer in Caffe

`BUILD_tools:BOOL=ON`

Whether to build the Caffe tools (the contents of the tools/ subdirectory)

`CPU_ONLY:BOOL=OFF`

Turn on or off GPU support in Caffe

`USE_CUDNN:BOOL=ON`

Whether to use CUDNN in GPU mode (OFF means use standard CUBLAS)

`USE_HDF5:BOOL=ON`

Whether to support weights and data in the HDF5 database format

`USE_LEVELDB:BOOL=ON`

Whether to support weights and data in the LEVELDB database format

`USE_LMDB:BOOL=ON`

Whether to support weights and data in the LMDB database format

`USE_NCCL:BOOL=OFF`

Whether to use NCCL for multi-GPU support in GPU mode

`USE_OPENCV:BOOL=ON`

Whether to use OpenCV for handling images

`USE_OPENMP:BOOL=OFF`

Whether to use OpenMP parallelization

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
