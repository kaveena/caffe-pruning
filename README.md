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

## Saliency Computation

When pruning, it is important to know the relative importance, or
*saliency* of each of the items being pruned. triNNity-caffe extends
layers with weights with a new saliency parameter, which can be used to
control how the saliency is computed. Saliency is stored in a Blob,
just like weights and biases. The dimensions of the saliency Blob are
exactly the same as the weight Blob for the layer.

The parameters used to control saliency computation are `saliency`,
`saliency_term`, `accum`, `norm`, and `input`.

```
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5

    saliency_param {
      saliency_term: true
      saliency: TAYLOR
      accum: false
      norm: L2
      input: WEIGHT
    }
  }
```

The `saliency` is the name of the method to be used to compute
saliency. The available methods are `FISHER`, `TAYLOR`,
`HESSIAN_DIAG_LM`, `HESSIAN_DIAG_GN`, `TAYLOR_2ND_LM`,
`TAYLOR_2ND_GN`, `AVERAGE_INPUT`, `AVERAGE_GRADIENT`.

`LM` and `GN` refer to the Levenberg-Marquardt or Gauss-Newton
approximations of the relevant entity.

The `accum` parameters says whether or not to accumulate saliency
across minibatches.

The `norm` parameter selects which norm to use for the computed
saliency: `NONE`, `L1` or `L2`.

Finally, the `input` parameter is set to either `WEIGHT` or
`ACTIVATION` to select whether it is the saliency of weights or of
activations that is being computed.

To temporarily disable saliency computation for a layer without
removing the `saliency_param`, you can set `saliency_term` to `false`.

## Building Caffe

The Caffe build process uses CMake. The build is an out-of-tree build,
so don't run CMake in the Caffe source directory (the directory where this
file is located).

Instead, make a temporary directory anywhere *outside* of this directory,
and say `cmake <options> /path/to/source/directory`. CMake will generate
the build for you in that temporary directory, and then you can just say
`make` to build Caffe there.

### Build Options

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
