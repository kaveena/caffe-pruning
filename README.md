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

## Building Caffe for local modification

The `build.sh` script builds Caffe with a simple CPU-only configuration
for testing out modifications to the source. For more build options, 
including GPU support, see the end of this README.

When you have run `build.sh`, you need to `source env.sh` when you 
open a new shell, so that Caffe components are on your `PATH` and 
`PYTHONPATH`. If in doubt, `echo $PATH` and `echo $PYTHONPATH`.

## Building Caffe with Docker

Dockerfiles are provided in the `docker` folder to build an image with
Caffe installed. See `docker/README` for details.

## Manually Building Caffe without `build.sh`

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

## Using the triNNity-caffe pruning extensions

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
saliency. The available methods are  `TAYLOR`,
`HESSIAN_DIAG_APPROX1`, `HESSIAN_DIAG_APPROX2`, `TAYLOR_2ND_APPROX1`,
`TAYLOR_2ND_APPROX2`, `AVERAGE_INPUT`, `AVERAGE_GRADIENT`.

`APPROX1` and `APPROX2` methods both assume a diagonal Hessian for the activations and the weights.  `APPROX1` propagates the diagonal Hessians ins a similar way to backpropagation to compute the 2nd order derivatives.  `APPROX2` uses the Gauss-Newton approximation of Hessian and only retains the diagonal values.

The `accum` parameters says whether or not to accumulate saliency
across minibatches.

The `norm` parameter selects which norm to use for the computed
saliency: `NONE`, `L1` or `L2`.

Finally, the `input` parameter is set to either `WEIGHT` or
`ACTIVATION` to select whether it is the saliency of weights or of
activations that is being computed.

To temporarily disable saliency computation for a layer without
removing the `saliency_param`, you can set `saliency_term` to `false`.

### Second Order Derivatives Computation

Some pointwise saliency methods (`HESSIAN_DIAG_APPROX1` and `TAYLOR_2ND_APPROX1`)
require the computation of second order derivations in similar way to
backpropagation.  To enable the computation of the second order derivatives,
the Net Parameter `2nd_order_derivative` needs to be set to true.  This enables
the memory allocation and computation of the higher order derivatives.

This is global Caffe setting! Hence, if pycaffe is used all the networks need
to either have `2nd_order_derivative` set to true or false.  The default
behaviour is `2nd_order_derivative` set to false.

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
