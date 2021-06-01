// GELU neuron activation function layer.

#include <vector>

#include "caffe/layers/gelu_layer.hpp"

// approximation of \sqrt{\frac{2}{\pi}}
#define SQRT_2_over_pi 0.7978845608

namespace caffe {

template <typename Dtype>
__global__ void GELUForward(const int n, const float coeff, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx_data = tanh((SQRT_2_over_pi * in[index]) + (SQRT_2_over_pi * 0.044715 * in[index] * in[index] * in[index]));
    out[index] = coeff * 0.5 * in[index] * ( 1 + tanhx_data );
  }
}

template <typename Dtype>
void GELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const float coeff = this->layer_param_.gelu_param().coeff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  GELUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, coeff, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GELUBackward(const int n, const float coeff, const Dtype* bottom_data,
    const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x = bottom_data[index];
    Dtype tanhx_data = tanh((SQRT_2_over_pi * x) + (SQRT_2_over_pi * 0.044715 * x * x * x));
    bottom_diff[index] = top_diff[index] * (1 - (tanhx_data * tanhx_data));
    bottom_diff[index] = 0.5 + (0.5 * tanhx_data) + (0.5 * SQRT_2_over_pi * bottom_diff[index] * (x + 0.134145 * x * x * x));
    bottom_diff[index] = bottom_diff[index] * coeff;
  }
}

template <typename Dtype>
void GELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const float coeff = this->layer_param_.gelu_param().coeff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    GELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, coeff, bottom_data, top_diff, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GELULayer);

}  // namespace caffe
