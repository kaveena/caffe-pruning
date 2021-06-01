// GELU neuron activation function layer.
#include <vector>

#include "caffe/layers/gelu_layer.hpp"

// approximation of \sqrt{\frac{2}{\pi}}
#define SQRT_2_over_pi 0.7978845608

namespace caffe {

template <typename Dtype>
void GELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const float coeff = this->layer_param_.gelu_param().coeff();
  for (int i = 0; i < count; ++i) {
    Dtype x = bottom_data[i];
    Dtype tanhx_data = tanh((SQRT_2_over_pi * x) + (SQRT_2_over_pi * 0.044715 * x * x * x));
    top_data[i] = coeff * 0.5 * x * ( 1 + tanhx_data);
  }
}

template <typename Dtype>
void GELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const float coeff = this->layer_param_.gelu_param().coeff();
    for (int i = 0; i < count; ++i) {
      Dtype x = bottom_data[i];
      Dtype tanhx_data = tanh((SQRT_2_over_pi * x) + (SQRT_2_over_pi * 0.044715 * x * x * x));
      bottom_diff[i] = top_diff[i] * (1 - (tanhx_data * tanhx_data));
      bottom_diff[i] = 0.5 + (0.5 * tanhx_data) + (0.5 * SQRT_2_over_pi * bottom_diff[i] * (x + 0.134145 * x * x * x));
      bottom_diff[i] = bottom_diff[i] * coeff;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GELULayer);
#endif

INSTANTIATE_CLASS(GELULayer);

}  // namespace caffe
