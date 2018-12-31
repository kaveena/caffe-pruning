#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_masked_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionMaskedLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionMaskedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->mask_term_) {
    weights_masked_shape_.clear();
    weights_masked_shape_.push_back(this->blobs_[this->mask_pos_]->count());
    weights_masked_.Reshape(weights_masked_shape_);
    if (this->bias_term_) {
      bias_masked_shape_.clear();
      bias_masked_shape_.push_back(this->blobs_[this->mask_pos_+1]->count());
      bias_masked_.Reshape(bias_masked_shape_);
    }
  }
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void ConvolutionMaskedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias;
  const Dtype* mask = this->blobs_[this->mask_pos_]->cpu_data();
  Dtype* weight_masked = this->weights_masked_.mutable_cpu_data();
  caffe_mul(this->blobs_[0]->count(), mask, weight, weight_masked);
  weight = this->weights_masked_.cpu_data();
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
    const Dtype* bias_mask = this->blobs_[this->mask_pos_+1]->cpu_data();
    Dtype* bias_masked = this->bias_masked_.mutable_cpu_data();
    caffe_mul(this->blobs_[1]->count(), bias_mask, bias, bias_masked);
    bias = this->bias_masked_.cpu_data();
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionMaskedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
      caffe_mul(this->blobs_[1]->count(), this->blobs_[this->mask_pos_+1]->cpu_data(), this->blobs_[1]->mutable_cpu_diff(), this->blobs_[1]->mutable_cpu_diff());
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
          // Don't update weights that are masked off
          caffe_mul(this->blobs_[0]->count(), this->blobs_[this->mask_pos_]->cpu_data(), weight_diff, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionMaskedLayer);
#endif

INSTANTIATE_CLASS(ConvolutionMaskedLayer);

}  // namespace caffe
