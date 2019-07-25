#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_gpu(const Dtype *  act_data, const Dtype *  act_diff, const Dtype * input_data, const Dtype * input_diff,  Dtype * taylor_out, Dtype * taylor_in) {
  Dtype * output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_) {
    output_saliency_data = output_saliencies_points_.mutable_gpu_data();
    caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
    caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(-1 * this->num_), output_saliency_data);

  }
  if (this->input_channel_saliency_compute_) {
    input_saliency_data = input_saliencies_points_.mutable_gpu_data();
    caffe_gpu_mul(input_saliencies_points_.count(), input_data, input_diff, input_saliency_data);
    caffe_gpu_scal(input_saliencies_points_.count(), (Dtype)(-1 * this->num_), input_saliency_data);
  }
  compute_norm_and_batch_avg_gpu(output_saliency_data, input_saliency_data, taylor_out, taylor_in);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_out, Dtype * taylor_in) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  const Dtype* weights_n_diff = weights_n->gpu_diff();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();

  const Dtype* bias;
  const Dtype* bias_n_diff;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }

  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
  caffe_gpu_scal(weights_n->count(), (Dtype) (-1 * this->num_), points_saliency_data); // get unscaled diff back

  if (this->saliency_bias_ && this->bias_term_ && this->output_channel_saliency_compute_) {
    bias = this->blobs_[1]->gpu_data();
    bias_n_diff = bias_n->gpu_diff();
    bias_saliency_data = bias_n->mutable_gpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_gpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_n_diff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), (Dtype) (-1 * this->num_), bias_saliency_data); // get unscaled diff back
  }

  compute_norm_and_batch_avg_weights_gpu(points_saliency_data, bias_saliency_data, taylor_out, taylor_in);

}

template void ConvolutionLayer<float>::compute_taylor_gpu(const float *  act_data, const float *  act_diff, const float * input_data, const float * input_diff,  float * taylor_out, float * taylor_in);
template void ConvolutionLayer<double>::compute_taylor_gpu(const double *  act_data, const double *  act_diff, const double * input_data, const double * input_diff,  double * taylor_out, double * taylor_in);

template void ConvolutionLayer<float>::compute_taylor_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, float * taylor_out, float * taylor_in);
template void ConvolutionLayer<double>::compute_taylor_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, double * taylor_out, double * taylor_in);
}  // namespace caffe
