#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_gpu(const Dtype * bottom_data, const Dtype * bottom_diff, const * top_data, const Dtype * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_,  Dtype * taylor_in, Dtype * taylor_in) {
  Dtype * output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_){
    output_saliency_data = output_saliencies_points_.mutable_gpu_data();
    caffe_gpu_mul(output_saliencies_points_.count(), top_data, top_diff, output_saliency_data);
    caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(-1 * this->num_), output_saliency_data);
  }
  if (this->input_channel_saliency_compute_){
    input_saliency_data = input_saliencies_points_.mutable_gpu_data();
    caffe_gpu_mul(input_saliencies_points_.count(), bottom_data, bottom_diff, input_saliency_data);
    caffe_gpu_scal(input_saliencies_points_.count(), (Dtype)(-1 * this->num_), input_saliency_data);
  }
  compute_norm_and_batch_avg_gpu(input_saliency_data, output_saliency_data, saliency_norm_, taylor_in, taylor_out);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_in, Dtype * taylor_out) {
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

  compute_norm_and_batch_avg_weights_gpu(points_saliency_data, bias_saliency_data, saliency_norm_, taylor_in, taylor_out);

}

template void ConvolutionLayer<float>::compute_taylor_gpu(const float * bottom_data, const float * bottom_diff, const float * top_data, const float * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * taylor_in, float * taylor_out);
template void ConvolutionLayer<double>::compute_taylor_gpu(const double * bottom_data, const double * bottom_diff, const double * top_data, const double * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * taylor_in, double * taylor_out);

template void ConvolutionLayer<float>::compute_taylor_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * taylor_in, float * taylor_out);
template void ConvolutionLayer<double>::compute_taylor_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * taylor_in, double * taylor_out);
}  // namespace caffe
