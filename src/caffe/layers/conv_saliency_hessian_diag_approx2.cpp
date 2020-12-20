#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_hessian_diag_approx2_cpu(const Dtype * bottom_data, const * bottom_diff, const Dtype * top_data, const Dtype * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * hessian_diag_in, Dtype * hessian_diag_out) {

  Dtype * output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_){
    output_saliency_data = output_saliencies_points_.mutable_cpu_data();
    caffe_mul(output_saliencies_points_.count(), top_data, top_diff, output_saliency_data);
    caffe_powx(output_saliencies_points_.count(), output_saliency_data, (Dtype)2, output_saliency_data);
    caffe_scal(output_saliencies_points_.count(), (Dtype)(this->num_ * this->num_ * 0.5), output_saliency_data);
  }
  if (this->input_channel_saliency_compute_){
    input_saliency_data = input_saliencies_points_.mutable_cpu_data();
    caffe_mul(input_saliencies_points_.count(), bottom_data, bottom_diff, input_saliency_data);
    caffe_powx(input_saliencies_points_.count(), input_saliency_data, (Dtype)2, input_saliency_data);
    caffe_scal(input_saliencies_points_.count(), (Dtype)(this->num_ * this->num_ * 0.5), input_saliency_data);
  }

  compute_norm_and_batch_avg_cpu(input_saliency_data, output_saliency_data, saliency_norm_, hessian_diag_in, hessian_diag_out);

}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_hessian_diag_approx2_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * hessian_diag_in, Dtype * hessian_diag_out) {
  const Dtype* weights = this->blobs_[0]->cpu_data();
  const Dtype* weights_n_diff = weights_n->cpu_diff();
  Dtype* points_saliency_data = weights_n->mutable_cpu_data();

  const Dtype* bias;
  const Dtype* bias_n_diff;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.cpu_data();
  }

  for (int n = 0; n<this->num_; n++) {
    caffe_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
  caffe_powx(weights_n->count(), points_saliency_data, (Dtype) 2, points_saliency_data);

  caffe_scal(weights_n->count(), (Dtype)(this->num_ * this->num_ * 0.5), points_saliency_data);

  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
    bias_n_diff = bias_n->cpu_diff();
    bias_saliency_data = bias_n->mutable_cpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_cpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_mul(this->blobs_[1]->count(), bias, bias_n_diff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_powx(bias_n->count(), bias_saliency_data, (Dtype) 2, bias_saliency_data);
    caffe_scal(bias_n->count(), (Dtype)(this->num_ * this->num_ * 0.5), bias_saliency_data);
  }
  compute_norm_and_batch_avg_weights_cpu(points_saliency_data, bias_saliency_data, saliency_norm_, hessian_diag_in, hessian_diag_out);
}

template void ConvolutionLayer<float>::compute_hessian_diag_approx2_cpu(const float * bottom_data, const * bottom_diff, const float * top_data, const float * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * hessian_diag_in, float * hessian_diag_out);
template void ConvolutionLayer<double>::compute_hessian_diag_approx2_cpu(const double * bottom_data, const * bottom_diff, const double * top_data, const double * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * hessian_diag_in, double * hessian_diag_out);

template void ConvolutionLayer<float>::compute_hessian_diag_approx2_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * hessian_diag_in, float * hessian_diag_out);
template void ConvolutionLayer<double>::compute_hessian_diag_approx2_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * hessian_diag_in, double * hessian_diag_out);
}  // namespace caffe
