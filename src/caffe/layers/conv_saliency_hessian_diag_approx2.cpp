#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_approx2_cpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_powx(this->output_saliencies_points_.count(), output_saliency_data, (Dtype)2, output_saliency_data);
  caffe_scal(this->output_saliencies_points_.count(), (Dtype)(this->num_ * this->num_ / 2), output_saliency_data);
  

  compute_norm_and_batch_avg_cpu(output_saliencies_points_.count(2,4), output_saliency_data, hessian_diag);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_approx2_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * hessian_diag) {
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

  caffe_scal(weights_n->count(), (Dtype)(this->num_ * this->num_ / 2), points_saliency_data);
  
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
    caffe_scal(bias_n->count(), (Dtype)(this->num_ * this->num_ / 2), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_cpu(weights_n->count(2, 5), points_saliency_data, hessian_diag, bias_saliency_data);
}

template void ConvolutionSaliencyLayer<float>::compute_hessian_diag_approx2_cpu(const float *  act_data, const float *  act_diff, float * hessian_diag);
template void ConvolutionSaliencyLayer<double>::compute_hessian_diag_approx2_cpu(const double *  act_data, const double *  act_diff, double * hessian_diag);

template void ConvolutionSaliencyLayer<float>::compute_hessian_diag_approx2_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, float * hessian_diag);
template void ConvolutionSaliencyLayer<double>::compute_hessian_diag_approx2_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, double * hessian_diag);
}  // namespace caffe
