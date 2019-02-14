#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_cpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data(); 
  
  caffe_mul(output_saliencies_points_.count(), act_data, act_data, output_saliency_data);
  caffe_mul(output_saliencies_points_.count(), output_saliency_data, act_ddiff, output_saliency_data);
  
  caffe_scal(output_saliencies_points_.count(), 1/(Dtype)(2), output_saliency_data);

  compute_norm_and_batch_avg_cpu(output_saliency_data, NULL, hessian_diag, NULL);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * hessian_diag) {
  const Dtype* weights = this->blobs_[0]->cpu_data();
  const Dtype* weights_n_ddiff = weights_n->cpu_ddiff();
  Dtype* points_saliency_data = weights_n->mutable_cpu_data();
  
  const Dtype* bias;
  const Dtype* bias_n_ddiff;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.cpu_data();
  }
  
  for (int n = 0; n<this->num_; n++) {
    caffe_mul(this->blobs_[0]->count(), weights, weights_n_ddiff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
    caffe_mul(this->blobs_[0]->count(), weights, points_saliency_data + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
 
  caffe_scal(weights_n->count(), 1/(Dtype)(2), points_saliency_data);
  
  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
    bias_n_ddiff = bias_n->cpu_ddiff();
    bias_saliency_data = bias_n->mutable_cpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_cpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_mul(this->blobs_[1]->count(), bias, bias_n_ddiff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
      caffe_mul(this->blobs_[1]->count(), bias, bias_n_ddiff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_scal(bias_n->count(), 1/(Dtype)(2), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_cpu(points_saliency_data, bias_saliency_data, hessian_diag, NULL);
}

template void ConvolutionSaliencyLayer<float>::compute_hessian_diag_cpu(const float *  act_data, const float * act_diff, const float *  act_ddiff, float * hessian_diag);
template void ConvolutionSaliencyLayer<double>::compute_hessian_diag_cpu(const double *  act_data, const double * act_diff, const double *  act_ddiff, double * hessian_diag);

template void ConvolutionSaliencyLayer<float>::compute_hessian_diag_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, float * hessian_diag);
template void ConvolutionSaliencyLayer<double>::compute_hessian_diag_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, double * hessian_diag);
}  // namespace caffe
