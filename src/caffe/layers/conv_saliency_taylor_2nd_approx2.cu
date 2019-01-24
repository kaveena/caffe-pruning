#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_approx2_gpu(const Dtype *  act_data, const Dtype * act_diff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  
  caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_diff, output_saliency_data); //a * 1/N *dE/da
  caffe_gpu_mul(output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //a * 1/N *  1/N * (dE/da)**2
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(this->num_ / 2), output_saliency_data);  //1/2N * (a * (dE/da2)**2)
  caffe_gpu_sub(output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * (dE/da2)**2) - 1/N * dE/da 
  caffe_gpu_mul(output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * (dE/da2)**2) - a/N*dE/da
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_gpu(output_saliencies_points_.count(2,4), output_saliency_data, taylor_2nd);
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_2nd) {
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
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //a * 1/N *dE/da
  }
  
  caffe_gpu_mul(weights_n->count(), points_saliency_data, weights_n_diff, points_saliency_data); //a * 1/N *  1/N * (dE/da)**2
  caffe_gpu_scal(weights_n->count(), (Dtype)(this->num_ / 2), points_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_gpu_sub(weights_n->count(), points_saliency_data, weights_n_diff, points_saliency_data); //(a/2N * (dE/da)**2) - 1/N * dE/da 
  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, points_saliency_data + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //(a**2/2N * (dE/da)**2) - a/N*dE/da
  }
  caffe_gpu_scal(weights_n->count(), (Dtype)(this->num_), points_saliency_data);
  
  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    bias_n_diff = bias_n->gpu_diff();
    bias_saliency_data = bias_n->mutable_gpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_gpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_n_diff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_mul(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_ / 2), bias_saliency_data);
    caffe_gpu_sub(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_saliency_data + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_gpu(weights_n->count(2, 5), points_saliency_data, taylor_2nd, bias_saliency_data);
}

template void ConvolutionSaliencyLayer<float>::compute_taylor_2nd_approx2_gpu(const float *  act_data, const float * act_diff, float * taylor_2nd);
template void ConvolutionSaliencyLayer<double>::compute_taylor_2nd_approx2_gpu(const double *  act_data, const double * act_diff, double * taylor_2nd);

template void ConvolutionSaliencyLayer<float>::compute_taylor_2nd_approx2_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, float * taylor_2nd);
template void ConvolutionSaliencyLayer<double>::compute_taylor_2nd_approx2_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, double * taylor_2nd);
}  // namespace caffe
