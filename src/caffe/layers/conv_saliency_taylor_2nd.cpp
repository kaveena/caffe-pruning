#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_cpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_mul(output_saliencies_points_.count(), act_data, act_ddiff, output_saliency_data); //a * d2E/da2
  caffe_scal(output_saliencies_points_.count(), 1/(Dtype)(2*(this->num_)), output_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_sub(output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * d2E/da2) - 1/N * dE/da 
  caffe_mul(output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * d2E/da2) - a/N*dE/da
  caffe_scal(output_saliencies_points_.count(), (Dtype)(this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_cpu(output_saliencies_points_.count(2,4), output_saliency_data, taylor_2nd);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_2nd) {
  const Dtype* weights = this->blobs_[0]->cpu_data();
  const Dtype* weights_n_diff = weights_n->cpu_diff();
  const Dtype* weights_n_ddiff = weights_n->cpu_ddiff();
  Dtype* points_saliency_data = weights_n->mutable_cpu_data();
  
  const Dtype* bias;
  const Dtype* bias_n_diff;
  const Dtype* bias_n_ddiff;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.cpu_data();
  }
    
  for (int n = 0; n<this->num_; n++) {
    caffe_mul(this->blobs_[0]->count(), weights, weights_n_ddiff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //a * d2E/da2
  }
  
  caffe_scal(weights_n->count(), 1/(Dtype)(2*(this->num_)), points_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_sub(weights_n->count(), points_saliency_data, weights_n_diff, points_saliency_data); //(a/2N * d2E/da2) - 1/N * dE/da 
  for (int n = 0; n<this->num_; n++) {
    caffe_mul(this->blobs_[0]->count(), weights, points_saliency_data + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //(a**2/2N * d2E/da2) - a/N*dE/da
  }
  caffe_scal(weights_n->count(), (Dtype)(this->num_), points_saliency_data);
  
  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
    bias_n_diff = bias_n->cpu_diff();
    bias_n_ddiff = bias_n->cpu_ddiff();
    bias_saliency_data = bias_n->mutable_cpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_cpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_mul(this->blobs_[1]->count(), bias, bias_n_ddiff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_scal(bias_n->count(), 1/(Dtype)(2*(this->num_)), bias_saliency_data);
    caffe_sub(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    for (int n = 0; n<this->num_; n++) {
      caffe_mul(this->blobs_[1]->count(), bias, bias_saliency_data + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_scal(bias_n->count(), (Dtype)(this->num_), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_cpu(weights_n->count(2, 5), points_saliency_data, taylor_2nd, bias_saliency_data);
}

template void ConvolutionSaliencyLayer<float>::compute_taylor_2nd_cpu(const float *  act_data, const float * act_diff, const float *  act_ddiff, float * taylor_2nd);
template void ConvolutionSaliencyLayer<double>::compute_taylor_2nd_cpu(const double *  act_data, const double * act_diff, const double *  act_ddiff, double * taylor_2nd);

template void ConvolutionSaliencyLayer<float>::compute_taylor_2nd_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, float * taylor_2nd);
template void ConvolutionSaliencyLayer<double>::compute_taylor_2nd_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, double * taylor_2nd);
}  // namespace caffe