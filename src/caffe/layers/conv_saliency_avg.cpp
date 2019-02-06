#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_weight_avg_cpu(const Dtype *  act_data, Dtype * saliency_info) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_copy(output_saliencies_points_.count(), act_data, output_saliency_data);

  compute_norm_and_batch_avg_cpu(output_saliencies_points_.count(2,4), output_saliency_data, saliency_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_weight_avg_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info) {
  const Dtype* weights = this->blobs_[0]->cpu_data();
  Dtype* points_saliency_data = weights_n->mutable_cpu_data();
  
  const Dtype* bias;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.cpu_data();
  }
    
  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
    bias_saliency_data = bias_n->mutable_cpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_cpu_data();
    }
  }
  
  switch (this->saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_abs(this->blobs_[0]->count(), weights, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_abs(this->blobs_[1]->count(), bias, bias_saliency_data);
      }
    } break;
    
    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_powx(this->blobs_[0]->count(), weights, (Dtype) 2, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_powx(this->blobs_[1]->count(), bias, (Dtype) 2, bias_saliency_data);
      }
    } break;
  
    default: {
      caffe_copy(this->blobs_[0]->count(), weights, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_copy(this->blobs_[1]->count(), bias, bias_saliency_data);
      }
    } break;
  }
  caffe_sum(this->num_output_, this->blobs_[0]->count(1,4), points_saliency_data, saliency_info); //sum hxw
  if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
    caffe_add(this->num_output_, bias_saliency_data, saliency_info, saliency_info);
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_diff_avg_cpu(const Dtype *  act_diff, Dtype * saliency_info) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_copy(output_saliencies_points_.count(), act_diff, output_saliency_data);
  caffe_scal(output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data);

  compute_norm_and_batch_avg_cpu(output_saliencies_points_.count(2,4), output_saliency_data, saliency_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_diff_avg_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info) {
  const Dtype* weights_n_diff = weights_n->cpu_diff();
  Dtype* points_saliency_data = weights_n->mutable_cpu_data();
  
  const Dtype* bias_n_diff;
  Dtype* bias_saliency_data;

  caffe_copy(weights_n->count(), weights_n_diff, points_saliency_data);
  caffe_scal(weights_n->count(), (Dtype) this->num_, points_saliency_data);

  if (this->saliency_bias_ && this->bias_term_) {
    bias_n_diff = bias_n->cpu_diff();
    bias_saliency_data = bias_n->mutable_cpu_data();
    caffe_copy(bias_n->count(), bias_n_diff, bias_saliency_data);
    caffe_scal(bias_n->count(), (Dtype) this->num_, bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_cpu(weights_n->count(2, 5), points_saliency_data, saliency_info, bias_saliency_data);
}

template void ConvolutionSaliencyLayer<float>::compute_weight_avg_cpu(const float *  act_data, float * saliency_info);
template void ConvolutionSaliencyLayer<double>::compute_weight_avg_cpu(const double *  act_data, double * saliency_info);

template void ConvolutionSaliencyLayer<float>::compute_weight_avg_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, float * saliency_info);
template void ConvolutionSaliencyLayer<double>::compute_weight_avg_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, double * saliency_info);

template void ConvolutionSaliencyLayer<float>::compute_diff_avg_cpu(const float *  act_diff, float * saliency_info);
template void ConvolutionSaliencyLayer<double>::compute_diff_avg_cpu(const double *  act_diff, double * saliency_info);

template void ConvolutionSaliencyLayer<float>::compute_diff_avg_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, float * saliency_info);
template void ConvolutionSaliencyLayer<double>::compute_diff_avg_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, double * saliency_info);

}  // namespace caffe
