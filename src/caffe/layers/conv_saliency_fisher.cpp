#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_cpu(const Dtype *  act_data, const Dtype *  act_diff, const Dtype * input_data, const Dtype * input_diff,  Dtype * fisher_info_out, Dtype * fisher_info_in) {
  if (this->output_channel_saliency_compute_) {
    Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data(); 
    Dtype* filter_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
    
    caffe_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
    caffe_scal(this->output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data); //get unscaled diff back
    caffe_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
    caffe_powx(this->output_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);
    caffe_strided_sum(this->num_output_, this->num_, filter_saliency_data, fisher_info_out);
    caffe_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info_out);
  }

  if (this->input_channel_saliency_compute_) {
    Dtype* input_saliency_data = input_saliencies_points_.mutable_cpu_data(); 
    Dtype* filter_saliency_data = input_saliencies_filter_.mutable_cpu_data();    
    
    caffe_mul(this->input_saliencies_points_.count(), input_data, input_diff, input_saliency_data);
    caffe_scal(this->input_saliencies_points_.count(), (Dtype) this->num_, input_saliency_data); //get unscaled diff back
    caffe_sum(this->input_saliencies_points_.count(0, 2), input_saliencies_points_.count(2,4), input_saliency_data, filter_saliency_data); //sum hxw
    caffe_powx(this->input_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);
    caffe_strided_sum(this->channels_, this->num_, filter_saliency_data, filter_saliency_data);
    caffe_strided_sum(this->channels_ / this->group_, this->group_, filter_saliency_data, fisher_info_in);
    caffe_scal(this->channels_, 1/(Dtype)(2*(this->num_)), fisher_info_in);
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info_out, Dtype * fisher_info_in) {
  const Dtype* weights = this->blobs_[0]->cpu_data();
  const Dtype* weights_n_diff = weights_n->cpu_diff();
  Dtype* points_saliency_data = weights_n->mutable_cpu_data();
  
  const Dtype* bias;
  const Dtype* bias_n_diff;
  Dtype* bias_saliency_data;
  
  Dtype* filter_out_saliency_data;
  Dtype* filter_in_saliency_data;
  
  if (this->mask_term_) {
    weights = weights_masked_.cpu_data();
  }

  for (int n = 0; n<this->num_; n++) {
    caffe_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
  caffe_scal(weights_n->count(), (Dtype) this->num_, points_saliency_data); // get unscaled diff back
  
  if (this->output_channel_saliency_compute_) {
    filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
    caffe_sum(weights_n->count(0,2), weights_n->count(2, 5), points_saliency_data, filter_out_saliency_data);
    
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
      caffe_scal(bias_n->count(), (Dtype) this->num_, bias_saliency_data); // get unscaled diff back
      caffe_add(weights_n->count(0,2), filter_out_saliency_data, bias_saliency_data, filter_out_saliency_data);
    }
    
    caffe_powx(output_saliencies_filter_.count(), filter_out_saliency_data, (Dtype)2, filter_out_saliency_data);
    caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, fisher_info_out);
    caffe_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info_out);
  }

  if (this->input_channel_saliency_compute_) {
    filter_in_saliency_data = input_saliencies_filter_.mutable_cpu_data();    
    caffe_sum(weights_n->count(0,3), weights_n->count(3, 5), points_saliency_data, points_saliency_data);
    caffe_strided_sum_inner(this->num_, this->num_output_, this->channels_, points_saliency_data, filter_in_saliency_data);
    caffe_powx(input_saliencies_filter_.count(), filter_in_saliency_data, (Dtype)2, filter_in_saliency_data);
    caffe_strided_sum(this->channels_, this->num_, filter_in_saliency_data, filter_in_saliency_data);
    caffe_strided_sum(this->channels_ / this->group_, this->group_, filter_in_saliency_data, fisher_info_in);
    caffe_scal(this->channels_, 1/(Dtype)(2*(this->num_)), fisher_info_in);
  }
}

template void ConvolutionSaliencyLayer<float>::compute_fisher_cpu(const float *  act_data, const float *  act_diff, const float * input_data, const float * input_diff,  float * fisher_info_out, float * fisher_info_in);
template void ConvolutionSaliencyLayer<double>::compute_fisher_cpu(const double *  act_data, const double *  act_diff, const double * input_data, const double * input_diff,  double * fisher_info_out, double * fisher_info_in);

template void ConvolutionSaliencyLayer<float>::compute_fisher_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, float * fisher_info_out, float * fisher_info_in);
template void ConvolutionSaliencyLayer<double>::compute_fisher_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, double * fisher_info_out, double * fisher_info_in);
}  // namespace caffe
