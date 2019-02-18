#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReduceNMCKK(const int N, const int M, const int C, const int K, const Dtype * a, Dtype * y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N*C) {
    int n = i / C;
    int c = i % C;
    Dtype accum = (Dtype) 0;
    for (int m = 0; m < M; m++) {
      for (int k = 0; k < K; k++) {
        accum += a[(n*M*C*K) + (m*C*K) + (c*K) + k];
      }
    }
    y[i] = accum;
  } 
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_gpu(const Dtype *  act_data, const Dtype *  act_diff, const Dtype * input_data, const Dtype * input_diff,  Dtype * fisher_info_out, Dtype * fisher_info_in) {
  if (this->output_channel_saliency_compute_) {
    Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data(); 
    Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
    
    caffe_gpu_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
    caffe_gpu_scal(this->output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data); //get unscaled diff back
    caffe_gpu_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
    caffe_gpu_powx(this->output_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);
    caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, fisher_info_out);
    caffe_gpu_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info_out);
  }

  if (this->input_channel_saliency_compute_) {
    Dtype* input_saliency_data = input_saliencies_points_.mutable_gpu_data(); 
    Dtype* filter_saliency_data = input_saliencies_filter_.mutable_gpu_data();    
    
    caffe_gpu_mul(this->input_saliencies_points_.count(), input_data, input_diff, input_saliency_data);
    caffe_gpu_scal(this->input_saliencies_points_.count(), (Dtype) this->num_, input_saliency_data); //get unscaled diff back
    caffe_gpu_sum(this->input_saliencies_points_.count(0, 2), input_saliencies_points_.count(2,4), input_saliency_data, filter_saliency_data); //sum hxw
    caffe_gpu_powx(this->input_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);
    caffe_gpu_strided_sum(this->channels_, this->num_, filter_saliency_data, filter_saliency_data);
    caffe_gpu_strided_sum(this->channels_ / this->group_, this->group_, filter_saliency_data, fisher_info_in);
    caffe_gpu_scal(this->channels_, 1/(Dtype)(2*(this->num_)), fisher_info_in);
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info_out, Dtype * fisher_info_in) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  const Dtype* weights_n_diff = weights_n->gpu_diff();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();
  
  const Dtype* bias;
  const Dtype* bias_n_diff;
  Dtype* bias_saliency_data;
  
  Dtype* filter_out_saliency_data;
  Dtype* filter_in_saliency_data;
  
  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }

  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
  caffe_gpu_scal(weights_n->count(), (Dtype) this->num_, points_saliency_data); // get unscaled diff back
  
  if (this->output_channel_saliency_compute_) {
    filter_out_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
    caffe_gpu_sum(weights_n->count(0,2), weights_n->count(2, 5), points_saliency_data, filter_out_saliency_data);
    
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
      caffe_gpu_scal(bias_n->count(), (Dtype) this->num_, bias_saliency_data); // get unscaled diff back
      caffe_gpu_add(weights_n->count(0,2), filter_out_saliency_data, bias_saliency_data, filter_out_saliency_data);
    }
    
    caffe_gpu_powx(output_saliencies_filter_.count(), filter_out_saliency_data, (Dtype)2, filter_out_saliency_data);
    caffe_gpu_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, fisher_info_out);
    caffe_gpu_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info_out);
  }

  if (this->input_channel_saliency_compute_) {
    filter_in_saliency_data = input_saliencies_filter_.mutable_gpu_data();    
    const int kernel_size = this->blobs_[0]->count(2,4);
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReduceNMCKK<Dtype><<<(this->num_ * this->channels_ / this->group_),CAFFE_CUDA_NUM_THREADS>>>(this->num_, this->num_output_, this->channels_ / this->group_, kernel_size, points_saliency_data, filter_in_saliency_data);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_powx(this->num_ * this->channels_ / this->group_, filter_in_saliency_data, (Dtype)2, filter_in_saliency_data);
    caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_, filter_in_saliency_data, fisher_info_in);
    caffe_gpu_scal(this->channels_ / this->group_, 1/(Dtype)(2*(this->num_)), fisher_info_in);
  }
}

template void ConvolutionSaliencyLayer<float>::compute_fisher_gpu(const float *  act_data, const float *  act_diff, const float * input_data, const float * input_diff,  float * fisher_info_out, float * fisher_info_in);
template void ConvolutionSaliencyLayer<double>::compute_fisher_gpu(const double *  act_data, const double *  act_diff, const double * input_data, const double * input_diff,  double * fisher_info_out, double * fisher_info_in);

template void ConvolutionSaliencyLayer<float>::compute_fisher_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, float * fisher_info_out, float * fisher_info_in);
template void ConvolutionSaliencyLayer<double>::compute_fisher_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, double * fisher_info_out, double * fisher_info_in);
}  // namespace caffe
