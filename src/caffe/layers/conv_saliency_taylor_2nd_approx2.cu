#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void taylor_2nd_approx2_kernel_gpu(const int N, const int num, const Dtype * data, const Dtype * diff, Dtype * taylor_2nd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    taylor_2nd[i] = ( 0.5 * num * num * data[i] * data[i] * diff[i] * diff[i] ) - (num * data[i] * diff[i]);
  }   
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_approx2_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype * input_data, const Dtype * input_diff, Dtype * taylor_2nd_out, Dtype * taylor_2nd_in) {
  Dtype * output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_) {
    output_saliency_data = output_saliencies_points_.mutable_gpu_data(); 
    taylor_2nd_approx2_kernel_gpu<Dtype><<<CAFFE_GET_BLOCKS(output_saliencies_points_.count()), CAFFE_CUDA_NUM_THREADS>>>(
        output_saliencies_points_.count(), this->num_, act_data, act_diff, output_saliency_data);
  }

  if (this->input_channel_saliency_compute_) {
    input_saliency_data = input_saliencies_points_.mutable_gpu_data(); 
    taylor_2nd_approx2_kernel_gpu<Dtype><<<CAFFE_GET_BLOCKS(input_saliencies_points_.count()), CAFFE_CUDA_NUM_THREADS>>>(
        input_saliencies_points_.count(), this->num_, input_data, input_diff, input_saliency_data);
  }

  CUDA_POST_KERNEL_CHECK;
  compute_norm_and_batch_avg_gpu(output_saliency_data, input_saliency_data, taylor_2nd_out, taylor_2nd_in);
  
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_2nd_out, Dtype *taylor_2nd_in) {
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
  caffe_gpu_scal(weights_n->count(), (Dtype)(this->num_ * 0.5), points_saliency_data);  //1/2N * (a * d2E/da2)
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
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_ * 0.5), bias_saliency_data);
    caffe_gpu_sub(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_saliency_data + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_), bias_saliency_data);
  }

  compute_norm_and_batch_avg_weights_gpu(points_saliency_data, bias_saliency_data, taylor_2nd_out, taylor_2nd_in);
}

template void ConvolutionLayer<float>::compute_taylor_2nd_approx2_gpu(const float *  act_data, const float * act_diff, const float * input_data, const float * input_diff, float * taylor_2nd_out, float * taylor_2nd_in);
template void ConvolutionLayer<double>::compute_taylor_2nd_approx2_gpu(const double *  act_data, const double * act_diff, const double * input_data, const double * input_diff, double * taylor_2nd_out, double * taylor_2nd_in);

template void ConvolutionLayer<float>::compute_taylor_2nd_approx2_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, float * taylor_2nd_out, float *taylor_2nd_in);
template void ConvolutionLayer<double>::compute_taylor_2nd_approx2_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, double * taylor_2nd_out, double *taylor_2nd_in);
}  // namespace caffe
