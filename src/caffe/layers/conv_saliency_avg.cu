#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PositiveCount(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? 1 : 0;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_weight_avg_gpu(const Dtype * bottom_data, const Dtype *  top_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_in, Dtype * saliency_info_out) {
  Dtype* output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_){
    output_saliency_data = output_saliencies_points_.mutable_gpu_data();
    caffe_copy(output_saliencies_points_.count(), top_data, output_saliency_data);
  }
  if (this->input_channel_saliency_compute_){
    input_saliency_data = input_saliencies_points_.mutable_gpu_data();
    caffe_copy(input_saliencies_points_.count(), bottom_data, input_saliency_data);
  }
  compute_norm_and_batch_avg_gpu(input_saliency_data, output_saliency_data, saliency_norm_, saliency_info_in, saliency_info_out);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_weight_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_in, Dtype * saliency_info_out) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();

  const Dtype* bias;
  Dtype* bias_saliency_data;

  int kernel_size = this->blobs_[0]->count(2,4);
  int weights_count = this->blobs_[0]->count();
  int bias_count;

  if (this->bias_term_) {
    bias_count = this->blobs_[1]->count();
  }

  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }

  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    bias_saliency_data = bias_n->mutable_gpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_gpu_data();
    }
  }

  switch (saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_gpu_abs(weights_count, weights, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_abs(bias_count, bias, bias_saliency_data);
      }
    } break;

    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_gpu_powx(weights_count, weights, (Dtype) 2, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_powx(bias_count, bias, (Dtype) 2, bias_saliency_data);
      }
    } break;

    default: {
      caffe_copy(weights_count, weights, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_copy(bias_count, bias, bias_saliency_data);
      }
    } break;
  }
  if (this->output_channel_saliency_compute_) {
    caffe_gpu_sum(this->num_output_, this->blobs_[0]->count(1,4), points_saliency_data, saliency_info_out);
    if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
      caffe_gpu_add(this->num_output_, bias_saliency_data, saliency_info_out, saliency_info_out);
    }
  }
  if (this->input_channel_saliency_compute_) {
    caffe_gpu_sum(this->num_output_ * this->channels_ / this->group_, kernel_size, points_saliency_data, points_saliency_data);
    caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_output_, points_saliency_data, saliency_info_in);
  }

  switch (saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
      if (this->output_channel_saliency_compute_) {
        caffe_gpu_abs(this->num_output_, saliency_info_out, saliency_info_out);
      }
      if (this->input_channel_saliency_compute_) {
        caffe_gpu_abs(this->channels_ / this->group_, saliency_info_in, saliency_info_in);
      }
    } break;

    case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
      if (this->output_channel_saliency_compute_) {
        caffe_gpu_powx(this->num_output_, saliency_info_out, (Dtype) 2, saliency_info_out);
      }
      if (this->input_channel_saliency_compute_) {
        caffe_gpu_powx(this->channels_ / this->group_, saliency_info_in, (Dtype) 2, saliency_info_in);
      }
    } break;

    default: {
    } break;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_apoz_gpu(const Dtype * bottom_data, const Dtype *  top_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_in, Dtype * saliency_info_out) {
  Dtype* output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_){
    output_saliency_data = output_saliencies_points_.mutable_gpu_data();
    PositiveCount<Dtype><<<CAFFE_GET_BLOCKS(output_saliencies_points_.count()), CAFFE_CUDA_NUM_THREADS>>>( 
        output_saliencies_points_.count(), top_data, output_saliency_data);
  }
  if (this->input_channel_saliency_compute_){
    input_saliency_data = input_saliencies_points_.mutable_gpu_data();
    PositiveCount<Dtype><<<CAFFE_GET_BLOCKS(input_saliencies_points_.count()), CAFFE_CUDA_NUM_THREADS>>>( 
        input_saliencies_points_.count(), bottom_data, input_saliency_data);
  }
  compute_norm_and_batch_avg_gpu(input_saliency_data, output_saliency_data, saliency_norm_, saliency_info_in, saliency_info_out);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_apoz_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_in, Dtype * saliency_info_out) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();

  const Dtype* bias;
  Dtype* bias_saliency_data;

  int kernel_size = this->blobs_[0]->count(2,4);
  int weights_count = this->blobs_[0]->count();
  int bias_count;

  if (this->bias_term_) {
    bias_count = this->blobs_[1]->count();
  }

  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }

  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    bias_saliency_data = bias_n->mutable_gpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_gpu_data();
    }
  }

  PositiveCount<Dtype><<<CAFFE_GET_BLOCKS(weights_count), CAFFE_CUDA_NUM_THREADS>>>( 
      weights_count, weights, points_saliency_data);
  if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL) {
    PositiveCount<Dtype><<<CAFFE_GET_BLOCKS(bias_count), CAFFE_CUDA_NUM_THREADS>>>( 
        bias_count, bias, bias_saliency_data);
  }
  CUDA_POST_KERNEL_CHECK;

  if (this->output_channel_saliency_compute_) {
    caffe_gpu_sum(this->num_output_, this->blobs_[0]->count(1,4), points_saliency_data, saliency_info_out);
    if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
      caffe_gpu_add(this->num_output_, bias_saliency_data, saliency_info_out, saliency_info_out);
    }
  }
  if (this->input_channel_saliency_compute_) {
    caffe_gpu_sum(this->num_output_ * this->channels_ / this->group_, kernel_size, points_saliency_data, points_saliency_data);
    caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_output_, points_saliency_data, saliency_info_in);
  }

  switch (saliency_norm_) {

    case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
      if (this->output_channel_saliency_compute_) {
        caffe_gpu_powx(this->num_output_, saliency_info_out, (Dtype) 2, saliency_info_out);
      }
      if (this->input_channel_saliency_compute_) {
        caffe_gpu_powx(this->channels_ / this->group_, saliency_info_in, (Dtype) 2, saliency_info_in);
      }
    } break;

    default: {
    } break;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_diff_avg_gpu(const Dtype * bottom_diff, const Dtype * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_in, Dtype * saliency_info_out) {
  Dtype* output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_){
    output_saliency_data = output_saliencies_points_.mutable_gpu_data();
    caffe_copy(output_saliencies_points_.count(), top_diff, output_saliency_data);
    caffe_gpu_scal(output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data);
  }
  if (this->input_channel_saliency_compute_){
    input_saliency_data = input_saliencies_points_.mutable_gpu_data();
    caffe_copy(input_saliencies_points_.count(), bottom_diff, input_saliency_data);
    caffe_gpu_scal(input_saliencies_points_.count(), (Dtype) this->num_, input_saliency_data);
  }

  compute_norm_and_batch_avg_gpu(input_saliency_data, output_saliency_data, saliency_norm_, saliency_info_in, saliency_info_out);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_diff_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_in, Dtype * saliency_info_out) {
  const Dtype* weights_n_diff = weights_n->gpu_diff();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();

  const Dtype* bias_n_diff;
  Dtype* bias_saliency_data;

  caffe_copy(weights_n->count(), weights_n_diff, points_saliency_data);
  caffe_gpu_scal(weights_n->count(), (Dtype) this->num_, points_saliency_data);

  if (this->saliency_bias_ && this->bias_term_) {
    bias_n_diff = bias_n->gpu_diff();
    bias_saliency_data = bias_n->mutable_gpu_data();
    caffe_copy(bias_n->count(), bias_n_diff, bias_saliency_data);
    caffe_gpu_scal(bias_n->count(), (Dtype) this->num_, bias_saliency_data);
  }

  compute_norm_and_batch_avg_weights_gpu(points_saliency_data, bias_saliency_data, saliency_norm_, saliency_info_in, saliency_info_out);
}

template void ConvolutionLayer<float>::compute_weight_avg_gpu(const float * bottom_data, const float *  top_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * saliency_info_in, float * saliency_info_out);
template void ConvolutionLayer<double>::compute_weight_avg_gpu(const double * bottom_data, const double *  top_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * saliency_info_in, double * saliency_info_out);

template void ConvolutionLayer<float>::compute_weight_avg_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * saliency_info_in, float * saliency_info_out);
template void ConvolutionLayer<double>::compute_weight_avg_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * saliency_info_in, double * saliency_info_out);

template void ConvolutionLayer<float>::compute_apoz_gpu(const float * bottom_data, const float *  top_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * saliency_info_in, float * saliency_info_out);
template void ConvolutionLayer<double>::compute_apoz_gpu(const double * bottom_data, const double *  top_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * saliency_info_in, double * saliency_info_out);

template void ConvolutionLayer<float>::compute_apoz_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * saliency_info_in, float * saliency_info_out);
template void ConvolutionLayer<double>::compute_apoz_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * saliency_info_in, double * saliency_info_out);

template void ConvolutionLayer<float>::compute_diff_avg_gpu(const float * bottom_diff, const float * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * saliency_info_in, float * saliency_info_out);
template void ConvolutionLayer<double>::compute_diff_avg_gpu(const double * bottom_diff, const double * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * saliency_info_in, double * saliency_info_out);

template void ConvolutionLayer<float>::compute_diff_avg_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * saliency_info_in, float * saliency_info_out);
template void ConvolutionLayer<double>::compute_diff_avg_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * saliency_info_in, double * saliency_info_out);

}  // namespace caffe
