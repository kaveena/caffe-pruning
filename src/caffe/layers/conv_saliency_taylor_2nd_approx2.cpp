#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void taylor_2nd_approx2_kernel_cpu(const int N, const int num, const Dtype * data, const Dtype * diff, Dtype * taylor_2nd) {
#if USE_OPENMP
  #pragma omp parallel
  #pragma omp for
#endif
  for (int i=0; i<N; i++) {
    taylor_2nd[i] = ( 0.5 * num * num * data[i] * data[i] * diff[i] * diff[i] ) - (num * data[i] * diff[i]);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_approx2_cpu(const Dtype * bottom_data, const Dtype * bottom_diff, const Dtype *  top_data, const Dtype * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_in, Dtype * taylor_2nd_out) {
  Dtype * output_saliency_data = NULL;
  Dtype * input_saliency_data = NULL;
  if (this->output_channel_saliency_compute_){
    output_saliency_data = output_saliencies_points_.mutable_cpu_data();
    taylor_2nd_approx2_kernel_cpu<Dtype>(
        output_saliencies_points_.count(), this->num_, top_data, top_diff, output_saliency_data);
  }
  if (this->input_channel_saliency_compute_){
    input_saliency_data = input_saliencies_points_.mutable_cpu_data();
    taylor_2nd_approx2_kernel_cpu<Dtype>(
        input_saliencies_points_.count(), this->num_, bottom_data, bottom_diff, input_saliency_data);
  }
  compute_norm_and_batch_avg_cpu(input_saliency_data, output_saliency_data, saliency_norm_, taylor_2nd_in, taylor_2nd_out);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_approx2_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_in, Dtype * taylor_2nd_out) {
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
    caffe_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //a * 1/N *dE/da
  }

  caffe_mul(weights_n->count(), points_saliency_data, weights_n_diff, points_saliency_data); //a * 1/N *  1/N * (dE/da)**2
  caffe_scal(weights_n->count(), (Dtype)(this->num_ * 0.5), points_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_sub(weights_n->count(), points_saliency_data, weights_n_diff, points_saliency_data); //(a/2N * (dE/da)**2) - 1/N * dE/da
  for (int n = 0; n<this->num_; n++) {
    caffe_mul(this->blobs_[0]->count(), weights, points_saliency_data + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //(a**2/2N * (dE/da)**2) - a/N*dE/da
  }
  caffe_scal(weights_n->count(), (Dtype)(this->num_), points_saliency_data);

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
    caffe_mul(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    caffe_scal(bias_n->count(), (Dtype)(this->num_ * 0.5), bias_saliency_data);
    caffe_sub(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    for (int n = 0; n<this->num_; n++) {
      caffe_mul(this->blobs_[1]->count(), bias, bias_saliency_data + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_scal(bias_n->count(), (Dtype)(this->num_), bias_saliency_data);
  }

  compute_norm_and_batch_avg_weights_cpu(points_saliency_data, bias_saliency_data, saliency_norm_, taylor_2nd_in, taylor_2nd_out);
}

template void ConvolutionLayer<float>::compute_taylor_2nd_approx2_cpu(const float * bottom_data, const float * bottom_diff, const float *  top_data, const float * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * taylor_2nd_in, float * taylor_2nd_out);
template void ConvolutionLayer<double>::compute_taylor_2nd_approx2_cpu(const double * bottom_data, const double * bottom_diff, const double *  top_data, const double * top_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * taylor_2nd_in, double * taylor_2nd_out);

template void ConvolutionLayer<float>::compute_taylor_2nd_approx2_weights_cpu(Blob<float> * weights_n, Blob<float> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * taylor_2nd_in, float * taylor_2nd_out);
template void ConvolutionLayer<double>::compute_taylor_2nd_approx2_weights_cpu(Blob<double> * weights_n, Blob<double> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * taylor_2nd_in, double * taylor_2nd_out);
}  // namespace caffe
