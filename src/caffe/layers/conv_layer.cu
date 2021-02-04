#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {


template <typename Dtype>
__global__ void reduce_nmckk_kernel(const int N, const int M, const int C, const int K, const Dtype * a, Dtype * y) {
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
void reduce_nmckk_gpu(const int N, const int M, const int C, const int K, const Dtype * a, Dtype * y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  reduce_nmckk_kernel<Dtype><<<CAFFE_GET_BLOCKS(N*C), CAFFE_CUDA_NUM_THREADS>>>(
      N, M, C, K, a, y);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias;
  LayerParameter layer_param(this->layer_param_);
  if (layer_param.phase() == caffe::TRAIN) {
    this->quantize_clock_ += 1;
    this->activation_quantize_clock_ += 1;
  }
  if (this->quantize_term_) {
    Dtype* weight_masked = this->weights_masked_.mutable_gpu_data();
    caffe_gpu_and(this->blobs_[0]->count(), this->quantization_mask, weight, weight_masked);

    if (this->quantize_clock_ == this->quantize_interval_) {
      LOG(INFO) << "Quantizing weights";
      Dtype* weight_mut = this->blobs_[0]->mutable_gpu_data();
      caffe_copy(this->blobs_[0]->count(), weight_masked, weight_mut);
    }
    weight = this->weights_masked_.gpu_data();
  }
  if (this->mask_term_) {
    const Dtype* mask = this->blobs_[this->mask_pos_]->gpu_data();
    Dtype* weight_masked = this->weights_masked_.mutable_gpu_data();
    caffe_gpu_mul(this->blobs_[0]->count(), mask, weight, weight_masked);
    weight = this->weights_masked_.gpu_data();
  }
  if (this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    if (this->quantize_term_) {
      Dtype* bias_masked = this->bias_masked_.mutable_gpu_data();
      caffe_gpu_and(this->blobs_[1]->count(), this->quantization_mask, bias, bias_masked);
      if (this->quantize_clock_ == this->quantize_interval_) {
        LOG(INFO) << "Quantizing biases";
        Dtype* bias_mut = this->blobs_[1]->mutable_gpu_data();
        caffe_copy(this->blobs_[1]->count(), bias_masked, bias_mut);
      }
      bias = this->bias_masked_.gpu_data();
    }
    if (this->mask_term_) {
      const Dtype* bias_mask = this->blobs_[this->mask_pos_+1]->gpu_data();
      Dtype* bias_masked = this->bias_masked_.mutable_gpu_data();
      caffe_gpu_mul(this->blobs_[1]->count(), bias_mask, bias, bias_masked);
      bias = this->bias_masked_.gpu_data();
    }
  }
  for (int i = 0; i < bottom.size(); ++i) {
    Dtype* bottom_data = bottom[i]->mutable_gpu_data();
    if (this->activation_quantize_term_ && (this->activation_quantize_clock_ == this->activation_quantize_interval_)) {
      caffe_gpu_and(bottom[i]->count(), this->activation_quantization_mask, bottom_data, bottom_data);
    }
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
  if ((layer_param.phase() == caffe::TRAIN) && (this->quantize_clock_ >= this->quantize_interval_)) {
    this->quantize_clock_ = 0;
  }
  if ((layer_param.phase() == caffe::TRAIN) && (this->activation_quantize_clock_ >= this->activation_quantize_interval_)) {
    this->activation_quantize_clock_ = 0;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* weights_sqr = this->weights_sqr_.mutable_gpu_data();
  Blob<Dtype>  weights_n_masked_;
  Blob<Dtype> bias_n_masked_;
  Blob<Dtype> input_shaped_blob_;
  Dtype* full_weights_diff;

  Dtype* weight_ddiff;
  Dtype* full_weights_ddiff;

  Dtype* bias_diff;
  Dtype* full_bias_diff;
  Dtype* bias_ddiff;
  Dtype* full_bias_ddiff;

  ConvolutionSaliencyParameter conv_saliency_param = this->layer_param_.convolution_saliency_param();

  if (this->saliency_term_ && this->separate_weight_diff_) {
    weights_n_masked_.Reshape({this->num_, this->blobs_[0]->shape()[0], this->blobs_[0]->shape()[1], this->blobs_[0]->shape()[2], this->blobs_[0]->shape()[3]});
    full_weights_diff = weights_n_masked_.mutable_gpu_diff();
  }

  if (this->mask_term_) {
    weight = this->weights_masked_.gpu_data();
  }

  if (this->saliency_term_) {
    if (this->layer_param_.compute_2nd_derivative()) {
      weight_ddiff = this->blobs_[0]->mutable_gpu_diff();
      if (this->separate_weight_diff_) {
        full_weights_ddiff = weights_n_masked_.mutable_gpu_ddiff();
      }
    }
  }

  if (this->bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();

    if (this->saliency_term_) {
      if (this->separate_weight_diff_) {
        bias_n_masked_.Reshape({this->num_, this->blobs_[1]->shape()[0]});
        full_bias_diff = bias_n_masked_.mutable_gpu_diff();
      }
    }

    if (this->layer_param_.compute_2nd_derivative()) {
      bias_ddiff = this->blobs_[1]->mutable_gpu_ddiff();
      if (this->separate_weight_diff_) {
        full_bias_ddiff = bias_n_masked_.mutable_gpu_ddiff();
      }
    }
  }

  caffe_gpu_powx(this->blobs_[0]->count(), weight, (Dtype)2, weights_sqr);

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* top_data = top[i]->gpu_data();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    const Dtype* top_ddiff;
    Dtype* bottom_ddiff;
    Dtype* input_sqr_;
    if (this->layer_param_.compute_2nd_derivative()) {
      input_shaped_blob_.Reshape(bottom[i]->shape());
      top_ddiff = top[i]->gpu_ddiff();
      bottom_ddiff = bottom[i]->mutable_gpu_ddiff();
      weight_ddiff = this->blobs_[0]->mutable_gpu_ddiff();
      if (this->separate_weight_diff_) {
        full_weights_ddiff = weights_n_masked_.mutable_gpu_ddiff();
      }
      input_sqr_ = input_shaped_blob_.mutable_gpu_data();
      caffe_gpu_powx(bottom[i]->count(), bottom[i]->gpu_data(), (Dtype) 2, input_sqr_);
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < this->num_; ++n) {
        if (this->separate_weight_diff_) {
          this->backward_gpu_bias_no_accum(full_bias_diff + n * this->blobs_[1]->count(), top_diff + n * this->top_dim_);
          caffe_gpu_add(this->blobs_[1]->count(), full_bias_diff + n * this->blobs_[1]->count(), bias_diff, bias_diff);
          if (this->layer_param_.compute_2nd_derivative()) {
            this->backward_gpu_bias_no_accum(full_bias_ddiff + n * this->blobs_[1]->count(), top_ddiff + n * this->top_dim_);
            caffe_gpu_add(this->blobs_[1]->count(), full_bias_ddiff + n * this->blobs_[1]->count(), bias_ddiff, bias_ddiff);
          }
        }
        else {
          this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
          if (this->layer_param_.compute_2nd_derivative()) {
            this->backward_gpu_bias(bias_ddiff, top_ddiff + n * this->top_dim_);
          }
        }
      }
      if (this->mask_term_) {
        caffe_gpu_mul(this->blobs_[1]->count(), this->blobs_[this->mask_pos_+1]->gpu_data(), bias_diff, bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          if (this->saliency_term_ && this->separate_weight_diff_) {
            this->weight_gpu_gemm_no_accum(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, full_weights_diff + n * this->blobs_[0]->count());
            caffe_gpu_add(this->blobs_[0]->count(), full_weights_diff + n * this->blobs_[0]->count(), weight_diff, weight_diff);
            if (this->layer_param_.compute_2nd_derivative()) {
              this->weight_gpu_gemm_no_accum(input_sqr_ + n * this->bottom_dim_,
                  top_ddiff + n * this->top_dim_, full_weights_ddiff + n * this->blobs_[0]->count());
              caffe_gpu_add(this->blobs_[0]->count(), full_weights_ddiff + n * this->blobs_[0]->count(), weight_ddiff, weight_ddiff);
            }
          }
          else {
            this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
            if (this->layer_param_.compute_2nd_derivative()) {
              this->weight_gpu_gemm(input_sqr_ + n * this->bottom_dim_,
                top_ddiff + n * this->top_dim_, weight_ddiff);
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (this->layer_param_.compute_2nd_derivative()) {
          if (propagate_down[i]) {
            this->backward_gpu_gemm(top_ddiff + n * this->top_dim_, weights_sqr,
                bottom_ddiff + n * this->bottom_dim_);
          }
        }
      }
      if (this->mask_term_) {
        // Don't update weights that are masked off
        caffe_gpu_mul(this->blobs_[0]->count(), this->blobs_[this->mask_pos_]->gpu_data(), weight_diff, weight_diff);
      }
    }

    // Compute Channel saliency
    // MULTIPLE INPUTS NOT TREATED
    if (this->saliency_term_) {
      int pos_output_channel_saliency = this->saliency_pos_;
      int pos_input_channel_saliency = this->saliency_pos_;
      if (this->output_channel_saliency_compute_ && this->input_channel_saliency_compute_){
        pos_input_channel_saliency++;
      }
      Dtype * output_channel_saliency_data = NULL;
      Dtype * output_channel_saliency_accum_data = NULL;
      Dtype * input_channel_saliency_data = NULL;
      Dtype * input_channel_saliency_accum_data = NULL;
      Dtype * out_channel_saliency = NULL;
      Dtype * in_channel_saliency = NULL;
      if (this->output_channel_saliency_compute_){
        output_channel_saliency_data = output_saliencies_channel_.mutable_gpu_data();
        output_channel_saliency_accum_data = this->blobs_[pos_output_channel_saliency]->mutable_gpu_data();
      }
      if (this->input_channel_saliency_compute_){
        input_channel_saliency_data = input_saliencies_channel_.mutable_gpu_data();
        input_channel_saliency_accum_data = this->blobs_[pos_input_channel_saliency]->mutable_gpu_data();
      }
      for (int i_s = 0; i_s < conv_saliency_param.saliency_size(); i_s++) {
        if (this->output_channel_saliency_compute_){
          out_channel_saliency = output_channel_saliency_data + (i_s * this->num_output_);
        }
        if (this->input_channel_saliency_compute_){
          in_channel_saliency = input_channel_saliency_data + (i_s * (this->channels_ / this->group_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_taylor_gpu(bottom_data, bottom_diff, top_data, top_diff, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_hessian_diag_gpu(bottom_data, bottom_ddiff, top_data, top_ddiff, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_hessian_diag_approx2_gpu(bottom_data, bottom_diff, top_data, top_diff, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_taylor_2nd_gpu(bottom_data, bottom_diff, bottom_ddiff, top_data, top_diff, top_ddiff, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_taylor_2nd_approx2_gpu(bottom_data, bottom_diff, top_data, top_diff, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_INPUT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_weight_avg_gpu(bottom_data, top_data, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::APOZ) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_apoz_gpu(bottom_data, top_data, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_GRADIENT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_diff_avg_gpu(bottom_diff, top_diff, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_taylor_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_hessian_diag_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_hessian_diag_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_taylor_2nd_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_taylor_2nd_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_INPUT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_weight_avg_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::APOZ) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_apoz_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_GRADIENT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_diff_avg_weights_gpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), in_channel_saliency, out_channel_saliency);
        }
      }
      if (this->layer_param_.convolution_saliency_param().accum()) {
        if (this->output_channel_saliency_compute_) {
          caffe_gpu_add(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data, output_channel_saliency_accum_data);
        }
        if (this->input_channel_saliency_compute_) {
          caffe_gpu_add(input_saliencies_channel_.count(), input_channel_saliency_data, input_channel_saliency_accum_data, input_channel_saliency_accum_data);
        }
      }
      else {
        if (this->output_channel_saliency_compute_) {
          caffe_copy(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data);
        }
        if (this->input_channel_saliency_compute_) {
          caffe_copy(input_saliencies_channel_.count(), input_channel_saliency_data, input_channel_saliency_accum_data);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_gpu(Dtype * in_saliency_data, Dtype * out_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * in_channel_saliency, Dtype * out_channel_saliency) {
  int count, channels;
  Dtype* filter_data, *saliency_data, *channel_saliency;
  if (this->input_channel_saliency_compute_){
    saliency_data = in_saliency_data;
    channel_saliency = in_channel_saliency;
    count = this->input_saliencies_points_.count(2,4);
    filter_data = this->input_saliencies_filter_.mutable_gpu_data();
    channels = this->channels_;
    switch (saliency_norm_) {
      case (caffe::ConvolutionSaliencyParameter::L1): {
        caffe_gpu_abs(this->num_ * channels * count, saliency_data, saliency_data);
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_strided_sum(channels, this->num_, filter_data, filter_data);
    } break;

      case (caffe::ConvolutionSaliencyParameter::L2): {
        caffe_gpu_powx(this->num_ * channels * count, saliency_data, (Dtype) 2, saliency_data);
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_strided_sum(channels, this->num_, filter_data, filter_data);
      } break;

      case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_abs(this->num_ * channels, filter_data, filter_data);
        caffe_gpu_strided_sum(channels, this->num_, filter_data, filter_data);
      } break;

      case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_powx(this->num_ * channels, filter_data, (Dtype) 2, filter_data);
        caffe_gpu_strided_sum(channels, this->num_, filter_data, filter_data);
      } break;

      default: {
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_strided_sum(channels, this->num_, filter_data, filter_data);
      } break;
    }
    caffe_gpu_strided_sum(this->channels_ /this->group_, this->group_, filter_data, channel_saliency);
    caffe_gpu_scal(channels, (Dtype) 1.0 / (Dtype)(this->num_), channel_saliency);
  }
  if (this->output_channel_saliency_compute_){
    saliency_data = out_saliency_data;
    channel_saliency = out_channel_saliency;
    count = this->output_saliencies_points_.count(2,4);
    filter_data = this->output_saliencies_filter_.mutable_gpu_data();
    channels = this->num_output_;
    switch (saliency_norm_) {
      case (caffe::ConvolutionSaliencyParameter::L1): {
        caffe_gpu_abs(this->num_ * channels * count, saliency_data, saliency_data);
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_strided_sum(channels, this->num_, filter_data, channel_saliency);
    } break;

      case (caffe::ConvolutionSaliencyParameter::L2): {
        caffe_gpu_powx(this->num_ * channels * count, saliency_data, (Dtype) 2, saliency_data);
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_strided_sum(channels, this->num_, filter_data, channel_saliency);
      } break;

      case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_abs(this->num_ * channels, filter_data, filter_data);
        caffe_gpu_strided_sum(channels, this->num_, filter_data, channel_saliency);
      } break;

      case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_powx(this->num_ * channels, filter_data, (Dtype) 2, filter_data);
        caffe_gpu_strided_sum(channels, this->num_, filter_data, channel_saliency);
      } break;

      default: {
        caffe_gpu_sum(this->num_ * channels, count, saliency_data, filter_data); //sum hxw
        caffe_gpu_strided_sum(channels, this->num_, filter_data, channel_saliency);
      } break;
    }
    caffe_gpu_scal(channels, (Dtype) 1.0 / (Dtype)(this->num_), channel_saliency);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_weights_gpu(Dtype * weight_saliency_data, Dtype * bias_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * in_channel_saliency, Dtype * out_channel_saliency) {

  Dtype* filter_data;

  int kernel_size = this->blobs_[0]->count(2,4);
  int weights_count = this->blobs_[0]->count();
  int bias_count;

  if (this->bias_term_) {
    bias_count = this->blobs_[1]->count();
  }
  switch (saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_gpu_abs(this->num_ * weights_count, weight_saliency_data, weight_saliency_data);
      if (this->output_channel_saliency_compute_){
        filter_data = output_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_gpu_abs(this->num_ * bias_count, bias_saliency_data, bias_saliency_data);
          caffe_gpu_add(this->num_ * bias_count, bias_saliency_data, filter_data, filter_data);
        }
        caffe_gpu_strided_sum(this->num_output_, this->num_, filter_data, out_channel_saliency);
        caffe_gpu_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), out_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        filter_data = input_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_ * this->channels_ / this->group_, kernel_size, weight_saliency_data, weight_saliency_data);
        caffe_gpu_strided_sum_inner(this->num_, this->num_output_, this->channels_ / this->group_, weight_saliency_data, filter_data);
        caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_, filter_data, in_channel_saliency);
        caffe_gpu_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), in_channel_saliency);
      }
    } break;

    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_gpu_powx(this->num_ * weights_count, weight_saliency_data, (Dtype) 2, weight_saliency_data);
      if (this->output_channel_saliency_compute_){
        filter_data = output_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_gpu_powx(this->num_ * bias_count, bias_saliency_data, (Dtype) 2, bias_saliency_data);
          caffe_gpu_add(this->num_ * bias_count, bias_saliency_data, filter_data, filter_data);
        }
        caffe_gpu_strided_sum(this->num_output_, this->num_, filter_data, out_channel_saliency);
        caffe_gpu_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), out_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        filter_data = input_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_ * this->channels_ / this->group_, kernel_size, weight_saliency_data, weight_saliency_data);
        caffe_gpu_strided_sum_inner(this->num_, this->num_output_, this->channels_ / this->group_, weight_saliency_data, filter_data);
        caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_, filter_data, in_channel_saliency);
        caffe_gpu_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), in_channel_saliency);
      }
    } break;

    case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
      if (this->output_channel_saliency_compute_){
        filter_data = output_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_gpu_add(this->num_ * bias_count, bias_saliency_data, filter_data, filter_data);
        }
        caffe_gpu_abs(this->num_ * this->num_output_, filter_data, filter_data);
        caffe_gpu_strided_sum(this->num_output_, this->num_, filter_data, out_channel_saliency);
        caffe_gpu_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), out_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        filter_data = input_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_ * this->channels_ / this->group_, kernel_size, weight_saliency_data, weight_saliency_data);
        caffe_gpu_strided_sum_inner(this->num_, this->num_output_, this->channels_ / this->group_, weight_saliency_data, filter_data);
        caffe_gpu_abs(this->num_ * this->channels_ / this->group_, filter_data, filter_data);
        caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_, filter_data, in_channel_saliency);
        caffe_gpu_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), in_channel_saliency);
      }
    } break;

    case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
      if (this->output_channel_saliency_compute_){
        filter_data = output_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_add(this->num_ * bias_count, bias_saliency_data, filter_data, filter_data);
        }
        caffe_gpu_powx(this->num_ * this->num_output_, filter_data, (Dtype) 2, filter_data);
        caffe_gpu_strided_sum(this->num_output_, this->num_, filter_data, out_channel_saliency);
        caffe_gpu_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), out_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        filter_data = input_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_ * this->channels_ / this->group_, kernel_size, weight_saliency_data, weight_saliency_data);
        caffe_gpu_strided_sum_inner(this->num_, this->num_output_, this->channels_ / this->group_, weight_saliency_data, filter_data);
        caffe_gpu_powx(this->num_ * this->channels_ / this->group_, filter_data, (Dtype) 2, filter_data);
        caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_, filter_data, in_channel_saliency);
        caffe_gpu_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), in_channel_saliency);
      }
    } break;

    default: {
      if (this->output_channel_saliency_compute_){
        filter_data = output_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_gpu_add(this->num_ * bias_count, bias_saliency_data, filter_data, filter_data);
        }
        caffe_gpu_strided_sum(this->num_output_, this->num_, filter_data, out_channel_saliency);
        caffe_gpu_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), out_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        filter_data = input_saliencies_filter_.mutable_gpu_data();
        caffe_gpu_sum(this->num_ * this->num_output_ * this->channels_ / this->group_, kernel_size, weight_saliency_data, weight_saliency_data);
        caffe_gpu_strided_sum_inner(this->num_, this->num_output_, this->channels_ / this->group_, weight_saliency_data, filter_data);
        caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_, filter_data, in_channel_saliency);
        caffe_gpu_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), in_channel_saliency);
      }
    } break;
  }
}

template void ConvolutionLayer<float>::compute_norm_and_batch_avg_gpu(float * in_saliency_data, float * out_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * in_channel_saliency, float * out_channel_saliency);
template void ConvolutionLayer<double>::compute_norm_and_batch_avg_gpu(double * in_saliency_data, double * out_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * in_channel_saliency, double * out_channel_saliency);
template void ConvolutionLayer<float>::compute_norm_and_batch_avg_weights_gpu(float * weight_saliency_data, float * bias_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * in_channel_saliency, float * out_channel_saliency);
template void ConvolutionLayer<double>::compute_norm_and_batch_avg_weights_gpu(double * weight_saliency_data, double * bias_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * in_channel_saliency, double * out_channel_saliency);

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif


INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
