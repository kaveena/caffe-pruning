#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void reduce_nmckk_cpu(const int N, const int M, const int C, const int K, const Dtype * a, Dtype * y) {
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      Dtype accum = (Dtype) 0;
      for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
          accum += a[(n*M*C*K) + (m*C*K) + (c*K) + k];
        }
      }
      y[n*C +c] = accum;
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  Caffe::set_derivative_compute(true); //if any Convolution layer exists then need ddiff computation
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  if (this->saliency_term_) {
    this->saliency_ = this->layer_param_.convolution_saliency_param().saliency();
    this->saliency_norm_ = this->layer_param_.convolution_saliency_param().norm();
    this->saliency_input_ = this->layer_param_.convolution_saliency_param().input();
    this->saliency_ = this->layer_param_.convolution_saliency_param().saliency();
    this->output_channel_saliency_compute_ = this->layer_param_.convolution_saliency_param().output_channel_compute();
    this->input_channel_saliency_compute_ = this->layer_param_.convolution_saliency_param().input_channel_compute();
    if (this->bias_term_) {
      this->saliency_bias_ = true;
    }
    else {
      this->saliency_bias_ = false;
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->mask_term_) {
    weights_masked_shape_.clear();
    weights_masked_shape_.push_back(this->blobs_[this->mask_pos_]->count());
    weights_masked_.Reshape(weights_masked_shape_);
    if (this->bias_term_) {
      bias_masked_shape_.clear();
      bias_masked_shape_.push_back(this->blobs_[this->mask_pos_+1]->count());
      bias_masked_.Reshape(bias_masked_shape_);
    }
  }
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  this->compute_output_shape();
  if (this->saliency_term_) {
    if (this->output_channel_saliency_compute_) {
      if (this->layer_param_.convolution_saliency_param().saliency() == caffe::ConvolutionSaliencyParameter::ALL) {
        output_saliencies_channel_.Reshape({(int)(caffe::ConvolutionSaliencyParameter::ALL), this->num_output_});
      }
      else {
        output_saliencies_channel_.Reshape({1, this->num_output_});
      }
      output_saliencies_points_.Reshape(top[0]->shape()); //shape nmhw
      output_saliencies_filter_.Reshape({this->num_, this->num_output_}); //shape nm
    }
    if (this->input_channel_saliency_compute_) {
      if (this->layer_param_.convolution_saliency_param().saliency() == caffe::ConvolutionSaliencyParameter::ALL) {
        input_saliencies_channel_.Reshape({(int)(caffe::ConvolutionSaliencyParameter::ALL), this->channels_ / this->group_});
      }
      else {
        input_saliencies_channel_.Reshape({1, this->channels_ / this->group_});
      }
      input_saliencies_points_.Reshape(bottom[0]->shape()); //shape nchw
      input_saliencies_filter_.Reshape({this->num_, this->channels_}); //shape nc
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if ((this->saliency_ == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG) || 
      (this->saliency_ == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND) ||
      (this->saliency_ == caffe::ConvolutionSaliencyParameter::ALL)) {
    Caffe::set_derivative_compute(true); //if any Convolution Saliency layer exists then need ddiff computation
  }
  else {
    Caffe::set_derivative_compute(false);
  }
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias;
  if (this->mask_term_) {
    const Dtype* mask = this->blobs_[this->mask_pos_]->cpu_data();
    Dtype* weight_masked = this->weights_masked_.mutable_cpu_data();
    caffe_mul(this->blobs_[0]->count(), mask, weight, weight_masked);
    weight = this->weights_masked_.cpu_data();
  }
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
    if (this->mask_term_) {
      const Dtype* bias_mask = this->blobs_[this->mask_pos_+1]->cpu_data();
      Dtype* bias_masked = this->bias_masked_.mutable_cpu_data();
      caffe_mul(this->blobs_[1]->count(), bias_mask, bias, bias_masked);
      bias = this->bias_masked_.cpu_data();
    }
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* weights_sqr = this->weights_sqr_.mutable_cpu_data();
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

  if (this->saliency_term_ && (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT)) {
    weights_n_masked_.Reshape({this->num_, this->blobs_[0]->shape()[0], this->blobs_[0]->shape()[1], this->blobs_[0]->shape()[2], this->blobs_[0]->shape()[3]});
    full_weights_diff = weights_n_masked_.mutable_cpu_diff();
  }

  if (this->mask_term_) {
    weight = this->weights_masked_.cpu_data();
  }

  if (this->saliency_term_) {
    if (Caffe::derivative_compute()) {
      weight_ddiff = this->blobs_[0]->mutable_cpu_diff();
      if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
        full_weights_ddiff = weights_n_masked_.mutable_cpu_ddiff();
      }
    }
  }

  if (this->bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();

    if (this->saliency_term_) {
      if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
        bias_n_masked_.Reshape({this->num_, this->blobs_[1]->shape()[0]});
        full_bias_diff = bias_n_masked_.mutable_cpu_diff();
      }

      if (Caffe::derivative_compute()) {
        bias_ddiff = this->blobs_[1]->mutable_cpu_ddiff();
        if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
          full_bias_ddiff = bias_n_masked_.mutable_cpu_ddiff();
        }
      }
    }
  }

  caffe_powx(this->blobs_[0]->count(), weight, (Dtype)2, weights_sqr);

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* top_data = top[i]->cpu_data();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    const Dtype* top_ddiff;
    Dtype* bottom_ddiff;
    Dtype* input_sqr_;
    if (Caffe::derivative_compute()) {
      input_shaped_blob_.Reshape(bottom[i]->shape());
      top_ddiff = top[i]->cpu_ddiff();
      bottom_ddiff = bottom[i]->mutable_cpu_ddiff();
      weight_ddiff = this->blobs_[0]->mutable_cpu_ddiff();
      if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
        full_weights_ddiff = weights_n_masked_.mutable_cpu_ddiff();
      }
      input_sqr_ = input_shaped_blob_.mutable_cpu_data();
      caffe_powx(bottom[i]->count(), bottom[i]->cpu_data(), (Dtype) 2, input_sqr_);
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < this->num_; ++n) {
        if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
          this->backward_cpu_bias_no_accum(full_bias_diff + n * this->blobs_[1]->count(), top_diff + n * this->top_dim_);
          caffe_add(this->blobs_[1]->count(), full_bias_diff + n * this->blobs_[1]->count(), bias_diff, bias_diff);
          if (Caffe::derivative_compute()) {
            this->backward_cpu_bias_no_accum(full_bias_ddiff + n * this->blobs_[1]->count(), top_ddiff + n * this->top_dim_);
            caffe_add(this->blobs_[1]->count(), full_bias_ddiff + n * this->blobs_[1]->count(), bias_ddiff, bias_ddiff);
          }
        }
        else {
          this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
          if (Caffe::derivative_compute()) {
            this->backward_cpu_bias(bias_ddiff, top_ddiff + n * this->top_dim_);
          }
        }
      }
      if (this->mask_term_) {
        caffe_mul(this->blobs_[1]->count(), this->blobs_[this->mask_pos_+1]->cpu_data(), this->blobs_[1]->mutable_cpu_diff(), this->blobs_[1]->mutable_cpu_diff());
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          if (this->saliency_term_ && (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT)) {
            this->weight_cpu_gemm_no_accum(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, full_weights_diff + n * this->blobs_[0]->count());
            caffe_add(this->blobs_[0]->count(), full_weights_diff + n * this->blobs_[0]->count(), weight_diff, weight_diff);
            if (Caffe::derivative_compute()) {
              this->weight_cpu_gemm_no_accum(input_sqr_ + n * this->bottom_dim_,
                  top_ddiff + n * this->top_dim_, full_weights_ddiff + n * this->blobs_[0]->count());
              caffe_add(this->blobs_[0]->count(), full_weights_ddiff + n * this->blobs_[0]->count(), weight_ddiff, weight_ddiff);
            }
          }
          else {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
            if (Caffe::derivative_compute()) {
              this->weight_cpu_gemm(input_sqr_ + n * this->bottom_dim_,
                top_ddiff + n * this->top_dim_, weight_ddiff);
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i] || this->input_channel_saliency_compute_) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (Caffe::derivative_compute()) {
          if (propagate_down[i] || this->input_channel_saliency_compute_) {
            this->backward_cpu_gemm(top_ddiff + n * this->top_dim_, weights_sqr,
                bottom_ddiff + n * this->bottom_dim_);
          }
        }
      }
      if (this->mask_term_) {
        // Don't update weights that are masked off
        caffe_mul(this->blobs_[0]->count(), this->blobs_[this->mask_pos_]->cpu_data(), weight_diff, weight_diff);
      }
    }

    // Compute Channel saliency
    // MULTIPLE INPUTS NOT TREATED
    if (this->saliency_term_) {
      Dtype * output_channel_saliency_data = NULL;
      Dtype * input_channel_saliency_data = NULL;
      Dtype * output_channel_saliency_accum_data = NULL;
      Dtype * input_channel_saliency_accum_data = NULL;
      if (this->output_channel_saliency_compute_) {
        output_channel_saliency_data = output_saliencies_channel_.mutable_cpu_data();
        output_channel_saliency_accum_data = this->blobs_[this->saliency_pos_]->mutable_cpu_data();
      }
      if (this->input_channel_saliency_compute_) {
        input_channel_saliency_data = input_saliencies_channel_.mutable_cpu_data();
        input_channel_saliency_accum_data = this->blobs_[this->saliency_pos_+1]->mutable_cpu_data();
      }
      if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {

        switch (this->saliency_) {
          case (caffe::ConvolutionSaliencyParameter::TAYLOR): { // Taylor Series
            compute_taylor_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG): {
            compute_hessian_diag_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX2): {
            compute_hessian_diag_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::TAYLOR_2ND): {
            compute_taylor_2nd_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX2): {
            compute_taylor_2nd_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::WEIGHT_AVG): {
            compute_weight_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::DIFF_AVG): {
            compute_diff_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::ALL): {
            compute_taylor_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (0 * this->num_output_), input_channel_saliency_data + (0 * this->channels_ / this->group_));
            compute_hessian_diag_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (1 * this->num_output_), input_channel_saliency_data + (1 * this->channels_ / this->group_));
            compute_hessian_diag_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (2 * this->num_output_), input_channel_saliency_data + (2 * this->channels_ / this->group_));
            compute_taylor_2nd_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (3 * this->num_output_), input_channel_saliency_data + (3 * this->channels_ / this->group_));
            compute_taylor_2nd_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (4 * this->num_output_), input_channel_saliency_data + (4 * this->channels_ / this->group_));
            compute_weight_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (5 * this->num_output_), input_channel_saliency_data + (5 * this->channels_ / this->group_));
            compute_diff_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (6 * this->num_output_), input_channel_saliency_data + (6 * this->channels_ / this->group_));
          } break;

          default: {
          } break;
        }
      }
      else if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::ACTIVATION) {

        switch (this->saliency_) {
          case (caffe::ConvolutionSaliencyParameter::TAYLOR): { // Taylor Series
            compute_taylor_cpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG): {
            compute_hessian_diag_cpu(top_data, top_ddiff, bottom_data, bottom_ddiff, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX2): {
            compute_hessian_diag_approx2_cpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::TAYLOR_2ND): {
            compute_taylor_2nd_cpu(top_data, top_diff, top_ddiff, bottom_data, bottom_diff, bottom_ddiff, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX2): {
            compute_taylor_2nd_approx2_cpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::WEIGHT_AVG): {
            compute_weight_avg_cpu(top_data, bottom_data, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::DIFF_AVG): {
            compute_diff_avg_cpu(top_diff, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
          } break;

          case (caffe::ConvolutionSaliencyParameter::ALL): {
            compute_taylor_cpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data + (0 * this->num_output_), input_channel_saliency_data + (0 * this->channels_ / this->group_));
            compute_hessian_diag_cpu(top_data, top_ddiff, bottom_data, bottom_ddiff, output_channel_saliency_data + (1 * this->num_output_), input_channel_saliency_data + (1 * this->channels_ / this->group_));
            compute_hessian_diag_approx2_cpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data + (2 * this->num_output_), input_channel_saliency_data + (2 * this->channels_ / this->group_));
            compute_taylor_2nd_cpu(top_data, top_diff, top_ddiff, bottom_data, bottom_diff, bottom_ddiff, output_channel_saliency_data + (3 * this->num_output_), input_channel_saliency_data + (3 * this->channels_ / this->group_));
            compute_taylor_2nd_approx2_cpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data + (4 * this->num_output_), input_channel_saliency_data + (4 * this->channels_ / this->group_));
            compute_weight_avg_cpu(top_data, bottom_data, output_channel_saliency_data + (5 * this->num_output_), input_channel_saliency_data + (5 * this->channels_ / this->group_));
            compute_diff_avg_cpu(top_diff, bottom_diff, output_channel_saliency_data + (6 * this->num_output_), input_channel_saliency_data + (6 * this->channels_ / this->group_));
          } break;

          default: {
          } break;
        }
      }
      if (this->output_channel_saliency_compute_) {
        if (this->layer_param_.convolution_saliency_param().accum()) {
          caffe_add(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data, output_channel_saliency_accum_data); 
        }
        else {
          caffe_copy(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data);
        }
      }
      if (this->input_channel_saliency_compute_) {
        if (this->layer_param_.convolution_saliency_param().accum()) {
          caffe_add(input_saliencies_channel_.count(), input_channel_saliency_data, input_channel_saliency_accum_data, input_channel_saliency_accum_data); 
        }
        else {
          caffe_copy(input_saliencies_channel_.count(), input_channel_saliency_data, input_channel_saliency_accum_data);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_cpu(Dtype * output_saliency_data, Dtype * input_saliency_data, Dtype * output_channel_saliency, Dtype * input_channel_saliency) {

  if (this->output_channel_saliency_compute_) {
    int count = this->output_saliencies_points_.count(2,4); 
    Dtype* filter_out_saliency_data = this->output_saliencies_filter_.mutable_cpu_data();    
    switch (this->saliency_norm_) {
      case (caffe::ConvolutionSaliencyParameter::L1): {
        caffe_abs(this->num_ * this->num_output_ * count, output_saliency_data, output_saliency_data);
        caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
    } break;
    
      case (caffe::ConvolutionSaliencyParameter::L2): {
        caffe_powx(this->num_ * this->num_output_ * count, output_saliency_data, (Dtype) 2, output_saliency_data);
        caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      } break;
    
      case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
        caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
        caffe_abs(this->num_ * this->num_output_, filter_out_saliency_data, filter_out_saliency_data);
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      } break;
    
      case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
        caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
        caffe_powx(this->num_ * this->num_output_, filter_out_saliency_data, (Dtype) 2, filter_out_saliency_data);
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      } break;

      default: {
        caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      } break;
    }
    caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
  }
  if (this->input_channel_saliency_compute_) {
    int count = this->input_saliencies_points_.count(2,4); 
    Dtype* filter_in_saliency_data = this->input_saliencies_filter_.mutable_cpu_data();    
    switch (this->saliency_norm_) {
      case (caffe::ConvolutionSaliencyParameter::L1): {
        caffe_abs(this->num_ * this->channels_ * count, input_saliency_data, input_saliency_data);
        caffe_sum(this->num_ * this->channels_, count, input_saliency_data, filter_in_saliency_data); //sum hxw
        caffe_strided_sum(this->channels_ / this->group_, this->num_ * this->group_, filter_in_saliency_data, input_channel_saliency);
    } break;
    
      case (caffe::ConvolutionSaliencyParameter::L2): {
        caffe_powx(this->num_ * this->channels_ * count, input_saliency_data, (Dtype) 2, input_saliency_data);
        caffe_sum(this->num_ * this->channels_, count, input_saliency_data, filter_in_saliency_data); //sum hxw
        caffe_strided_sum(this->channels_ / this->group_, this->num_ * this->group_, filter_in_saliency_data, input_channel_saliency);
      } break;
    
      case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
        caffe_sum(this->num_ * this->channels_, count, input_saliency_data, filter_in_saliency_data); //sum hxw
        caffe_abs(this->num_ * this->channels_ * count, filter_in_saliency_data, filter_in_saliency_data);
        caffe_strided_sum(this->channels_ / this->group_, this->num_ * this->group_, filter_in_saliency_data, input_channel_saliency);
      } break;
    
      case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
        caffe_sum(this->num_ * this->channels_, count, input_saliency_data, filter_in_saliency_data); //sum hxw
        caffe_powx(this->num_ * this->channels_ * count, filter_in_saliency_data, (Dtype) 2, filter_in_saliency_data);
        caffe_strided_sum(this->channels_ / this->group_, this->num_ * this->group_, filter_in_saliency_data, input_channel_saliency);
      } break;

      default: {
        caffe_sum(this->num_ * this->channels_, count, input_saliency_data, filter_in_saliency_data); //sum hxw
        caffe_strided_sum(this->channels_ / this->group_, this->num_ * this->group_, filter_in_saliency_data, input_channel_saliency);
      } break;
    }
    caffe_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_weights_cpu(Dtype * weight_saliency_data, Dtype * bias_saliency_data, Dtype * output_channel_saliency, Dtype * input_channel_saliency) {

  Dtype* filter_out_saliency_data;
  Dtype* filter_in_saliency_data;
  
  int kernel_size = this->blobs_[0]->count(2,4);
  int weights_count = this->blobs_[0]->count();
  int bias_count;
  
  if (this->bias_term_) {
    bias_count = this->blobs_[1]->count();
  }

  switch (this->saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_abs(this->num_ * weights_count, weight_saliency_data, weight_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_abs(this->num_ * bias_count, bias_saliency_data, bias_saliency_data);
     }
      if (this->output_channel_saliency_compute_) {
        filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
        caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
        }
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
        caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        caffe_strided_sum(this->channels_ * kernel_size / this->group_, this->num_ * this->num_output_, weight_saliency_data, weight_saliency_data);
        caffe_sum(this->channels_ / this->group_, kernel_size, weight_saliency_data, input_channel_saliency);
        caffe_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
      }
    } break;
  
    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_powx(this->num_ * weights_count, weight_saliency_data, (Dtype) 2, weight_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_powx(this->num_ * bias_count, bias_saliency_data, (Dtype) 2, bias_saliency_data);
      }
      if (this->output_channel_saliency_compute_) {
        filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
        caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
        }
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
        caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        caffe_strided_sum(this->channels_ * kernel_size / this->group_, this->num_ * this->num_output_, weight_saliency_data, weight_saliency_data);
        caffe_sum(this->channels_ / this->group_, kernel_size, weight_saliency_data, input_channel_saliency);
        caffe_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
      }
    } break;
  
    case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
      if (this->output_channel_saliency_compute_) {
        filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
        caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
        }
        caffe_abs(this->num_ * this->num_output_, filter_out_saliency_data, filter_out_saliency_data);
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
        caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        filter_in_saliency_data = input_saliencies_filter_.mutable_cpu_data();    
        reduce_nmckk_cpu(this->num_, this->num_output_, this->channels_ / this->group_, kernel_size, weight_saliency_data, filter_in_saliency_data);
        caffe_abs(this->num_ * this->channels_ / this->group_, filter_in_saliency_data, filter_in_saliency_data);
        caffe_strided_sum(this->channels_ / this->group_, this->num_, filter_in_saliency_data, input_channel_saliency);
        caffe_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
      }
    } break;
  
    case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
      if (this->output_channel_saliency_compute_) {
        filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
        caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
        }
        caffe_powx(this->num_ * this->num_output_, filter_out_saliency_data, (Dtype) 2, filter_out_saliency_data);
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
        caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        filter_in_saliency_data = input_saliencies_filter_.mutable_cpu_data();    
        reduce_nmckk_cpu(this->num_, this->num_output_, this->channels_ / this->group_, kernel_size, weight_saliency_data, filter_in_saliency_data);
        caffe_powx(this->num_ * this->channels_ / this->group_, filter_in_saliency_data, (Dtype) 2, filter_in_saliency_data);
        caffe_strided_sum(this->channels_ / this->group_, this->num_, filter_in_saliency_data, input_channel_saliency);
        caffe_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
      }
    } break;

    default: {
      if (this->output_channel_saliency_compute_) {
        filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
        caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
        if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
          caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
        }
        caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
        caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
      }
      if (this->input_channel_saliency_compute_) {
        caffe_strided_sum(this->channels_ * kernel_size / this->group_, this->num_ * this->num_output_, weight_saliency_data, weight_saliency_data);
        caffe_sum(this->channels_ / this->group_, kernel_size, weight_saliency_data, input_channel_saliency);
        caffe_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
      }
    } break;
  }
}

template void ConvolutionLayer<float>::compute_norm_and_batch_avg_cpu(float * output_saliency_data, float * input_saliency_data, float * output_channel_saliency, float * input_channel_saliency);
template void ConvolutionLayer<double>::compute_norm_and_batch_avg_cpu(double * output_saliency_data, double * input_saliency_data, double * output_channel_saliency, double * input_channel_saliency);
template void ConvolutionLayer<float>::compute_norm_and_batch_avg_weights_cpu(float * weight_saliency_data, float * bias_saliency_data, float * output_saliency_data, float * input_saliency_data);
template void ConvolutionLayer<double>::compute_norm_and_batch_avg_weights_cpu(double * weight_saliency_data, double * bias_saliency_data, double * output_saliency_data, double * input_saliency_data);

#ifdef CPU_ONLY
template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_fisher_gpu(const Dtype *  act_data, const Dtype *  act_diff, const Dtype * input_data, const Dtype * input_diff,  Dtype * fisher_info_out, Dtype * fisher_info_in) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_gpu(const Dtype *  act_data, const Dtype *  act_diff, const Dtype * input_data, const Dtype * input_diff,  Dtype * taylor_out, Dtype * taylor_in) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_hessian_diag_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * hessian_diag) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_hessian_diag_approx2_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * hessian_diag) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * taylor_2nd) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_approx2_gpu(const Dtype *  act_data, const Dtype * act_diff, Dtype * taylor_2nd) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_weight_avg_gpu(const Dtype * act_data, Dtype * saliency_info) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_diff_avg_gpu(const Dtype *  act_diff, Dtype * saliency_info) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_fisher_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_hessian_diag_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * hessian_diag) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_hessian_diag_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * hessian_diag) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_2nd) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_2nd) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_weight_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_diff_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_gpu(int count, Dtype * output_saliency_data, Dtype * saliency_data, Dtype * bias_saliency_data) { NO_GPU; }
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
