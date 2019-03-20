#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  if (this->saliency_term_) {
    this->saliency_ = this->layer_param_.convolution_saliency_param().saliency();
    this->saliency_norm_ = this->layer_param_.convolution_saliency_param().norm();
    this->saliency_input_ = this->layer_param_.convolution_saliency_param().input();
    if (this->bias_term_) {
      this->saliency_bias_ = true;
    }
    else {
      this->saliency_bias_ = false;
    }
    if ((this->saliency_ == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_LM) ||
        (this->saliency_ == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_LM) ||
        (this->saliency_ == caffe::ConvolutionSaliencyParameter::ALL)) {
      Caffe::set_derivative_compute(true); //if any Convolution Saliency layer exists then need ddiff computation
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
    if (this->layer_param_.convolution_saliency_param().saliency() == caffe::ConvolutionSaliencyParameter::ALL) {
      output_saliencies_channel_.Reshape({(int)(caffe::ConvolutionSaliencyParameter::ALL), this->num_output_});
    }
    else {
      output_saliencies_channel_.Reshape({1, this->num_output_});
    }
    output_saliencies_points_.Reshape(top[0]->shape()); //shape nchw
    output_saliencies_filter_.Reshape({this->num_, this->num_output_}); //shape nc
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
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
      if(this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
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
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (Caffe::derivative_compute()) {
          if (propagate_down[i]) {
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
      if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
        Dtype* channel_saliency_data = output_saliencies_channel_.mutable_cpu_data();

        switch (this->saliency_) {
          case (0): { // Fisher Information
            compute_fisher_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (1): { // Taylor Series
            compute_taylor_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (2): {
            compute_hessian_diag_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (3): {
            compute_hessian_diag_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (4): {
            compute_taylor_2nd_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (5): {
            compute_taylor_2nd_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (6): {
            compute_weight_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (7): {
            compute_diff_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          } break;

          case (8): {
            compute_fisher_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
            compute_taylor_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + this->num_output_);
            compute_hessian_diag_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (2*this->num_output_));
            compute_hessian_diag_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (3*this->num_output_));
            compute_taylor_2nd_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (4*this->num_output_));
            compute_taylor_2nd_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (5*this->num_output_));
            compute_weight_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (5*this->num_output_));
            compute_diff_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (6*this->num_output_));
          } break;

          default: {
          } break;
        }
      }
      else {
        Dtype* channel_saliency_data = output_saliencies_channel_.mutable_cpu_data();

        switch (this->saliency_) {
          case (0): { // Fisher Information
            compute_fisher_cpu(top_data, top_diff, channel_saliency_data);
          } break;

          case (1): { // Taylor Series
            compute_taylor_cpu(top_data, top_diff, channel_saliency_data);
          } break;

          case (2): {
            compute_hessian_diag_cpu(top_data, top_diff, top_ddiff, channel_saliency_data);
          } break;

          case (3): {
            compute_hessian_diag_approx2_cpu(top_data, top_diff, channel_saliency_data);
          } break;

          case (4): {
            compute_taylor_2nd_cpu(top_data, top_diff, top_ddiff, channel_saliency_data);
          } break;

          case (5): {
            compute_taylor_2nd_approx2_cpu(top_data, top_diff, channel_saliency_data);
          } break;

          case (6): {
            compute_weight_avg_cpu(top_data, channel_saliency_data);
          } break;

          case (7): {
            compute_diff_avg_cpu(top_diff, channel_saliency_data);
          } break;

          case (8): {
            compute_fisher_cpu(top_data, top_diff, channel_saliency_data);
            compute_taylor_cpu(top_data, top_diff, channel_saliency_data + this->num_output_);
            compute_hessian_diag_cpu(top_data, top_diff, top_ddiff, channel_saliency_data + (2*this->num_output_));
            compute_hessian_diag_approx2_cpu(top_data, top_diff, channel_saliency_data + (3*this->num_output_));
            compute_taylor_2nd_cpu(top_data, top_diff, top_ddiff, channel_saliency_data + (4*this->num_output_));
            compute_taylor_2nd_approx2_cpu(top_data, top_diff, channel_saliency_data + (5*this->num_output_));
            compute_weight_avg_cpu(top_data, channel_saliency_data + (6*this->num_output_));
            compute_diff_avg_cpu(top_diff, channel_saliency_data + (7*this->num_output_));
          } break;

          default: {
          } break;
        }
      }
      if (this->layer_param_.convolution_saliency_param().accum()) {
        caffe_add(output_saliencies_channel_.count(), output_saliencies_channel_.mutable_cpu_data(), this->blobs_[this->saliency_pos_]->mutable_cpu_data(), this->blobs_[this->saliency_pos_]->mutable_cpu_data());
      }
      else {
        caffe_copy(output_saliencies_channel_.count(), output_saliencies_channel_.mutable_cpu_data(), this->blobs_[this->saliency_pos_]->mutable_cpu_data());
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_cpu(int count, Dtype * output_saliency_data, Dtype * saliency_data, Dtype * bias_saliency_data) {

  Dtype* filter_saliency_data = this->output_saliencies_filter_.mutable_cpu_data();

  switch (this->saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_abs(this->num_ * this->num_output_ * count, output_saliency_data, output_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_abs(this->num_ * this->num_output_, bias_saliency_data, bias_saliency_data);
      }
    } break;

    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_powx(this->num_ * this->num_output_ * count, output_saliency_data, (Dtype) 2, output_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_powx(this->num_ * this->num_output_, bias_saliency_data, (Dtype) 2, bias_saliency_data);
      }
    } break;

    default: {
    } break;
  }

  caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_saliency_data); //sum hxw
  if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
    caffe_add(this->num_ * this->num_output_, bias_saliency_data, filter_saliency_data, filter_saliency_data);
  }
  caffe_strided_sum(this->num_output_, this->num_, filter_saliency_data, saliency_data);
  caffe_scal(this->num_output_, 1 / (Dtype)(this->num_), saliency_data);

}

template void ConvolutionLayer<float>::compute_norm_and_batch_avg_cpu(int count, float * output_saliency_data, float * saliency_data, float * bias_saliency_data);
template void ConvolutionLayer<double>::compute_norm_and_batch_avg_cpu(int count, double * output_saliency_data, double * saliency_data, double * bias_saliency_data);

#ifdef CPU_ONLY
template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_fisher_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * fisher_info) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * taylor) { NO_GPU; }

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
