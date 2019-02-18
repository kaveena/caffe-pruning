#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {


template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if ((this->saliency_ == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG) || 
      (this->saliency_ == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND) ||
      (this->saliency_ == caffe::ConvolutionSaliencyParameter::ALL)) {
    Caffe::set_derivative_compute(true); //if any Convolution Saliency layer exists then need ddiff computation
  }
  else {
    Caffe::set_derivative_compute(false);
  }
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias;
  if (this->mask_term_) {
    const Dtype* mask = this->blobs_[this->mask_pos_]->gpu_data();
    Dtype* weight_masked = this->weights_masked_.mutable_gpu_data();
    caffe_gpu_mul(this->blobs_[0]->count(), mask, weight, weight_masked);
    weight = this->weights_masked_.gpu_data();
  }
  if (this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    if (this->mask_term_) {
      const Dtype* bias_mask = this->blobs_[this->mask_pos_+1]->gpu_data();
      Dtype* bias_masked = this->bias_masked_.mutable_gpu_data();
      caffe_gpu_mul(this->blobs_[1]->count(), bias_mask, bias, bias_masked);
      bias = this->bias_masked_.gpu_data();
    }
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* weights_sqr = this->weights_sqr_.mutable_gpu_data();
  Blob<Dtype>  weights_n_masked_;
  Blob<Dtype> bias_n_masked_;
  Blob<Dtype> input_shaped_blob_;
  Dtype* full_weights_diff;
  if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
    weights_n_masked_.Reshape({this->num_, this->blobs_[0]->shape()[0], this->blobs_[0]->shape()[1], this->blobs_[0]->shape()[2], this->blobs_[0]->shape()[3]});
    full_weights_diff = weights_n_masked_.mutable_gpu_diff();
  }
  Dtype* weight_ddiff;
  Dtype* full_weights_ddiff;
  
  Dtype* bias_diff;
  Dtype* full_bias_diff;
  Dtype* bias_ddiff;
  Dtype* full_bias_ddiff;
  
  if (this->mask_term_) {
    weight = this->weights_masked_.gpu_data();
  }
  if (Caffe::derivative_compute()) {
    weight_ddiff = this->blobs_[0]->mutable_gpu_diff();
    if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
      full_weights_ddiff = weights_n_masked_.mutable_gpu_ddiff();
    }
  }

  if (this->bias_term_) {
    if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
      bias_n_masked_.Reshape({this->num_, this->blobs_[1]->shape()[0]});
      full_bias_diff = bias_n_masked_.mutable_gpu_diff();
    }
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    if (Caffe::derivative_compute()) {
      bias_ddiff = this->blobs_[1]->mutable_gpu_ddiff();
      if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
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
    if (Caffe::derivative_compute()) {
      input_shaped_blob_.Reshape(bottom[i]->shape());
      top_ddiff = top[i]->gpu_ddiff();
      bottom_ddiff = bottom[i]->mutable_gpu_ddiff();
      weight_ddiff = this->blobs_[0]->mutable_gpu_ddiff();
      if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
        full_weights_ddiff = weights_n_masked_.mutable_gpu_ddiff();
      }
      input_sqr_ = input_shaped_blob_.mutable_gpu_data();
      caffe_gpu_powx(bottom[i]->count(), bottom[i]->gpu_data(), (Dtype) 2, input_sqr_);
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < this->num_; ++n) {
        if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
          this->backward_gpu_bias_no_accum(full_bias_diff + n * this->blobs_[1]->count(), top_diff + n * this->top_dim_);
          caffe_gpu_add(this->blobs_[1]->count(), full_bias_diff + n * this->blobs_[1]->count(), bias_diff, bias_diff);
          if (Caffe::derivative_compute()) {
            this->backward_gpu_bias_no_accum(full_bias_ddiff + n * this->blobs_[1]->count(), top_ddiff + n * this->top_dim_);
            caffe_gpu_add(this->blobs_[1]->count(), full_bias_ddiff + n * this->blobs_[1]->count(), bias_ddiff, bias_ddiff);
          }
        }
        else {
          this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
          if (Caffe::derivative_compute()) {
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
          if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
            this->weight_gpu_gemm_no_accum(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, full_weights_diff + n * this->blobs_[0]->count());
            caffe_gpu_add(this->blobs_[0]->count(), full_weights_diff + n * this->blobs_[0]->count(), weight_diff, weight_diff);
            if (Caffe::derivative_compute()) {
              this->weight_gpu_gemm_no_accum(input_sqr_ + n * this->bottom_dim_,
                  top_ddiff + n * this->top_dim_, full_weights_ddiff + n * this->blobs_[0]->count());
              caffe_gpu_add(this->blobs_[0]->count(), full_weights_ddiff + n * this->blobs_[0]->count(), weight_ddiff, weight_ddiff);
            }
          }
          else {
            this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
            if (Caffe::derivative_compute()) {
              this->weight_gpu_gemm(input_sqr_ + n * this->bottom_dim_,
                top_ddiff + n * this->top_dim_, weight_ddiff);
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i] || this->input_channel_saliency_compute_) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (Caffe::derivative_compute()) {
          if (propagate_down[i] || this->input_channel_saliency_compute_) {
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
    Dtype * output_channel_saliency_data = NULL;
    Dtype * input_channel_saliency_data = NULL;
    Dtype * output_channel_saliency_accum_data = NULL;
    Dtype * input_channel_saliency_accum_data = NULL;
    if (this->output_channel_saliency_compute_) {
      output_channel_saliency_data = output_saliencies_channel_.mutable_gpu_data();
      output_channel_saliency_accum_data = this->blobs_[this->saliency_pos_]->mutable_gpu_data();
    }
    if (this->input_channel_saliency_compute_) {
      input_channel_saliency_data = input_saliencies_channel_.mutable_gpu_data();
      input_channel_saliency_accum_data = this->blobs_[this->saliency_pos_+1]->mutable_gpu_data();
    }
    if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {

      switch (this->saliency_) {
        case (0): { // Fisher Information
          compute_fisher_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (1): { // Taylor Series
          compute_taylor_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (2): {
          compute_hessian_diag_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (3): {
          compute_hessian_diag_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (4): {
          compute_taylor_2nd_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (5): {
          compute_taylor_2nd_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (6): {
          compute_weight_avg_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (7): {
          compute_diff_avg_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (8): {
          compute_fisher_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data, input_channel_saliency_data);
          compute_taylor_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (this->num_output_), input_channel_saliency_data + (this->channels_ / this->group_));
          compute_hessian_diag_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (2 * this->num_output_), input_channel_saliency_data + (2 * this->channels_ / this->group_));
          compute_hessian_diag_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (3 * this->num_output_), input_channel_saliency_data + (3 * this->channels_ / this->group_));
          compute_taylor_2nd_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (4 * this->num_output_), input_channel_saliency_data + (4 * this->channels_ / this->group_));
          compute_taylor_2nd_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (5 * this->num_output_), input_channel_saliency_data + (5 * this->channels_ / this->group_));
          compute_weight_avg_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (6 * this->num_output_), input_channel_saliency_data + (6 * this->channels_ / this->group_));
          compute_diff_avg_weights_gpu(&weights_n_masked_, &bias_n_masked_, output_channel_saliency_data + (7 * this->num_output_), input_channel_saliency_data + (7 * this->channels_ / this->group_));
        } break;

        default: {
        } break;
      }
    }
    else if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::ACTIVATION) {

      switch (this->saliency_) {
        case (0): { // Fisher Information
          compute_fisher_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (1): { // Taylor Series
          compute_taylor_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (2): {
          compute_hessian_diag_gpu(top_data, top_ddiff, bottom_data, bottom_ddiff, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (3): {
          compute_hessian_diag_approx2_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (4): {
          compute_taylor_2nd_gpu(top_data, top_diff, top_ddiff, bottom_data, bottom_diff, bottom_ddiff, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (5): {
          compute_taylor_2nd_approx2_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (6): {
          compute_weight_avg_gpu(top_data, bottom_data, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (7): {
          compute_diff_avg_gpu(top_diff, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
        } break;

        case (8): {
          compute_fisher_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data, input_channel_saliency_data);
          compute_taylor_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data + (this->num_output_), input_channel_saliency_data + (this->channels_ / this->group_));
          compute_hessian_diag_gpu(top_data, top_ddiff, bottom_data, bottom_ddiff, output_channel_saliency_data + (2*this->num_output_), input_channel_saliency_data + (2 * this->channels_ / this->group_));
          compute_hessian_diag_approx2_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data + (3*this->num_output_), input_channel_saliency_data + (3 * this->channels_ / this->group_));
          compute_taylor_2nd_gpu(top_data, top_diff, top_ddiff, bottom_data, bottom_diff, bottom_ddiff, output_channel_saliency_data + (4*this->num_output_), input_channel_saliency_data + (4 * this->channels_ / this->group_));
          compute_taylor_2nd_approx2_gpu(top_data, top_diff, bottom_data, bottom_diff, output_channel_saliency_data + (5*this->num_output_), input_channel_saliency_data + (5 * this->channels_ / this->group_));
          compute_weight_avg_gpu(top_data, bottom_data, output_channel_saliency_data + (6*this->num_output_), input_channel_saliency_data + (6 * this->channels_ / this->group_));
          compute_diff_avg_gpu(top_diff, bottom_diff, output_channel_saliency_data + (7*this->num_output_), input_channel_saliency_data + (7 * this->channels_ / this->group_));
        } break;

        default: {
        } break;
      }
    }
    if (this->output_channel_saliency_compute_) {
      if (this->layer_param_.convolution_saliency_param().accum()) {
        caffe_gpu_add(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data, output_channel_saliency_accum_data); 
      }
      else {
        caffe_copy(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data);
      }
    }
    if (this->input_channel_saliency_compute_) {
      if (this->layer_param_.convolution_saliency_param().accum()) {
        caffe_gpu_add(input_saliencies_channel_.count(), input_channel_saliency_data, input_channel_saliency_accum_data, input_channel_saliency_accum_data); 
      }
      else {
        caffe_copy(input_saliencies_channel_.count(), input_channel_saliency_data, input_channel_saliency_accum_data);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_norm_and_batch_avg_gpu(Dtype * output_saliency_data, Dtype * input_saliency_data, Dtype * output_channel_saliency, Dtype * input_channel_saliency) {

  if (this->output_channel_saliency_compute_) {
    int count = this->output_saliencies_points_.count(2,4); 
    Dtype* filter_out_saliency_data = this->output_saliencies_filter_.mutable_gpu_data();    
    switch (this->saliency_norm_) {
      case (caffe::ConvolutionSaliencyParameter::L1): {
        caffe_gpu_abs(this->num_ * this->num_output_ * count, output_saliency_data, output_saliency_data);
    } break;
    
      case (caffe::ConvolutionSaliencyParameter::L2): {
        caffe_gpu_powx(this->num_ * this->num_output_ * count, output_saliency_data, (Dtype) 2, output_saliency_data);
      } break;

      default: {
      } break;
    }
    caffe_gpu_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
    caffe_gpu_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
    caffe_gpu_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
  }
  if (this->input_channel_saliency_compute_) {
    int count = this->input_saliencies_points_.count(2,4); 
    Dtype* filter_in_saliency_data = this->input_saliencies_filter_.mutable_gpu_data();    
    switch (this->saliency_norm_) {
      case (caffe::ConvolutionSaliencyParameter::L1): {
        caffe_gpu_abs(this->num_ * this->channels_ * count, input_saliency_data, input_saliency_data);
    } break;
    
      case (caffe::ConvolutionSaliencyParameter::L2): {
        caffe_gpu_powx(this->num_ * this->channels_ * count, input_saliency_data, (Dtype) 2, input_saliency_data);
      } break;

      default: {
      } break;
    }
    caffe_gpu_sum(this->num_ * this->channels_, count, input_saliency_data, filter_in_saliency_data); //sum hxw
    caffe_gpu_strided_sum(this->channels_ / this->group_, this->num_ * this->group_, filter_in_saliency_data, input_channel_saliency);
    caffe_gpu_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_norm_and_batch_avg_weights_gpu(Dtype * weight_saliency_data, Dtype * bias_saliency_data, Dtype * output_channel_saliency, Dtype * input_channel_saliency) {

  Dtype* filter_out_saliency_data;
  
  int kernel_size = this->blobs_[0]->count(2,4);
  int weights_count = this->blobs_[0]->count();
  int bias_count;
  
  if (this->bias_term_) {
    bias_count = this->blobs_[1]->count();
  }

  switch (this->saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_gpu_abs(this->num_ * weights_count, weight_saliency_data, weight_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_abs(this->num_ * bias_count, bias_saliency_data, bias_saliency_data);
     }
    } break;
  
    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_gpu_powx(this->num_ * weights_count, weight_saliency_data, (Dtype) 2, weight_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_powx(this->num_ * bias_count, bias_saliency_data, (Dtype) 2, bias_saliency_data);
      }
    } break;

    default: {
    } break;
  }
  
  if (this->output_channel_saliency_compute_) {
    filter_out_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
    caffe_gpu_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
    if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
      caffe_gpu_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
    }
    caffe_gpu_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
    caffe_gpu_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
  }
  if (this->input_channel_saliency_compute_) {
    caffe_gpu_strided_sum(this->channels_ * kernel_size / this->group_, this->num_ * this->num_output_, weight_saliency_data, weight_saliency_data);
    caffe_gpu_sum(this->channels_ / this->group_, kernel_size, weight_saliency_data, input_channel_saliency);
    caffe_gpu_scal(this->channels_ / this->group_, (Dtype) 1.0 / (Dtype)(this->num_), input_channel_saliency);
  }
}

template void ConvolutionSaliencyLayer<float>::compute_norm_and_batch_avg_gpu(float * output_saliency_data, float * input_saliency_data, float * output_channel_saliency, float * input_channel_saliency);
template void ConvolutionSaliencyLayer<double>::compute_norm_and_batch_avg_gpu(double * output_saliency_data, double * input_saliency_data, double * output_channel_saliency, double * input_channel_saliency);
template void ConvolutionSaliencyLayer<float>::compute_norm_and_batch_avg_weights_gpu(float * weight_saliency_data, float * bias_saliency_data, float * output_saliency_data, float * input_saliency_data);
template void ConvolutionSaliencyLayer<double>::compute_norm_and_batch_avg_weights_gpu(double * weight_saliency_data, double * bias_saliency_data, double * output_saliency_data, double * input_saliency_data);

#ifdef CPU_ONLY
STUB_GPU(ConvolutionSaliencyLayer);
#endif


INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionSaliencyLayer);

}  // namespace caffe
