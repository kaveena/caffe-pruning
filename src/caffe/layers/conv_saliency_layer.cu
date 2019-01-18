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
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (Caffe::derivative_compute()) {
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
    if (this->saliency_input_ == caffe::ConvolutionSaliencyParameter::WEIGHT) {
      Dtype* channel_saliency_data = output_saliencies_channel_.mutable_gpu_data();    
  
      switch (this->saliency_) {
        case (0): { // Fisher Information
          compute_fisher_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
        } break;

        case (1): { // Taylor Series
          compute_taylor_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
        } break;

        case (2): {
          compute_hessian_diag_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
        } break;

        case (3): {
          compute_hessian_diag_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
        } break;

        case (4): {
          compute_taylor_2nd_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
        } break;

        case (5): {
          compute_taylor_2nd_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
        } break;

        case (6): {
          compute_weight_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
        } break;

        case (7): {
          compute_fisher_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data);
          compute_taylor_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + this->num_output_);
          compute_hessian_diag_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (2*this->num_output_));
          compute_hessian_diag_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (3*this->num_output_));
          compute_taylor_2nd_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (4*this->num_output_));
          compute_taylor_2nd_approx2_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (5*this->num_output_));
          compute_weight_weights_gpu(&weights_n_masked_, &bias_n_masked_, channel_saliency_data + (5*this->num_output_));
        } break;

        default: {
        } break;
      }
    }
    else {
      Dtype* channel_saliency_data = output_saliencies_channel_.mutable_gpu_data();    
  
      switch (this->saliency_) {
        case (0): { // Fisher Information
          compute_fisher_gpu(top_data, top_diff, channel_saliency_data);
        } break;

        case (1): { // Taylor Series
          compute_taylor_gpu(top_data, top_diff, channel_saliency_data);
        } break;

        case (2): {
          compute_hessian_diag_gpu(top_data, top_diff, top_ddiff, channel_saliency_data);
        } break;

        case (3): {
          compute_hessian_diag_approx2_gpu(top_data, top_diff, channel_saliency_data);
        } break;

        case (4): {
          compute_taylor_2nd_gpu(top_data, top_diff, top_ddiff, channel_saliency_data);
        } break;

        case (5): {
          compute_taylor_2nd_approx2_gpu(top_data, top_diff, channel_saliency_data);
        } break;

        case (6): {
          compute_weight_gpu(top_data, channel_saliency_data);
        } break;

        case (7): {
          compute_fisher_gpu(top_data, top_diff, channel_saliency_data);
          compute_taylor_gpu(top_data, top_diff, channel_saliency_data + this->num_output_);
          compute_hessian_diag_gpu(top_data, top_diff, top_ddiff, channel_saliency_data + (2*this->num_output_));
          compute_hessian_diag_approx2_gpu(top_data, top_diff, channel_saliency_data + (3*this->num_output_));
          compute_taylor_2nd_gpu(top_data, top_diff, top_ddiff, channel_saliency_data + (4*this->num_output_));
          compute_taylor_2nd_approx2_gpu(top_data, top_diff, channel_saliency_data + (5*this->num_output_));
          compute_weight_gpu(top_data, channel_saliency_data + (6*this->num_output_));
        } break;

        default: {
        } break;
      }
    }
    if (this->layer_param_.convolution_saliency_param().accum()) {
      caffe_gpu_add(output_saliencies_channel_.count(), output_saliencies_channel_.mutable_gpu_data(), this->blobs_[this->saliency_pos_]->mutable_gpu_data(), this->blobs_[this->saliency_pos_]->mutable_gpu_data()); 
    }
    else {
      caffe_copy(output_saliencies_channel_.count(), output_saliencies_channel_.mutable_gpu_data(), this->blobs_[this->saliency_pos_]->mutable_gpu_data());
    }
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * fisher_info) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data(); 
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data); //get unscaled diff back

  caffe_gpu_sum(output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  
  caffe_gpu_powx(output_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);

  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, fisher_info);
  
  caffe_gpu_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * taylor) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data(); 
  
  caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(-1 * this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_gpu(output_saliencies_points_.count(2,4), output_saliency_data, taylor);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data(); 
  
  caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_data, output_saliency_data);
  caffe_gpu_mul(output_saliencies_points_.count(), output_saliency_data, act_ddiff, output_saliency_data);
  
  caffe_gpu_scal(output_saliencies_points_.count(), 1/(Dtype)(2), output_saliency_data);

  compute_norm_and_batch_avg_gpu(output_saliencies_points_.count(2,4), output_saliency_data, hessian_diag);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_approx2_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  
  caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_gpu_powx(output_saliencies_points_.count(), output_saliency_data, (Dtype)2, output_saliency_data);
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(this->num_ * this->num_ / 2), output_saliency_data);
  

  compute_norm_and_batch_avg_gpu(output_saliencies_points_.count(2,4), output_saliency_data, hessian_diag);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  
  caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_ddiff, output_saliency_data); //a * d2E/da2
  caffe_gpu_scal(output_saliencies_points_.count(), 1/(Dtype)(2*(this->num_)), output_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_gpu_sub(output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * d2E/da2) - 1/N * dE/da 
  caffe_gpu_mul(output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * d2E/da2) - a/N*dE/da
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_gpu(output_saliencies_points_.count(2,4), output_saliency_data, taylor_2nd);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_approx2_gpu(const Dtype *  act_data, const Dtype * act_diff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  
  caffe_gpu_mul(output_saliencies_points_.count(), act_data, act_diff, output_saliency_data); //a * 1/N *dE/da
  caffe_gpu_mul(output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //a * 1/N *  1/N * (dE/da)**2
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(this->num_ / 2), output_saliency_data);  //1/2N * (a * (dE/da2)**2)
  caffe_gpu_sub(output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * (dE/da2)**2) - 1/N * dE/da 
  caffe_gpu_mul(output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * (dE/da2)**2) - a/N*dE/da
  caffe_gpu_scal(output_saliencies_points_.count(), (Dtype)(this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_gpu(output_saliencies_points_.count(2,4), output_saliency_data, taylor_2nd);
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_weight_gpu(const Dtype *  act_data, Dtype * saliency_info) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  
  caffe_copy(output_saliencies_points_.count(), act_data, output_saliency_data);

  compute_norm_and_batch_avg_gpu(output_saliencies_points_.count(2,4), output_saliency_data, saliency_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  const Dtype* weights_n_diff = weights_n->gpu_diff();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();
  
  const Dtype* bias;
  const Dtype* bias_n_diff;
  Dtype* bias_saliency_data;
  
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }

  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
  caffe_gpu_scal(weights_n->count(), (Dtype) this->num_, points_saliency_data); // get unscaled diff back
  
  caffe_gpu_sum(weights_n->count(0,2), weights_n->count(2, 5), points_saliency_data, filter_saliency_data);
  
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
    caffe_gpu_add(weights_n->count(0,2), points_saliency_data, bias_saliency_data, points_saliency_data);
  }
  
  caffe_gpu_powx(output_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);
  
  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, fisher_info);
  
  caffe_gpu_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor) {
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
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
  caffe_gpu_scal(weights_n->count(), (Dtype) (-1 * this->num_), points_saliency_data); // get unscaled diff back
  
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
    caffe_gpu_scal(bias_n->count(), (Dtype) (-1 * this->num_), bias_saliency_data); // get unscaled diff back
  }
  
  compute_norm_and_batch_avg_gpu(weights_n->count(2, 5), points_saliency_data, taylor, bias_saliency_data);
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * hessian_diag) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  const Dtype* weights_n_ddiff = weights_n->gpu_ddiff();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();
  
  const Dtype* bias;
  const Dtype* bias_n_ddiff;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }
  
  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_ddiff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
    caffe_gpu_mul(this->blobs_[0]->count(), weights, points_saliency_data + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
 
  caffe_gpu_scal(weights_n->count(), 1/(Dtype)(2), points_saliency_data);
  
  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    bias_n_ddiff = bias_n->gpu_ddiff();
    bias_saliency_data = bias_n->mutable_gpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_gpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_n_ddiff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_n_ddiff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), 1/(Dtype)(2), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_gpu(weights_n->count(2, 5), points_saliency_data, hessian_diag, bias_saliency_data);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * hessian_diag) {
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
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_diff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count());
  }
  caffe_gpu_powx(weights_n->count(), points_saliency_data, (Dtype) 2, points_saliency_data);

  caffe_gpu_scal(weights_n->count(), (Dtype)(this->num_ * this->num_ / 2), points_saliency_data);
  
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
    caffe_gpu_powx(bias_n->count(), bias_saliency_data, (Dtype) 2, bias_saliency_data);
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_ * this->num_ / 2), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_gpu(weights_n->count(2, 5), points_saliency_data, hessian_diag, bias_saliency_data);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_2nd) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  const Dtype* weights_n_diff = weights_n->gpu_diff();
  const Dtype* weights_n_ddiff = weights_n->gpu_ddiff();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();
  
  const Dtype* bias;
  const Dtype* bias_n_diff;
  const Dtype* bias_n_ddiff;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }
    
  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_ddiff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //a * d2E/da2
  }
  
  caffe_gpu_scal(weights_n->count(), 1/(Dtype)(2*(this->num_)), points_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_gpu_sub(weights_n->count(), points_saliency_data, weights_n_diff, points_saliency_data); //(a/2N * d2E/da2) - 1/N * dE/da 
  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, points_saliency_data + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //(a**2/2N * d2E/da2) - a/N*dE/da
  }
  caffe_gpu_scal(weights_n->count(), (Dtype)(this->num_), points_saliency_data);
  
  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    bias_n_diff = bias_n->gpu_diff();
    bias_n_ddiff = bias_n->gpu_ddiff();
    bias_saliency_data = bias_n->mutable_gpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_gpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_n_ddiff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), 1/(Dtype)(2*(this->num_)), bias_saliency_data);
    caffe_gpu_sub(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_saliency_data + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_gpu(weights_n->count(2, 5), points_saliency_data, taylor_2nd, bias_saliency_data);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * taylor_2nd) {
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
  caffe_gpu_scal(weights_n->count(), (Dtype)(this->num_ / 2), points_saliency_data);  //1/2N * (a * d2E/da2)
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
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_ / 2), bias_saliency_data);
    caffe_gpu_sub(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_saliency_data + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_), bias_saliency_data);
  }
  
  compute_norm_and_batch_avg_gpu(weights_n->count(2, 5), points_saliency_data, taylor_2nd, bias_saliency_data);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_weight_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();
  
  const Dtype* bias;
  Dtype* bias_saliency_data;

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
  
  switch (this->saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_gpu_abs(this->blobs_[0]->count(), weights, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_abs(this->blobs_[1]->count(), bias, bias_saliency_data);
      }
    } break;
    
    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_gpu_powx(this->blobs_[0]->count(), weights, (Dtype) 2, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_powx(this->blobs_[1]->count(), bias, (Dtype) 2, bias_saliency_data);
      }
    } break;
  
    default: {
      caffe_copy(this->blobs_[0]->count(), weights, points_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_copy(this->blobs_[1]->count(), bias, bias_saliency_data);
      }
    } break;
  }

  caffe_gpu_sum(this->num_output_, this->blobs_[0]->count(1,4), points_saliency_data, saliency_info); //sum hxw
  if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
    caffe_gpu_add(this->num_output_, bias_saliency_data, saliency_info, saliency_info);
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_norm_and_batch_avg_gpu(int count, Dtype * output_saliency_data, Dtype * saliency_data, Dtype * bias_saliency_data) {

  Dtype* filter_saliency_data = this->output_saliencies_filter_.mutable_gpu_data();    

  switch (this->saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_gpu_abs(this->num_ * this->num_output_ * count, output_saliency_data, output_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_abs(this->num_ * this->num_output_, bias_saliency_data, bias_saliency_data);
      }
    } break;
    
    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_gpu_powx(this->num_ * this->num_output_ * count, output_saliency_data, (Dtype) 2, output_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_gpu_powx(this->num_ * this->num_output_, bias_saliency_data, (Dtype) 2, bias_saliency_data);
      }
    } break;
  
    default: {
    } break;
  }
  
  caffe_gpu_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_saliency_data); //sum hxw
  if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
    caffe_gpu_add(this->num_ * this->num_output_, bias_saliency_data, filter_saliency_data, filter_saliency_data);
  }
  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, saliency_data);
  caffe_gpu_scal(this->num_output_, 1 / (Dtype)(this->num_), saliency_data);

}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionSaliencyLayer);
#endif


INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionSaliencyLayer);

}  // namespace caffe
