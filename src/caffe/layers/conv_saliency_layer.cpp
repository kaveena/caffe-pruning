#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_output_shape() {
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
void ConvolutionSaliencyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  if (this->layer_param_.convolution_saliency_param().saliency() == caffe::ConvolutionSaliencyParameter::ALL) {
    output_saliencies_channel_.Reshape({(int)(caffe::ConvolutionSaliencyParameter::ALL), this->num_output_});
  }
  else {
    output_saliencies_channel_.Reshape({1, this->num_output_});
  }
  output_saliencies_points_.Reshape(top[0]->shape()); //shape nchw
  output_saliencies_filter_.Reshape({this->num_, this->num_output_}); //shape nc
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void ConvolutionSaliencyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* weights_sqr = this->weights_sqr_.mutable_cpu_data();
  if (this->mask_term_) {
    weight = this->weights_masked_.cpu_data();
  }
  caffe_powx(this->blobs_[0]->count(), weight, (Dtype)2, weights_sqr);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* top_data = top[i]->cpu_data();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    const Dtype* top_ddiff;
    Dtype* bottom_ddiff;
    if (this->phase_ == TEST) {
      top_ddiff = top[i]->cpu_ddiff();
      bottom_ddiff = bottom[i]->mutable_cpu_ddiff();
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
      if (this->mask_term_) {
        caffe_mul(this->blobs_[1]->count(), this->blobs_[this->mask_pos_+1]->cpu_data(), this->blobs_[1]->mutable_cpu_diff(), this->blobs_[1]->mutable_cpu_diff());
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
          if (this->mask_term_) {
            // Don't update weights that are masked off
            caffe_mul(this->blobs_[0]->count(), this->blobs_[this->mask_pos_]->cpu_data(), weight_diff, weight_diff);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (this->phase_ == TEST) {
          if (propagate_down[i]) {
            this->backward_cpu_gemm(top_ddiff + n * this->top_dim_, weights_sqr,
                bottom_ddiff + n * this->bottom_dim_);
          }
        }
      }
    }

    if (this->phase_ == TEST) {
      Dtype* channel_saliency_data = output_saliencies_channel_.mutable_cpu_data();    
  
      switch (this->layer_param_.convolution_saliency_param().saliency()) {
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
          compute_fisher_cpu(top_data, top_diff, channel_saliency_data);
          compute_taylor_cpu(top_data, top_diff, channel_saliency_data + this->num_output_);
          compute_hessian_diag_cpu(top_data, top_diff, top_ddiff, channel_saliency_data + (2*this->num_output_));
          compute_hessian_diag_approx2_cpu(top_data, top_diff, channel_saliency_data + (3*this->num_output_));
          compute_taylor_2nd_cpu(top_data, top_diff, top_ddiff, channel_saliency_data + (4*this->num_output_));
          compute_taylor_2nd_approx2_cpu(top_data, top_diff, channel_saliency_data + (5*this->num_output_));
        } break;

        default: {
        } break;
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
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_cpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * fisher_info) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_scal(this->output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data); //get unscaled diff back
  
  caffe_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  
  caffe_powx(this->output_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);
  
  caffe_strided_sum(this->num_output_, this->num_, filter_saliency_data, fisher_info);
  
  caffe_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_cpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * taylor) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_scal(this->output_saliencies_points_.count(), (Dtype)(this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_cpu(taylor);
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_cpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(), act_data, act_data, output_saliency_data);
  caffe_mul(this->output_saliencies_points_.count(), output_saliency_data, act_ddiff, output_saliency_data);
  
  caffe_scal(this->output_saliencies_points_.count(), 1/(Dtype)(2), output_saliency_data);

  compute_norm_and_batch_avg_cpu(hessian_diag);
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_approx2_cpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_powx(this->output_saliencies_points_.count(), output_saliency_data, (Dtype)2, output_saliency_data);
  caffe_scal(this->output_saliencies_points_.count(), (Dtype)(this->num_ * this->num_ / 2), output_saliency_data);
  

  compute_norm_and_batch_avg_cpu(hessian_diag);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_cpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(), act_data, act_ddiff, output_saliency_data); //a * d2E/da2
  caffe_scal(this->output_saliencies_points_.count(), 1/(Dtype)(2*(this->num_)), output_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_add(this->output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * d2E/da2) + 1/N * dE/da 
  caffe_mul(this->output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * d2E/da2) + a/N*dE/da
  caffe_scal(this->output_saliencies_points_.count(), (Dtype)(this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_cpu(taylor_2nd);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_approx2_cpu(const Dtype *  act_data, const Dtype * act_diff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_cpu_data();    
  
  caffe_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data); //a * 1/N *dE/da
  caffe_mul(this->output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //a * 1/N *  1/N * (dE/da)**2
  caffe_scal(this->output_saliencies_points_.count(), (Dtype)(this->num_ / 2), output_saliency_data);  //1/2N * (a * (dE/da2)**2)
  caffe_add(this->output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * (dE/da2)**2) + 1/N * dE/da 
  caffe_mul(this->output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * (dE/da2)**2) + a/N*dE/da
  caffe_scal(this->output_saliencies_points_.count(), (Dtype)(this->num_), output_saliency_data);
  
  compute_norm_and_batch_avg_cpu(taylor_2nd);
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_norm_and_batch_avg_cpu(Dtype * saliency_data) {

  Dtype* output_saliency_data = this->output_saliencies_points_.mutable_cpu_data();    
  Dtype* filter_saliency_data = this->output_saliencies_filter_.mutable_cpu_data();    

  switch (this->layer_param_.convolution_saliency_param().norm()) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_abs(this->output_saliencies_points_.count(), output_saliency_data, output_saliency_data);
    } break;
    
    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_powx(this->output_saliencies_points_.count(), output_saliency_data, (Dtype) 2, output_saliency_data);
    } break;
  
    default: {
    } break;
  }
  
  caffe_sum(this->output_saliencies_points_.count(0, 2), this->output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  caffe_strided_sum(this->num_output_, this->num_, filter_saliency_data, saliency_data);
  caffe_scal(this->num_output_, 1 / (Dtype)(this->num_), saliency_data);

}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionSaliencyLayer);
#endif

INSTANTIATE_CLASS(ConvolutionSaliencyLayer);

}  // namespace caffe
