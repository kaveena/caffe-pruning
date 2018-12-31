#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_saliency_layer.hpp"

namespace caffe {


template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
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
  if (this->mask_term_) {
    weight = this->weights_masked_.gpu_data();
  }
  caffe_gpu_powx(this->blobs_[0]->count(), weight, (Dtype)2, weights_sqr);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* top_data = top[i]->gpu_data();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    const Dtype* top_ddiff;
    Dtype* bottom_ddiff;
    if (this->phase_ == TEST) {
      bottom_ddiff = bottom[i]->mutable_gpu_ddiff();
      top_ddiff = top[i]->gpu_ddiff();
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
      if (this->mask_term_) {
        caffe_gpu_mul(this->blobs_[1]->count(), this->blobs_[this->mask_pos_+1]->gpu_data(), this->blobs_[1]->mutable_gpu_diff(), this->blobs_[1]->mutable_gpu_diff());
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
          if (this->mask_term_) {
            // Don't update weights that are masked off
            caffe_gpu_mul(this->blobs_[0]->count(), this->blobs_[this->mask_pos_]->gpu_data(), weight_diff, weight_diff);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (this->phase_ == TEST) {
          if (propagate_down[i]) {
            this->backward_gpu_gemm(top_ddiff + n * this->top_dim_, weights_sqr,
                bottom_ddiff + n * this->bottom_dim_);
          }
        }
      }
    }

    if (this->phase_ == TEST) {
      Dtype* channel_saliency_data = output_saliencies_channel_.mutable_gpu_data();    
  
      switch (this->layer_param_.convolution_saliency_param().saliency()) {
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
          compute_fisher_gpu(top_data, top_diff, channel_saliency_data);
          compute_taylor_gpu(top_data, top_diff, channel_saliency_data + this->num_output_);
          compute_hessian_diag_gpu(top_data, top_diff, top_ddiff, channel_saliency_data + (2*this->num_output_));
          compute_hessian_diag_approx2_gpu(top_data, top_diff, channel_saliency_data + (3*this->num_output_));
          compute_taylor_2nd_gpu(top_data, top_diff, top_ddiff, channel_saliency_data + (4*this->num_output_));
          compute_taylor_2nd_approx2_gpu(top_data, top_diff, channel_saliency_data + (5*this->num_output_));
        } break;

        default: {
        } break;
      }
      if (this->layer_param_.convolution_saliency_param().accum()) {
        caffe_gpu_add(output_saliencies_channel_.count(), output_saliencies_channel_.mutable_gpu_data(), this->blobs_[this->saliency_pos_]->mutable_gpu_data(), this->blobs_[this->saliency_pos_]->mutable_gpu_data()); 
      }
      else {
        caffe_copy(output_saliencies_channel_.count(), output_saliencies_channel_.mutable_gpu_data(), this->blobs_[this->saliency_pos_]->mutable_gpu_data());
      }
    }
  }
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_fisher_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * fisher_info) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  caffe_gpu_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_gpu_scal(this->output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data); //get unscaled diff back

  caffe_gpu_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  
  caffe_gpu_powx(this->output_saliencies_filter_.count(), filter_saliency_data, (Dtype)2, filter_saliency_data);

  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, fisher_info); // functionally it does not matter if we use sum or asum; sum across batches
  
  caffe_gpu_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), fisher_info);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * taylor) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  caffe_gpu_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  
  caffe_gpu_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, taylor); // functionally it does not matter if we use sum or asum; sum across batches
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  caffe_gpu_mul(this->output_saliencies_points_.count(), act_data, act_data, output_saliency_data);
  caffe_gpu_mul(this->output_saliencies_points_.count(), output_saliency_data, act_ddiff, output_saliency_data);
  
  caffe_gpu_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, hessian_diag); // functionally it does not matter if we use sum or asum; sum across batches
  
  caffe_gpu_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), hessian_diag);
  
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_hessian_diag_approx2_gpu(const Dtype *  act_data, const Dtype *  act_diff, Dtype * hessian_diag) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  caffe_gpu_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data);
  caffe_gpu_scal(this->output_saliencies_points_.count(), (Dtype) this->num_, output_saliency_data); //get unscaled diff back
  caffe_gpu_powx(this->output_saliencies_points_.count(), output_saliency_data, (Dtype)2, output_saliency_data);
  
  caffe_gpu_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, hessian_diag); // functionally it does not matter if we use sum or asum; sum across batches
  
  caffe_gpu_scal(this->num_output_, 1/(Dtype)(2*(this->num_)), hessian_diag);
}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  caffe_gpu_mul(this->output_saliencies_points_.count(), act_data, act_ddiff, output_saliency_data); //a * d2E/da2
  caffe_gpu_scal(this->output_saliencies_points_.count(), 1/(Dtype)(2*(this->num_)), output_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_gpu_add(this->output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * d2E/da2) + 1/N * dE/da 
  caffe_gpu_mul(this->output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * d2E/da2) + a/N*dE/da
  
  caffe_gpu_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, taylor_2nd);

}

template <typename Dtype>
void ConvolutionSaliencyLayer<Dtype>::compute_taylor_2nd_approx2_gpu(const Dtype *  act_data, const Dtype * act_diff, Dtype * taylor_2nd) {
  Dtype* output_saliency_data = output_saliencies_points_.mutable_gpu_data();    
  Dtype* filter_saliency_data = output_saliencies_filter_.mutable_gpu_data();    
  
  caffe_gpu_mul(this->output_saliencies_points_.count(), act_data, act_diff, output_saliency_data); //a * dE/da
  caffe_gpu_mul(this->output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //a * (dE/da)**2
  caffe_gpu_scal(this->output_saliencies_points_.count(), 1/(Dtype)(2*(this->num_)), output_saliency_data);  //1/2N * (a * (dE/da2)**2)
  caffe_gpu_add(this->output_saliencies_points_.count(), output_saliency_data, act_diff, output_saliency_data); //(a/2N * (dE/da2)**2) + 1/N * dE/da 
  caffe_gpu_mul(this->output_saliencies_points_.count(), output_saliency_data, act_data, output_saliency_data); //(a**2/2N * (dE/da2)**2) + a/N*dE/da
  
  caffe_gpu_sum(this->output_saliencies_points_.count(0, 2), output_saliencies_points_.count(2,4), output_saliency_data, filter_saliency_data); //sum hxw
  caffe_gpu_strided_sum(this->num_output_, this->num_, filter_saliency_data, taylor_2nd);
  
}



#ifdef CPU_ONLY
STUB_GPU(ConvolutionSaliencyLayer);
#endif


INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionSaliencyLayer);

}  // namespace caffe
