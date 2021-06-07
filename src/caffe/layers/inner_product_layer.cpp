#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  this->mask_term_ = this->layer_param_.inner_product_mask_param().mask_term();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  InnerProductSaliencyParameter inner_product_saliency_param = this->layer_param_.inner_product_saliency_param();
  this->saliency_term_ = inner_product_saliency_param.saliency_term();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // saliency shapes and positions
  int saliency_shape_0_ = 0;
  switch (inner_product_saliency_param.input_layer()) {
    case (caffe::InnerProductSaliencyParameter::NCHW): {
      this->saliency_out_num_ = this->N_; // m
      this->num_ = bottom[0]->shape(0); // n
      this->saliency_in_num_ = bottom[0]->shape(1); // c
      this->saliency_in_count_ = bottom[0]->count(2); // hw
    }
    case (caffe::InnerProductSaliencyParameter::NC): {
      this->saliency_out_num_ = this->N_; // m
      this->num_ = bottom[0]->shape(0); // n
      this->saliency_in_num_ = bottom[0]->shape(1); // c
      this->saliency_in_count_ = 1;
    }
  }
  int pos_output_channel_saliency;
  int pos_input_channel_saliency;
  vector<int> saliency_out_shape = {saliency_shape_0_, saliency_out_num_};
  vector<int> saliency_in_shape = {saliency_shape_0_, saliency_in_num_};
  if (this->saliency_term_) {
    // check if the correct number of pointwise saliency, saliency input and norm have been provided
    if (!((inner_product_saliency_param.saliency_size() == inner_product_saliency_param.saliency_input_size()) 
      && (inner_product_saliency_param.saliency_size() == inner_product_saliency_param.saliency_norm_size()))) {
      LOG(FATAL) << "saliency, saliency_input and saliency_norm for each saliency measure" ;
    }
    saliency_shape_0_ = inner_product_saliency_param.saliency_size();
    this->output_channel_saliency_compute_ = inner_product_saliency_param.output_channel_compute();
    this->input_channel_saliency_compute_ = inner_product_saliency_param.input_channel_compute();
    if (!(this->output_channel_saliency_compute_ || this->input_channel_saliency_compute_)){
      LOG(FATAL) << "Either output_channel_compute or input_channel_compute must be set if saliency_term is set" ;
    }
    if (this->bias_term_ && inner_product_saliency_param.saliency_bias()) {
      this->saliency_bias_ = true;
    }
    else {
      this->saliency_bias_ = false;
    }
//    this->separate_weight_diff_ = false;
//    for (int i_s = 0; i_s < inner_product_saliency_param.saliency_size(); i_s++){
//      if (inner_product_saliency_param.saliency_input(i_s) == caffe::InnerProductSaliencyParameter::WEIGHT) {
//        this->separate_weight_diff_ = true;
//      }
//    }
  }
  int total_blobs = 1;
  this->mask_pos_ = 1;
  this->saliency_pos_ = 1;
  if (this->bias_term_) {
    total_blobs++;
    this->mask_pos_++;
    this->saliency_pos_++;
  }
  if (this->mask_term_) {
    total_blobs++;
    this->saliency_pos_++;
    if (this->bias_term_) {
      total_blobs++;
      this->saliency_pos_++;
    }
    pos_output_channel_saliency = this->saliency_pos_;
    pos_input_channel_saliency = this->saliency_pos_;
    if (this->output_channel_saliency_compute_ && this->input_channel_saliency_compute_){
      pos_input_channel_saliency++;
    }
  }
  if (this->saliency_term_) {
    total_blobs++;
    if (this->output_channel_saliency_compute_ && this->input_channel_saliency_compute_){
      total_blobs++;
    }
  }
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(total_blobs);
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the bias term
    vector<int> bias_shape(1, N_);
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    // If necessary, initialize and fill the mask term
    if (this->mask_term_ && (weight_shape != this->blobs_[this->mask_pos_]->shape())) {
      LOG(INFO) << "Mask Initialization: " << this->layer_param_.name();
      this->blobs_[this->mask_pos_].reset(new Blob<Dtype>(weight_shape));
       if (this->layer_param_.inner_product_mask_param().default_init()) {
        Blob<Dtype> * mask_blob = this->blobs_[this->mask_pos_].get();
        for (int i=0; i<mask_blob->count(); ++i) {
          mask_blob->mutable_cpu_data()[i] = (Dtype)1.0;
        }
      }
      else {
        shared_ptr<Filler<Dtype> > mask_filler(GetFiller<Dtype>(
            this->layer_param_.inner_product_mask_param().mask_filler()));
        mask_filler->Fill(this->blobs_[this->mask_pos_].get());
      }
    }
    if (this->bias_term_ && this->mask_term_  && (bias_shape != this->blobs_[this->mask_pos_+1]->shape())) {
      this->blobs_[this->mask_pos_+1].reset(new Blob<Dtype>(bias_shape));
      if (this->layer_param_.inner_product_mask_param().default_init()) {
        Blob<Dtype> * mask_blob = this->blobs_[this->mask_pos_+1].get();
        for (int i=0; i<mask_blob->count(); ++i) {
          mask_blob->mutable_cpu_data()[i] = (Dtype)1.0;
        }
      }
      else {
        shared_ptr<Filler<Dtype> > mask_filler(GetFiller<Dtype>(
            this->layer_param_.inner_product_mask_param().mask_filler()));
        mask_filler->Fill(this->blobs_[this->mask_pos_+1].get());
      }
    }
    if (this->saliency_term_ && this->output_channel_saliency_compute_ && saliency_out_shape != this->blobs_[pos_output_channel_saliency]->shape()) {
      Blob<Dtype> saliency_out_shaped_blob(saliency_out_shape);
      LOG(FATAL) << "Incorrect saliency out shape: expected shape "
          << saliency_out_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[pos_output_channel_saliency]->shape_string();
      LOG(INFO) << "Saliency Initialization";
      this->blobs_[pos_output_channel_saliency].reset(new Blob<Dtype>(saliency_out_shape));
      Blob<Dtype> * saliency_out_blob = this->blobs_[pos_output_channel_saliency].get();
      for (int i=0; i<saliency_out_blob->count(); ++i) {
        saliency_out_blob->mutable_cpu_data()[i] = (Dtype)0.0;
      }
    }
    if (this->saliency_term_ && this->input_channel_saliency_compute_ && saliency_in_shape != this->blobs_[pos_input_channel_saliency]->shape()) {
      Blob<Dtype> saliency_in_shaped_blob(saliency_in_shape);
      LOG(FATAL) << "Incorrect saliency in shape: expected shape "
          << saliency_in_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[pos_input_channel_saliency]->shape_string();
      LOG(INFO) << "Saliency Initialization";
      this->blobs_[pos_input_channel_saliency].reset(new Blob<Dtype>(saliency_in_shape));
      Blob<Dtype> * saliency_in_blob = this->blobs_[pos_input_channel_saliency].get();
      for (int i=0; i<saliency_in_blob->count(); ++i) {
        saliency_in_blob->mutable_cpu_data()[i] = (Dtype)0.0;
      }
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->weights_sqr_.Reshape(this->blobs_[0]->shape());
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  //Set up the mask
  if (mask_term_) {
    this->weights_masked_shape_.clear();
    this->weights_masked_shape_.push_back(this->blobs_[this->mask_pos_]->count());
    this->weights_masked_.Reshape(this->weights_masked_shape_);
    if (bias_term_) {
      this->bias_masked_shape_.clear();
      this->bias_masked_shape_.push_back(this->blobs_[this->mask_pos_+1]->count());
      this->bias_masked_.Reshape(this->bias_masked_shape_);
    }
  }
  if (this->saliency_term_) {
    InnerProductSaliencyParameter inner_product_saliency_param = this->layer_param_.inner_product_saliency_param();
    if (this->output_channel_saliency_compute_) {
      output_saliencies_channel_.Reshape({(int)(inner_product_saliency_param.saliency_size()), this->N_});
    }
    if (this->input_channel_saliency_compute_) {
      input_saliencies_channel_.Reshape({(int)(inner_product_saliency_param.saliency_size()), this->saliency_in_num_});
    }
    bool need_weight_saliencies = false;
    bool need_fm_saliencies = false;
    for (int i_s = 0; i_s < inner_product_saliency_param.saliency_size(); i_s++){
      if (inner_product_saliency_param.saliency_input(i_s) == caffe::InnerProductSaliencyParameter::WEIGHT) {
        need_weight_saliencies = true;
      }
      if (inner_product_saliency_param.saliency_input(i_s) == caffe::InnerProductSaliencyParameter::ACTIVATION) {
        need_fm_saliencies = true;
      }
    }
    if (need_weight_saliencies) {
      weight_saliencies_points_.Reshape(this->blobs_[0]->shape());
      if (bias_term_) {
        bias_saliencies_points_.Reshape(this->blobs_[1]->shape());
      }
    }
    if (need_fm_saliencies) {
      if (this->output_channel_saliency_compute_) {
        output_saliencies_points_.Reshape(top[0]->shape()); //shape nmhw
      }
      if (this->input_channel_saliency_compute_) {
        input_saliencies_points_.Reshape(bottom[0]->shape()); //shape nchw
      }
    }
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias;
  if (this->mask_term_) {
    const Dtype* mask_weights = this->blobs_[this->mask_pos_]->cpu_data();
    Dtype* weight_masked = this->weights_masked_.mutable_cpu_data();
    caffe_mul(this->blobs_[0]->count(), mask_weights, weight, weight_masked);
    weight = this->weights_masked_.cpu_data();
    if (this->bias_term_) {
      Dtype* bias_masked = this->bias_masked_.mutable_cpu_data();
      const Dtype* mask_bias = this->blobs_[this->mask_pos_+1]->cpu_data();
      caffe_mul(this->blobs_[1]->count(), mask_bias, bias, bias_masked);
      bias = this->bias_masked_.cpu_data();
    }
  }
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        bias, (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    // ddloss = ddtop * (dtopdbottom**2)
    const Dtype* top_ddiff;
    if (this->layer_param_.compute_2nd_derivative()) {
      top_ddiff = top[0]->cpu_ddiff();
    }
    Dtype* weights_sqr = this->weights_sqr_.mutable_cpu_data();
    caffe_powx(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), (Dtype)2, weights_sqr);
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
      if (this->layer_param_.compute_2nd_derivative()) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
            M_, K_, N_,
            (Dtype)1., top_ddiff, weights_sqr,
            (Dtype)0., bottom[0]->mutable_cpu_ddiff());
      }
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
      if (this->layer_param_.compute_2nd_derivative()) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
            (Dtype)1., top_ddiff, weights_sqr,
            (Dtype)0., bottom[0]->mutable_cpu_ddiff());
        }
    }
    // Compute Channel saliency
    // MULTIPLE INPUTS NOT TREATED
    if (this->saliency_term_) {
      InnerProductSaliencyParameter inner_product_saliency_param = this->layer_param_.inner_product_saliency_param();
      const Dtype* weight = this->blobs_[0]->cpu_data();
      const Dtype* weight_diff = this->blobs_[0]->cpu_diff();
      const Dtype* top_data = top[0]->cpu_data();
      const Dtype* top_diff = top[0]->cpu_diff();
      const Dtype* bottom_data = bottom[0]->cpu_data();
      const Dtype* bottom_diff = bottom[0]->cpu_diff();
      const Dtype* bias;
      const Dtype* bias_diff;
      if (this->bias_term_) {
        bias = this->blobs_[1]->cpu_data();
        bias_diff = this->blobs_[1]->cpu_diff();
      }
      if (this->mask_term_) {
        weight = this->weights_masked_.cpu_data();
        if (this->bias_term_) {
          bias = this->bias_masked_.cpu_data();
        }
      }
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
        output_channel_saliency_data = output_saliencies_channel_.mutable_cpu_data();
        output_channel_saliency_accum_data = this->blobs_[pos_output_channel_saliency]->mutable_cpu_data();
      }
      if (this->input_channel_saliency_compute_){
        input_channel_saliency_data = input_saliencies_channel_.mutable_cpu_data();
        input_channel_saliency_accum_data = this->blobs_[pos_input_channel_saliency]->mutable_cpu_data();
      }
      for (int i_s = 0; i_s < inner_product_saliency_param.saliency_size(); i_s++) {
        if (this->output_channel_saliency_compute_){
          out_channel_saliency = output_channel_saliency_data + (i_s * this->saliency_out_num_);
        }
        if (this->input_channel_saliency_compute_){
          in_channel_saliency = input_channel_saliency_data + (i_s * (this->saliency_in_num_));
        }
        if (inner_product_saliency_param.saliency_input(i_s) == caffe::InnerProductSaliencyParameter::ACTIVATION) {
          if (inner_product_saliency_param.saliency(i_s) == caffe::InnerProductSaliencyParameter::TAYLOR){
            if (this->output_channel_saliency_compute_) {
              output_channel_saliency_data = output_saliencies_points_.mutable_cpu_data();
              caffe_compute_taylor_cpu(top_data, top_diff, this->num_, top[0]->count(), output_channel_saliency_data);
            }
            if (this->input_channel_saliency_compute_) {
              input_channel_saliency_data = input_saliencies_points_.mutable_cpu_data();
              caffe_compute_taylor_cpu(bottom_data, bottom_diff, this->num_, bottom[0]->count(), input_channel_saliency_data);
            }
          }
          compute_norm_and_batch_avg_cpu(output_channel_saliency_data, input_channel_saliency_data, inner_product_saliency_param.saliency_norm(i_s), out_channel_saliency, in_channel_saliency);
        }
        if (inner_product_saliency_param.saliency_input(i_s) == caffe::InnerProductSaliencyParameter::WEIGHT) {
          Dtype * weight_saliencies_points_ = this->weight_saliencies_points_.mutable_cpu_data();
          Dtype * bias_saliencies_points_ = this->bias_saliencies_points_.mutable_cpu_data();
          if (inner_product_saliency_param.saliency(i_s) == caffe::InnerProductSaliencyParameter::TAYLOR){
            caffe_compute_taylor_cpu(weight, weight_diff, 1, this->blobs_[0]->count(), weight_saliencies_points_);
            if (this->saliency_bias_ && this->bias_term_) {
              caffe_compute_taylor_cpu(bias, bias_diff, 1, this->blobs_[1]->count(), bias_saliencies_points_);
            }
          }
          compute_norm_and_batch_avg_weights_cpu(weight_saliencies_points_, bias_saliencies_points_, inner_product_saliency_param.saliency_norm(i_s), out_channel_saliency, in_channel_saliency);
        }
        if (this->layer_param_.convolution_saliency_param().accum()) {
          if (this->output_channel_saliency_compute_) {
            caffe_add(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data, output_channel_saliency_accum_data);
          }
          if (this->input_channel_saliency_compute_) {
            caffe_add(input_saliencies_channel_.count(), input_channel_saliency_data, input_channel_saliency_accum_data, input_channel_saliency_accum_data);
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
}

template <typename Dtype>
void InnerProductLayer<Dtype>::compute_norm_and_batch_avg_cpu(Dtype * out_saliency_data, Dtype * in_saliency_data, caffe::InnerProductSaliencyParameter::NORM saliency_norm_, Dtype * out_channel_saliency, Dtype * in_channel_saliency) {
  Dtype *saliency_data, *channel_saliency;
  if (this->input_channel_saliency_compute_) {
    saliency_data = in_saliency_data;
    channel_saliency = in_channel_saliency;
    switch (saliency_norm_) {
      case (caffe::InnerProductSaliencyParameter::L1): {
        caffe_abs(this->num_ * this->saliency_in_num_ * this->saliency_in_count_, saliency_data, saliency_data);
        if (this->layer_param().inner_product_saliency_param().input_layer() == InnerProductSaliencyParameter::NCHW) {
          caffe_sum(this->num_ * this->saliency_in_num_, this->saliency_in_count_, saliency_data, saliency_data);
        }
      }
      default: {
        if (this->layer_param().inner_product_saliency_param().input_layer() == InnerProductSaliencyParameter::NCHW) {
          caffe_sum(this->num_ * this->saliency_in_num_, this->saliency_in_count_, saliency_data, saliency_data);
        }
      }
    }
    caffe_strided_sum(this->saliency_in_num_, this->num_, saliency_data, saliency_data);
    caffe_scal(this->saliency_in_num_, (Dtype) (1.0) / (Dtype)(this->num_), channel_saliency);
  }
  if (this->output_channel_saliency_compute_) {
    saliency_data = out_saliency_data;
    channel_saliency = out_channel_saliency;
    switch (saliency_norm_) {
      case (caffe::InnerProductSaliencyParameter::L1): {
        caffe_abs(this->M_ * this->N_, saliency_data, saliency_data);
      }
      default: {
      }
    }
    caffe_strided_sum(this->saliency_out_num_, this->N_, saliency_data, channel_saliency);
    caffe_scal(this->saliency_out_num_, (Dtype) (1.0) / (Dtype)(this->num_), channel_saliency);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::compute_norm_and_batch_avg_weights_cpu(Dtype * weight_saliency_data, Dtype * bias_saliency_data, caffe::InnerProductSaliencyParameter::NORM saliency_norm_, Dtype * out_channel_saliency, Dtype * in_channel_saliency) {
  switch (saliency_norm_) {
    case (caffe::InnerProductSaliencyParameter::L1): {
      caffe_abs(this->blobs_[0]->count(), weight_saliency_data, weight_saliency_data);
    }
    default: {
    }
  }
  if (this->output_channel_saliency_compute_) {
    caffe_sum(this->N_, this->K_, weight_saliency_data, out_channel_saliency);
    if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL) {
      caffe_abs(this->blobs_[1]->count(), bias_saliency_data, bias_saliency_data);
      caffe_add(this->N_, bias_saliency_data, out_channel_saliency, out_channel_saliency);
    }
  }
  if (this->input_channel_saliency_compute_) {
    if (transpose_) {
      if (this->layer_param().inner_product_saliency_param().input_layer() == InnerProductSaliencyParameter::NCHW) {
        caffe_sum(this->K_, this->N_, weight_saliency_data, weight_saliency_data);
        caffe_strided_sum(this->saliency_in_num_, this->saliency_in_count_, weight_saliency_data, in_channel_saliency);
      }
      else {
        caffe_sum(this->K_, this->N_, weight_saliency_data, in_channel_saliency);
      }
    }
    else {
      if (this->layer_param().inner_product_saliency_param().input_layer() == InnerProductSaliencyParameter::NCHW) {
      caffe_strided_sum(this->K_, this->N_, weight_saliency_data, weight_saliency_data);
      caffe_sum(this->saliency_in_num_, this->saliency_in_count_, weight_saliency_data, in_channel_saliency);
    }
      else {
        caffe_strided_sum(this->K_, this->N_, weight_saliency_data, in_channel_saliency);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
