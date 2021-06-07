#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/saliency_computation.hpp"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  vector<int> weights_masked_shape_;
  vector<int> bias_masked_shape_;
  Blob<Dtype> weights_masked_;
  Blob<Dtype> bias_masked_;
  int mask_pos_;
  int saliency_pos_;
  bool saliency_bias_;
  bool output_channel_saliency_compute_;
  bool input_channel_saliency_compute_;
  Blob<Dtype> output_saliencies_channel_;
  Blob<Dtype> input_saliencies_channel_;
  // Helpers for channel saliency
  Blob<Dtype> output_saliencies_points_;
  Blob<Dtype> input_saliencies_points_;
  Blob<Dtype> weight_saliencies_points_;
  Blob<Dtype> bias_saliencies_points_;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void compute_norm_and_batch_avg_cpu(Dtype * out_saliency_data, Dtype * in_saliency_data, caffe::InnerProductSaliencyParameter::NORM saliency_norm_, Dtype * out_channel_saliency, Dtype * in_channel_saliency);
  void compute_norm_and_batch_avg_weights_cpu(Dtype * weight_saliency_data, Dtype * bias_saliency_data, caffe::InnerProductSaliencyParameter::NORM saliency_norm_, Dtype * out_channel_saliency, Dtype * in_channel_saliency);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  bool mask_term_;
  bool saliency_term_;
  int num_, saliency_out_num_, saliency_in_num_, saliency_in_count_;
  Blob<Dtype> bias_multiplier_;
  // Helper for computing ddiff
  Blob<Dtype> weights_sqr_;
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
