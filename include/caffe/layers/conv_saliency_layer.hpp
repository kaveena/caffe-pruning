#ifndef CAFFE_CONV_SALIENCY_LAYER_HPP_
#define CAFFE_CONV_SALIENCY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class ConvolutionSaliencyLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionMaskedLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - dilation (\b optional, default 1). The filter
   *  dilation, given by dilation_size for equal dimensions for different
   *  dilation. By default the convolution has dilation 1.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit ConvolutionSaliencyLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "ConvolutionSaliency"; }

 protected:
  /// @brief The spatial dimensions of the weights_masked_.
  vector<int> weights_masked_shape_;
  vector<int> bias_masked_shape_;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

 private:
  Blob<Dtype> weights_masked_;
  Blob<Dtype> bias_masked_;
  // Helpers for channel saliency
  Blob<Dtype> output_saliencies_channel_;
  Blob<Dtype> output_saliencies_points_;
  Blob<Dtype> output_saliencies_filter_;
  
  Blob<Dtype> input_saliencies_channel_;
  Blob<Dtype> input_saliencies_points_;
  Blob<Dtype> input_saliencies_filter_;
  
  void compute_fisher_cpu(const Dtype *  act_data, const Dtype *  act_diff, const Dtype * input_data, const Dtype * input_diff,  Dtype * fisher_info_out, Dtype * fisher_info_in);
  void compute_fisher_gpu(const Dtype *  act_data, const Dtype *  act_diff, const Dtype * input_data, const Dtype * input_diff,  Dtype * fisher_info_out, Dtype * fisher_info_in);
  void compute_taylor_cpu(const Dtype * act_data, const Dtype * act_diff, Dtype * taylor);
  void compute_taylor_gpu(const Dtype * act_data, const Dtype * act_diff, Dtype * taylor);
  void compute_hessian_diag_cpu(const Dtype * act_data, const Dtype * act_diff, const Dtype * act_ddiff, Dtype * hessian_diag);
  void compute_hessian_diag_gpu(const Dtype * act_data, const Dtype * act_diff, const Dtype * act_ddiff, Dtype * hessian_diag);
  void compute_hessian_diag_approx2_cpu(const Dtype * act_data, const Dtype * act_diff, Dtype * hessian_diag);
  void compute_hessian_diag_approx2_gpu(const Dtype * act_data, const Dtype * act_diff, Dtype * hessian_diag);
  void compute_taylor_2nd_cpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * taylor_2nd);
  void compute_taylor_2nd_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, Dtype * taylor_2nd);
  void compute_taylor_2nd_approx2_cpu(const Dtype *  act_data, const Dtype * act_diff, Dtype * taylor_2nd);
  void compute_taylor_2nd_approx2_gpu(const Dtype *  act_data, const Dtype * act_diff, Dtype * taylor_2nd);
  void compute_norm_and_batch_avg_cpu(int count, Dtype * output_saliency_data, Dtype * saliency_data, Dtype * bias_saliency_data=NULL);
  void compute_norm_and_batch_avg_gpu(int count, Dtype * output_saliency_data, Dtype * saliency_data, Dtype * bias_saliency_data=NULL);
  void compute_fisher_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info_out, Dtype * fisher_info_in);
  void compute_fisher_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info_out, Dtype * fisher_info_in);
  void compute_taylor_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_taylor_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_hessian_diag_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_hessian_diag_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_hessian_diag_approx2_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_hessian_diag_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_taylor_2nd_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_taylor_2nd_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_taylor_2nd_approx2_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_taylor_2nd_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * fisher_info);
  void compute_weight_avg_cpu(const Dtype * act_data, Dtype * saliency_info);
  void compute_weight_avg_gpu(const Dtype * act_data, Dtype * saliency_info);
  void compute_diff_avg_cpu(const Dtype * act_diff, Dtype * saliency_info);
  void compute_diff_avg_gpu(const Dtype * act_diff, Dtype * saliency_info);
  void compute_weight_avg_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info);
  void compute_weight_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info);
  void compute_diff_avg_weights_cpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info);
  void compute_diff_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, Dtype * saliency_info);

};

}  // namespace caffe

#endif  // CAFFE_CONV_SALIENCY_LAYER_HPP_