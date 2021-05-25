#ifndef CAFFE_GELU_LAYER_HPP_
#define CAFFE_GELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Gaussian  Error Linear Unit non-linearity @f$ 
 * y = \frac{x}{2}(1 + tanh(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3))) 
 * @f$.  This is the approximate implementation of GELU For details see
 * https://arxiv.org/abs/1606.08415 
 */

template <typename Dtype>
class GELULayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides GELUParameter gelu_param
   *
   */
   explicit GELULayer (const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}
   virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GELU"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *      y = \frac{x}{2}(1 + tanh(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3))) 
   *      @f$ by default.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the sigmoid inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *      y = \frac{1}{2}(1 + tanh(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3))   
   *           + x \sqrt{\frac{2}{\pi}} (1 + 0.134145x^2 )(1-tanh^2(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)))) 
   *      @f$ if propagate_down[0]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  // helper blobs for gradient computation
  Blob<Dtype> tanhx;
};

}  // namespace caffe

#endif  // CAFFE_GELU_LAYER_HPP_
