#ifndef CAFFE_SALIENCY_COMPUTATION_HPP_
#define CAFFE_SALIENCY_COMPUTATION_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void caffe_compute_taylor_cpu(const Dtype * data, const Dtype * diff, const int num, const int count, Dtype * saliencies_points);

template <typename Dtype>
void compute_taylor_weights_separate_diff_cpu(const Dtype * weights, const Dtype * weights_n_diff, const int num, const int count, Dtype * points_saliency_data);
}

#endif
