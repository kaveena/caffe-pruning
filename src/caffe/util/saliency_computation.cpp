#include "caffe/util/saliency_computation.hpp"

namespace caffe {

template <typename Dtype>
void caffe_compute_taylor_cpu(const Dtype * data, const Dtype * diff, const int num, const int count, Dtype * saliencies_points) {
  caffe_mul(count, data, diff, saliencies_points);
  caffe_scal(count, (Dtype) (-1 * num), saliencies_points);
}
template void caffe_compute_taylor_cpu<float>(const float * data, const float * diff, const int num, const int count, float * saliencies_points);
template void caffe_compute_taylor_cpu<double>(const double * data, const double * diff, const int num, const int count, double * saliencies_points);

template <typename Dtype>
void compute_taylor_weights_separate_diff_cpu(const Dtype * weights, const Dtype * weights_n_diff, const int num, const int count, Dtype * points_saliency_data) {
  for (int n = 0; n<num; n++) {
    caffe_mul(count, weights, weights_n_diff + n * count, points_saliency_data + n * count);
  }
  caffe_scal(count * num, (Dtype) (-1 * num), points_saliency_data);
}
template void compute_taylor_weights_separate_diff_cpu<float>(const float * weights, const float * weights_n_diff, const int num, const int count, float * points_saliency_data);
template void compute_taylor_weights_separate_diff_cpu<double>(const double * weights, const double * weights_n_diff, const int num, const int count, double * points_saliency_data);

}
