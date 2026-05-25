#ifndef NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DEPTH_PREPROCESSOR_HPP_
#define NVIDIA_ISAAC_ROS_EXTENSIONS_FOUNDATIONPOSE_DEPTH_PREPROCESSOR_HPP_

#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac_ros {

class FoundationposeDepthPreprocessor : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) noexcept override;
  gxf_result_t start() noexcept override;
  gxf_result_t tick() noexcept override;
  gxf_result_t stop() noexcept override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> depth_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> point_cloud_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;

  gxf::Handle<gxf::CudaStream> cuda_stream_handle_;
  cudaStream_t cuda_stream_{nullptr};

  float* erode_depth_device_{nullptr};
  float* bilateral_filter_depth_device_{nullptr};
  float* filtered_xyz_device_{nullptr};

  uint32_t cached_width_{0};
  uint32_t cached_height_{0};
};

}  // namespace isaac_ros
}  // namespace nvidia

#endif