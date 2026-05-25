#include "foundationpose_depth_preprocessor.hpp"

#include <array>
#include <cstdint>

#include "foundationpose_sampling.cu.hpp"
#include "foundationpose_utils.hpp"

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {

namespace {
constexpr char kNamePoints[] = "points";
constexpr char RAW_CAMERA_MODEL_GXF_NAME[] = "intrinsics";
}  // namespace

gxf_result_t FoundationposeDepthPreprocessor::registerInterface(
    gxf::Registrar* registrar) noexcept {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      depth_receiver_,
      "depth_input",
      "Depth Input",
      "Depth image input with camera model");

  result &= registrar->parameter(
      point_cloud_transmitter_,
      "point_cloud_output",
      "Point Cloud Output",
      "Filtered XYZ output tensor");

  result &= registrar->parameter(
      allocator_,
      "allocator",
      "Allocator",
      "Allocator for filtered XYZ tensor");

  result &= registrar->parameter(
      cuda_stream_pool_,
      "cuda_stream_pool",
      "CUDA Stream Pool",
      "CUDA stream pool");

  return gxf::ToResultCode(result);
}

gxf_result_t FoundationposeDepthPreprocessor::start() noexcept {
  auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
  if (!maybe_stream) {
    return gxf::ToResultCode(maybe_stream);
  }

  cuda_stream_handle_ = std::move(maybe_stream.value());

  if (!cuda_stream_handle_->stream()) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] CUDA stream is not initialized");
    return GXF_FAILURE;
  }

  cuda_stream_ = cuda_stream_handle_->stream().value();

  return GXF_SUCCESS;
}

gxf_result_t FoundationposeDepthPreprocessor::tick() noexcept {
  auto maybe_depth_message = depth_receiver_->receive();
  if (!maybe_depth_message) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Failed to receive depth message");
    return maybe_depth_message.error();
  }

  auto depth_message = maybe_depth_message.value();

  auto maybe_depth_image = depth_message.get<gxf::VideoBuffer>();
  if (!maybe_depth_image) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Failed to get depth VideoBuffer");
    return maybe_depth_image.error();
  }

  auto depth_handle = maybe_depth_image.value();
  auto depth_info = depth_handle->video_frame_info();

  if (depth_info.color_planes[0].stride != depth_info.width * sizeof(float)) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Expected float depth image without padding");
    return GXF_FAILURE;
  }

  auto maybe_camera_model = depth_message.get<nvidia::gxf::CameraModel>(
      RAW_CAMERA_MODEL_GXF_NAME);
  if (!maybe_camera_model) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Failed to get CameraModel");
    return maybe_camera_model.error();
  }

  auto camera_model = maybe_camera_model.value();

  const uint32_t height = depth_info.height;
  const uint32_t width = depth_info.width;

  if (width != cached_width_ || height != cached_height_) {
    if (erode_depth_device_ != nullptr) {
      CHECK_CUDA_ERRORS(cudaFree(erode_depth_device_));
      erode_depth_device_ = nullptr;
    }

    if (bilateral_filter_depth_device_ != nullptr) {
      CHECK_CUDA_ERRORS(cudaFree(bilateral_filter_depth_device_));
      bilateral_filter_depth_device_ = nullptr;
    }

    if (filtered_xyz_device_ != nullptr) {
      CHECK_CUDA_ERRORS(cudaFree(filtered_xyz_device_));
      filtered_xyz_device_ = nullptr;
    }

    CHECK_CUDA_ERRORS(cudaMalloc(
        &erode_depth_device_,
        width * height * sizeof(float)));

    CHECK_CUDA_ERRORS(cudaMalloc(
        &bilateral_filter_depth_device_,
        width * height * sizeof(float)));

    CHECK_CUDA_ERRORS(cudaMalloc(
        &filtered_xyz_device_,
        width * height * 3 * sizeof(float)));

    cached_width_ = width;
    cached_height_ = height;
  }

  erode_depth(
      cuda_stream_,
      reinterpret_cast<float*>(depth_handle->pointer()),
      erode_depth_device_,
      height,
      width);

  CHECK_CUDA_ERRORS(cudaGetLastError());

  bilateral_filter_depth(
      cuda_stream_,
      erode_depth_device_,
      bilateral_filter_depth_device_,
      height,
      width);

  CHECK_CUDA_ERRORS(cudaGetLastError());

  depth_to_xyz(
      cuda_stream_,
      bilateral_filter_depth_device_,
      filtered_xyz_device_,
      height,
      width,
      camera_model->focal_length.x,
      camera_model->focal_length.y,
      camera_model->principal_point.x,
      camera_model->principal_point.y);

  CHECK_CUDA_ERRORS(cudaGetLastError());

  auto maybe_output_message = gxf::Entity::New(context());
  if (!maybe_output_message) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Failed to create output entity");
    return gxf::ToResultCode(maybe_output_message);
  }

  auto output_message = maybe_output_message.value();

  auto maybe_added_timestamp = AddInputTimestampToOutput(output_message, depth_message);
  if (!maybe_added_timestamp) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Failed to add timestamp");
    return gxf::ToResultCode(maybe_added_timestamp);
  }

  auto maybe_points = output_message.add<gxf::Tensor>(kNamePoints);
  if (!maybe_points) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Failed to add points tensor");
    return gxf::ToResultCode(maybe_points);
  }

  auto points = maybe_points.value();

  std::array<int32_t, nvidia::gxf::Shape::kMaxRank> xyz_shape{
      static_cast<int32_t>(height),
      static_cast<int32_t>(width),
      3};

  auto reshape_result = points->reshape<float>(
      nvidia::gxf::Shape{xyz_shape, 3},
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator_);

  if (!reshape_result) {
    GXF_LOG_ERROR("[FoundationposeDepthPreprocessor] Failed to reshape points tensor");
    return gxf::ToResultCode(reshape_result);
  }

  CHECK_CUDA_ERRORS(cudaMemcpyAsync(
      points->pointer(),
      filtered_xyz_device_,
      height * width * 3 * sizeof(float),
      cudaMemcpyDeviceToDevice,
      cuda_stream_));

  CHECK_CUDA_ERRORS(cudaStreamSynchronize(cuda_stream_));

  return gxf::ToResultCode(
      point_cloud_transmitter_->publish(std::move(output_message)));
}

gxf_result_t FoundationposeDepthPreprocessor::stop() noexcept {
  if (erode_depth_device_ != nullptr) {
    CHECK_CUDA_ERRORS(cudaFree(erode_depth_device_));
    erode_depth_device_ = nullptr;
  }

  if (bilateral_filter_depth_device_ != nullptr) {
    CHECK_CUDA_ERRORS(cudaFree(bilateral_filter_depth_device_));
    bilateral_filter_depth_device_ = nullptr;
  }

  if (filtered_xyz_device_ != nullptr) {
    CHECK_CUDA_ERRORS(cudaFree(filtered_xyz_device_));
    filtered_xyz_device_ = nullptr;
  }

  return GXF_SUCCESS;
}

}  // namespace isaac_ros
}  // namespace nvidia