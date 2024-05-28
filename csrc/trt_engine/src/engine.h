/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <glog/logging.h>



namespace paddle {
namespace inference {
namespace tensorrt {

namespace plugin {
class PluginTensorRT;
}  // namespace plugin

#define PADDLE_ENFORCE_NOT_NULL(obj, msg)                      \
    do {                                                       \
        if ((obj) == nullptr) {                                \
            std::ostringstream oss;                            \
            oss << "Null pointer exception: " << (msg);        \
            throw std::invalid_argument(oss.str());            \
        }                                                      \
    } while (0)

using ShapeMapType = std::map<std::string, std::vector<int>>;

class NaiveLogger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override {
    switch (severity) {
      case Severity::kVERBOSE:
        VLOG(3) << msg;
        break;
      case Severity::kINFO:
        VLOG(2) << msg;
        break;
      case Severity::kWARNING:
        LOG(WARNING) << msg;
        break;
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        LOG(ERROR) << msg;
        break;
      default:
        break;
    }
  }

  static nvinfer1::ILogger& Global() {
    static nvinfer1::ILogger* x = new NaiveLogger;
    return *x;
  }

  ~NaiveLogger() override {}
};

template <typename T>
struct Destroyer {
  void operator()(T* x) {
    if (x) {
      x->destroy();
    }
  }
};
template <typename T>
using infer_ptr = std::unique_ptr<T, Destroyer<T>>;
using PredictorID = int;

class TRTInt8Calibrator;

struct ConstructionParams {
  // The max batch size.
  int32_t max_batch_size;

  // The max memory size the engine uses.
  int64_t max_workspace_size;

  // // The precision of engine.
  // phi::DataType precision{phi::DataType::FLOAT32};

  int device_id{0};

  bool with_dynamic_shape{false};

  ShapeMapType min_input_shape;
  ShapeMapType max_input_shape;
  ShapeMapType optim_input_shape;
  ShapeMapType min_shape_tensor;
  ShapeMapType max_shape_tensor;
  ShapeMapType optim_shape_tensor;

  bool use_inspector{false};
  std::string engine_info_path{""};

  int optimization_level{3};
};

// Weight is model parameter.
class Weight {
public:
  Weight() { w_ = nvinfer1::Weights{}; }
  Weight(nvinfer1::DataType dtype, void* value, size_t num_elem) {
    w_.type = dtype;
    w_.values = value;
    w_.count = num_elem;
  }
  const nvinfer1::Weights& get() { return w_; }

  // void SetDataType(nvinfer1::DataType type) { w_.type = type; }

  // void SetDataType(phi::DataType type);

  void SetValues(const void* values) { w_.values = values; }

  void SetCount(int64_t num) { w_.count = num; }

  std::vector<int64_t> dims;

private:
  nvinfer1::Weights w_;
};


class TensorRTEngine {
  

public:
  TensorRTEngine(const ConstructionParams& params,
                 nvinfer1::ILogger& logger = NaiveLogger::Global())
      : params_(params), logger_(logger) {
        TensorRTEngine::predictor_id_per_thread = 0;
  }

  void ClearTensorMap() { itensor_map_.clear(); }

  inline nvinfer1::ICudaEngine* engine() { return infer_engine_.get(); }

  nvinfer1::IExecutionContext* context() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (infer_context_.find(predictor_id_per_thread) == infer_context_.end()) {
      nvinfer1::IExecutionContext* infer_context = infer_engine_->createExecutionContext();
      PADDLE_ENFORCE_NOT_NULL(infer_context, 
        "TensorRT engine can not build execution context.");
      if (with_dynamic_shape()) {
        if (cur_profile_num_ > 0) {
          infer_context->setOptimizationProfile(cur_profile_num_);
        }
        profile_index_[predictor_id_per_thread] = cur_profile_num_;
        ++cur_profile_num_;
      }
      infer_context_[predictor_id_per_thread].reset(infer_context);
    }
    return infer_context_[predictor_id_per_thread].get();
  }

  int GetBindingsOffset() { return 0; }

  int GetNbBindings() { return binding_num_; }

  void ResetContext() {
    std::unique_lock<std::mutex> lock(mutex_);
    infer_context_[predictor_id_per_thread].reset(nullptr);
    infer_context_.erase(predictor_id_per_thread);
    cur_profile_num_ = 0;
  }

  nvinfer1::IHostMemory* Serialize() {
    ihost_memory_.reset(infer_engine_->serialize());
    return ihost_memory_.get();
  }

  void Deserialize(const std::string& engine_serialized_data){
    FreshDeviceId();
    infer_runtime_.reset(nvinfer1::createInferRuntime(logger_));
    infer_engine_.reset(infer_runtime_->deserializeCudaEngine(
      engine_serialized_data.c_str(), engine_serialized_data.size()));
    binding_num_ = infer_engine_->getNbBindings();
    if (params_.use_inspector) {
      LogEngineInfo(params_.engine_info_path);
    }
  }


  bool Enqueue(nvinfer1::IExecutionContext* context,
               std::vector<void*>* buffers,
               cudaStream_t stream) {
    bool ret = context->enqueueV2(buffers->data(), stream, nullptr);
    return ret;
  }

  nvinfer1::INetworkDefinition* network() { return infer_network_.get(); }

  ShapeMapType& min_input_shape() { return params_.min_input_shape; }
  ShapeMapType& max_input_shape() { return params_.max_input_shape; }
  ShapeMapType& optim_input_shape() { return params_.optim_input_shape; }
  ShapeMapType& min_shape_tensor() { return params_.min_shape_tensor; }
  ShapeMapType& max_shape_tensor() { return params_.max_shape_tensor; }
  ShapeMapType& optim_shape_tensor() { return params_.optim_shape_tensor; }

  bool AdjustDynamicShapeRange(const ShapeMapType& runtime_input_shape,
                               const ShapeMapType& runtime_shape_tensor,
                               std::vector<std::string>* changed,
                               std::vector<std::string>* tensor_changed);

  bool with_dynamic_shape() { return params_.with_dynamic_shape; }


private:
  void FreshDeviceId(){
    cudaSetDevice(params_.device_id);
  }


  void LogEngineInfo(const std::string& engine_info_path) {
    std::unique_ptr<nvinfer1::IEngineInspector> infer_inspector(
        infer_engine_->createEngineInspector());
    auto* infer_context = context();
    infer_inspector->setExecutionContext(infer_context);
    if (engine_info_path.empty()) {
      LOG(INFO) << "====== engine info ======";
      for (int i = 0; i < infer_engine_->getNbLayers(); ++i) {
        LOG(INFO) << infer_inspector->getLayerInformation(
            i, nvinfer1::LayerInformationFormat::kJSON);
      }
      LOG(INFO) << "====== engine info end ======";
    }
  }

  inline int device_id() { return params_.device_id; }

  ConstructionParams params_;
  int max_profile_num_{1};
  int cur_profile_num_{0};
  std::unordered_map<PredictorID, int> profile_index_;
  nvinfer1::ILogger& logger_;

  // max data size for the buffers.
  std::unordered_map<std::string /*name*/, nvinfer1::ITensor* /*ITensor*/>
      itensor_map_;

  // TensorRT related internal members
  infer_ptr<nvinfer1::IBuilder> infer_builder_;
  infer_ptr<nvinfer1::INetworkDefinition> infer_network_;
  infer_ptr<nvinfer1::IRuntime> infer_runtime_;
  infer_ptr<nvinfer1::ICudaEngine> infer_engine_;
  std::unordered_map<PredictorID, infer_ptr<nvinfer1::IExecutionContext>>
      infer_context_;
  infer_ptr<nvinfer1::IHostMemory> ihost_memory_;

  int binding_num_;
  infer_ptr<nvinfer1::IBuilderConfig> infer_builder_config_;
  std::vector<nvinfer1::IOptimizationProfile*> optim_profiles_;
  std::mutex mutex_;

public:
  thread_local static int predictor_id_per_thread;
};  // class TensorRTEngine


}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
