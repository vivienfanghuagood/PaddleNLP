// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <paddle/extension.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <fstream>
#include <sstream>
#include <string>

#include "engine.h"

constexpr static int NUM_INPUTS = 1;
constexpr static int NUM_OUTPUTS = 1;
using namespace paddle::inference::tensorrt;
class TRTEngineOp {
public:
  TRTEngineOp(const std::string& engine_file) {
    std::map<std::string, std::vector<int>> min_input_shape = {
        {"input", {1, 32}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"input", {18, 32}}};
    std::map<std::string, std::vector<int>> optim_input_shape = {
        {"input", {18, 32}}};
    std::map<std::string, std::vector<int>> min_input_value = {
        {"shape", {1, 8, 4}}};
    std::map<std::string, std::vector<int>> max_input_value = {
        {"shape", {18, 8, 4}}};
    std::map<std::string, std::vector<int>> optim_input_value = {
        {"shape", {18, 8, 4}}};
    ConstructionParams params;
    params.max_batch_size = 16;
    params.max_workspace_size = 1 << 10;
    params.with_dynamic_shape = true;
    params.min_input_shape = min_input_shape;
    params.max_input_shape = max_input_shape;
    params.optim_input_shape = optim_input_shape;
    params.min_shape_tensor = min_input_value;
    params.max_shape_tensor = max_input_value;
    params.optim_shape_tensor = optim_input_value;

    engine_ = std::make_unique<TensorRTEngine>(params, NaiveLogger::Global());

    {
      std::string engine_data = readBinaryFileToString(engine_file);
      engine_->Deserialize(engine_data);
      if (!engine_) {
        throw std::runtime_error("Failed to create engine.");
      }
    }
  }

  ~TRTEngineOp() {
    
  }
  
private:
  std::unique_ptr<TensorRTEngine> engine_;

public:
  static std::string readBinaryFileToString(const std::string& filePath) {
    std::ifstream inputFile(filePath, std::ios::binary);

    if (!inputFile) {
      throw std::runtime_error("Failed to open file: " + filePath);
    }

    inputFile.seekg(0, std::ios::end);
    std::streamsize fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    std::string fileContent(static_cast<size_t>(fileSize), '\0');

    if (!inputFile.read(&fileContent[0], fileSize)) {
      throw std::runtime_error("Failed to read file: " + filePath);
    }

    return fileContent;
  }

  static std::vector<std::string> splitString(const std::string& str,
                                              char delimiter) {
    std::vector<std::string> result;
    std::istringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
      if (!token.empty()) {
        result.push_back(token);
      }
    }

    return result;
  }

  template <typename T>
  inline nvinfer1::Dims Vec2TRT_Dims(const std::vector<T>& shape) {
    if (shape.size() == 4UL) {
      return nvinfer1::Dims4(shape[0], shape[1], shape[2], shape[3]);
    } else if (shape.size() == 3UL) {
      return nvinfer1::Dims3(shape[0], shape[1], shape[2]);
    }
    nvinfer1::Dims dims;
    dims.nbDims = shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
      dims.d[i] = shape[i];
    }
    return dims;
  };


  std::vector<paddle::Tensor> forward(
      const std::vector<paddle::Tensor>& inputs,
      const std::vector<std::string>& input_name_list) {

    int batch_size = 1;
    auto engine = engine_->engine();
    const int num_bindings = engine->getNbBindings();
    std::vector<void*> buffers(num_bindings, nullptr);
    if(inputs.size() != input_name_list.size()){
        throw std::runtime_error(
            "input size must be equal to input name list");
    }
    std::vector<paddle::Tensor> outputs{};

    auto place = inputs[0].place();
    auto trt_context = engine_->context();
    auto cu_stream = inputs[0].stream();

    for (int i = 0; i < num_bindings; ++i) {
      auto x_name = engine->getBindingName(i);
      VLOG(0) << "process " << x_name;
      size_t input_index =
          std::find(input_name_list.begin(), input_name_list.end(), x_name) -
          input_name_list.begin();
      if (input_index < 0) {
        throw std::runtime_error("input name not found in input_name_list");
      }

      if (engine->bindingIsInput(i)) {
        // const auto& input_name = input_name_list[i];
        auto input_dims = inputs[input_index].shape();
        auto trt_dims = Vec2TRT_Dims(input_dims);
        batch_size = input_dims[0];
        buffers[i] = const_cast<void*>(inputs[input_index].data());
        trt_context->setInputShape(x_name, trt_dims);
      } else {
        std::vector<int> ddim {};
        auto dims = engine->getTensorShape(x_name);

        int nb_dims = dims.nbDims;
        for (; nb_dims > 0; nb_dims--) {
          if (dims.d[nb_dims - 1] != 1)
            break;
        }
        for (int i = 0; i < nb_dims; i++) {
          auto d = dims.d[i] > 0? dims.d[i]: batch_size;
          ddim.push_back(d);
        }
        auto trt_type = engine->getTensorDataType(x_name);
        if (trt_type == nvinfer1::DataType::kFLOAT) {
          VLOG(0) << "output dim is  " << ddim[0] << ", " << ddim[1] << ", " << ddim[2] << ", " << ddim[3] ;
          auto output = paddle::empty(ddim, paddle::DataType::FLOAT32, place);
          auto output_ptr = reinterpret_cast<void*>(output.data<float>());
          outputs.emplace_back(output);
          buffers[i] = output_ptr;
        } else if (trt_type == nvinfer1::DataType::kHALF) {
          auto output = paddle::empty(ddim, paddle::DataType::FLOAT16, place);
          auto output_ptr = reinterpret_cast<void*>(output.data<paddle::float16>());
          outputs.emplace_back(output);
          buffers[i] = output_ptr;
        }
      }
    }
    engine_->Enqueue(trt_context, &buffers, cu_stream);
    return outputs;
  }

};

std::vector<paddle::Tensor> trt_engine_forward(
      const paddle::Tensor& input,
      const std::string& engine_file,
      const std::string& names) {
    static TRTEngineOp op(engine_file);
    std::vector<paddle::Tensor> inputs{input};
    auto input_name_list = TRTEngineOp::splitString(names, ';');
    return op.forward(inputs, input_name_list);
  }

PD_BUILD_OP(trtengine_op)
    .Inputs({"X"})
    .Attrs({"engine_file: std::string", "names: std::string"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(trt_engine_forward));